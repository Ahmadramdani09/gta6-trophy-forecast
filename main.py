"""gta6-trophy-rarity-forecast-01 entry: a two-part adoption model.

The score is pinball loss on the logit scale, so this program thinks in logits
throughout.  Its central idea is that an Exophase earn rate is a ratio,

    rate = earners / players_tracked,

and the two sides move differently in launch week: the tracked pool grows in
waves (and the page only changes when Exophase refreshes it, so the observed
series is a step function), while a trophy's earners only ever accumulate as
players reach it.  Forecasting the ratio directly confuses dilution with
regression, so the model forecasts the two sides separately and divides.

Program flow per case:

  1. Collapse the read history into distinct page states (consecutive reads
     that print the same players and rates are one state of the page).
  2. A trophy pinned at 100% in every state stays at the top: tight quantiles.
  3. With at least four distinct states, take the last day of states and
     fit, on x = ln(1 + hours since the release midnight), a
     weighted least-squares line for ln(players) and one per trophy for
     ln(earners + 3); read both at the target and divide.  A direct logit
     trend of the rate over the same window cross-checks the division, the
     two are blended on the logit scale, and the implied move is shrunk
     toward the last observed rate.  Slopes, the pool's growth, the earners
     multiple and the share of the pool a trophy may claim are all capped:
     launch ignition must not extrapolate into nonsense.
  4. With fewer states the page has effectively never refreshed inside the
     window (the launch-night practice shape): no line can be fit, so fall
     back on a dilution prior - the pool multiplies several-fold at the next
     refresh while earners barely move - with a wide spread.
  5. Quantiles are logit-normal around the median: q_tau = sigmoid(logit(m)
     + z_tau * sigma), sigma growing with the extrapolation distance, the
     pool's forecast growth and the fit residuals.

Deterministic, standard library only, no network.  Reads every *.in file in
$MAKEFUN_INPUT_DIR (else /input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR
(else /output), in the case file order, so two runs agree byte for byte.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The release midnight the case's game names (its release_date at 00:00 UTC);
# every fit measures hours from here, so x = ln(1 + hours) is 0 at release.
FALLBACK_ANCHOR = datetime(2026, 11, 19, tzinfo=timezone.utc)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, whole numbers as ints (JSON-compact)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def quantiles(median_pct, sigma):
    """Five logit-normal quantiles around a median rate, monotone, in range."""
    centre = logit(median_pct)
    qs = []
    for z in Z:
        q = 100.0 * sigmoid(centre + z * sigma)
        qs.append(round4(min(100.0, max(0.0, q))))
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def page_states(case, anchor):
    """Collapse the read history into distinct states of the printed page.

    Returns a list of (hours_since_release, players_tracked, rates) with one
    entry per change of the page, keeping the first read that showed it.
    """
    states = []
    for read in case["history"]:
        rates = read["rates"]
        players = read["players_tracked"]
        if states and states[-1][1] == players and states[-1][2] == rates:
            continue
        hours = (parse_time(read["at"]) - anchor).total_seconds() / 3600.0
        states.append((hours, players, rates))
    return states


def fit_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b, residual_sd)."""
    sw = sum(ws)
    if sw <= 0:
        return ys[-1], 0.0, 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1.0, sw - 2.0)
    sd = math.sqrt(sum(w * r * r for r, w in zip(resid, ws)) / dof)
    return a, b, sd


def state_weights(states):
    """Later states carry more players (less rounding noise) and more signal
    about where the curve is heading; weight by sqrt(players), tilted recent."""
    k = len(states)
    top = max(p for _, p, _ in states) or 1
    return [
        math.sqrt(max(p, 1) / top) * (0.55 + 0.45 * (i + 1) / k)
        for i, (_, p, _) in enumerate(states)
    ]


def trend_forecast(states, weights, idx, x_target):
    """Decomposition forecast for one trophy from the recent page states.

    Fits ln(earners + 3) against ln(1 + hours) over the window the caller cut.
    Early reads have earners near zero, which would bend any log fit, so the
    window is recent and the growth the fit may imply is capped three ways:
    the slope, the earners multiple, and the share of the pool it may claim.
    Returns (earners_at_target, residual_sd).
    """
    xs = [math.log1p(max(h, 0.0)) for h, _, _ in states]
    earners = []
    for _, players, rates in states:
        r = rates[idx]
        earners.append((r if r is not None else 0.0) / 100.0 * players)
    ys = [math.log(e + 3.0) for e in earners]
    a, b, sd = fit_line(xs, ys, weights)
    b = min(3.0, max(-0.1, b))
    e_target = math.exp(a + b * x_target) - 3.0
    return e_target, sd


def direct_forecast(states, weights, idx, x_target):
    """A saturating trend of the rate itself, the decomposition's cross-check.

    Earn rates bend toward a ceiling, so a straight logit line overshoots
    exactly the fast climbers.  Fit instead logit(rate) = L - exp(a - b*x):
    for a trial ceiling L the model is linear in ln(L - logit(rate)), so a
    small grid over L (from just above the last logit to near 100%) with a
    least-squares line inside picks the best-fitting bend.  Falls back to the
    capped straight line when no ceiling fits better.
    """
    xs = [math.log1p(max(h, 0.0)) for h, _, _ in states]
    ys = []
    for _, _, rates in states:
        r = rates[idx]
        ys.append(logit(r if r is not None else 0.01))
    def logit_sse(preds):
        return sum(w * (y - p) ** 2 for y, p, w in zip(ys, preds, weights))

    a, b, sd = fit_line(xs, ys, weights)
    b = min(2.5, max(-1.5, b))
    best_sse = logit_sse([a + b * x for x in xs])
    best_pred = a + b * x_target

    top = max(ys)
    ceiling = top + 0.15
    while ceiling <= 9.2:
        gaps = [ceiling - y for y in ys]
        if min(gaps) > 1e-6:
            la, lb, _lsd = fit_line(xs, [math.log(g) for g in gaps], weights)
            if lb < 0:  # the gap to the ceiling must close, not widen
                preds = [ceiling - math.exp(la + lb * x) for x in xs]
                sse = logit_sse(preds)
                if sse < best_sse:
                    best_sse = sse
                    best_pred = ceiling - math.exp(la + lb * x_target)
        ceiling += 0.35
    return 100.0 * sigmoid(best_pred), sd


def frozen_forecast(rate, hours_target):
    """Dilution prior for a page that never refreshed inside the window.

    The tracked pool of a just-launched game multiplies at the next refresh
    (launch night's first refresh took it sixfold) while the earners of the
    small early pool barely move in a short horizon, so rates fall by most of
    the pool's growth factor.  The factor fades as the target moves away from
    launch, when a frozen page means something quieter.
    """
    if hours_target <= 8:
        pool_growth = 6.0
    elif hours_target <= 48:
        pool_growth = 3.0
    else:
        pool_growth = 1.6
    earner_growth = 1.25
    median = rate * earner_growth / pool_growth
    return median, 1.0


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    history = case["history"]
    if not history:
        raise ValueError("empty history")

    anchor = FALLBACK_ANCHOR
    release = (case.get("game") or {}).get("release_date")
    if release:
        anchor = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)

    at = parse_time(case["at"])
    target = parse_time(case["target"])
    hours_last = (at - anchor).total_seconds() / 3600.0
    hours_target = (target - anchor).total_seconds() / 3600.0
    x_last = math.log1p(max(hours_last, 0.0))
    x_target = math.log1p(max(hours_target, 0.0))
    days_ahead = case["horizon_minutes"] / 1440.0

    states = page_states(case, anchor)
    last_rates = history[-1]["rates"]

    # The trend branch works on a recent window of states: launch ignition
    # (earners exploding off zero, the pool multiplying hourly) would bend a
    # whole-history log fit into nonsense, while the last day and a half of
    # states shows the regime the target actually continues.
    pool_target = None
    pool_sd = 0.0
    window = weights = None
    if len(states) >= 4:
        h_last_state = states[-1][0]
        window = [s for s in states if s[0] >= h_last_state - 24.0]
        if len(window) < 4:
            window = states[-max(4, len(states) // 2):]
        weights = state_weights(window)
        xs = [math.log1p(max(h, 0.0)) for h, _, _ in window]
        ys = [math.log(max(p, 1)) for _, p, _ in window]
        a, b, pool_sd = fit_line(xs, ys, weights)
        b = min(2.0, max(0.0, b))
        pool_last = states[-1][1]
        pool_target = math.exp(a + b * x_target)
        # A tracked pool's growth only decelerates in week one; even a
        # generous cap of two and a half times over the horizon is loose.
        pool_target = min(pool_target, pool_last * 2.5)
        pool_target = max(pool_target, pool_last * 1.0)

    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        rate = last_rates[idx]
        if rate is None:  # fall back to the last read that printed this trophy
            for read in reversed(history):
                if read["rates"][idx] is not None:
                    rate = read["rates"][idx]
                    break
            else:
                rate = 0.05

        if rate >= 99.995:
            # Pinned at 100% (the page's everyone-has-it trophy): it can only
            # wobble by rounding; keep the band tight against the ceiling.
            rows.append({"id": trophy["id"], "q": [99.85, 99.97, 100, 100, 100]})
            continue

        if pool_target is not None:
            e_target, sd_e = trend_forecast(window, weights, idx, x_target)
            e_last = rate / 100.0 * states[-1][1]
            share_last = rate / 100.0
            share_cap = min(0.999, max(share_last * 4.0, share_last + 0.25))
            e_target = max(e_target, e_last * 0.97)
            e_target = min(e_target, e_last * 30.0, pool_target * share_cap)
            median_split = 100.0 * e_target / pool_target
            median_direct, sd_d = direct_forecast(window, weights, idx, x_target)
            # Blend the two extrapolations on the logit scale, then shrink
            # the move from the last printed rate: the fits see a step
            # function, not the smooth truth, and both can overshoot.
            blended = 0.5 * (logit(median_split) + logit(median_direct))
            moved = blended - logit(rate)
            moved = min(2.6, max(-2.6, 0.8 * moved))
            median = 100.0 * sigmoid(logit(rate) + moved)
            disagree = abs(logit(median_split) - logit(median_direct))
            sigma = 0.55 + 0.15 * math.sqrt(days_ahead) + 0.18 * abs(moved)
            sigma += 0.5 * sd_e + 0.5 * sd_d + 0.4 * pool_sd + 0.35 * disagree
            sigma = min(1.6, sigma)
        else:
            median, sigma = frozen_forecast(rate, hours_target)

        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
