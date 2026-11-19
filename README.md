# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ahmadramdani09/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ahmadramdani09/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for make.fun's [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty, "Forecast how many GTA VI players earn each trophy in week one" ($250 in USDC).

`main.py` forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, the
percent of Exophase-tracked players who will have earned it at the case target,
as five quantiles (5/25/50/75/95) scored by pinball loss on the logit scale.

How it works:

- An earn rate is a ratio, **earners / players tracked**, and the two sides move
  differently in launch week. The model collapses the read history into
  distinct page states (Exophase refreshes on its own schedule), then forecasts
  the pool and each trophy's earners separately on ln(1 + hours since release)
  over the last day of states, and divides.
- A saturating logit trend of the rate itself cross-checks the division; the
  two are blended, the implied move shrunk and capped, so launch ignition
  cannot extrapolate into nonsense.
- When the page has never refreshed inside the window (the launch-night
  practice shape), a dilution prior takes over: the pool multiplies at the
  next refresh while earners barely move, so rates fall, with a wide spread.
- Quantiles are logit-normal around the median, widening with extrapolation
  distance, fit residuals and disagreement between the two forecasts.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic: the same case always writes
the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
