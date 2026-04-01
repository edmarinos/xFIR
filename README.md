# xFIR — Expected First Inning Runs
### MLB First Inning Scoring Predictor | Statcast 2022–2025 | Live Daily App

A machine learning app that predicts the probability of each team scoring in 
the first inning for every MLB game today. Built using Statcast pitch-level 
data and the official MLB Stats API for daily game and starter updates.

**[Try the live app → xfirmlb.streamlit.app](https://xfirmlb.streamlit.app)**

---

## What It Does

- Loads today's MLB schedule automatically via the MLB Stats API
- Auto-populates probable starting pitchers for each game
- Predicts scoring probability and expected runs for both half-innings
- Calculates Expected Value (EV) for scoreless/scoring first inning bets
- Navigate forward and backward by date to explore any day's slate

---

## Models

| Task | Model | Performance |
|------|-------|-------------|
| Did team score in 1st? | XGBoost Classifier | ROC-AUC: 0.521 |
| How many runs? | Linear Regression | RMSE: 0.991 |

Both models are benchmarked against a naive baseline (always predict league 
average). First inning scoring is highly stochastic — the models marginally 
outperform the baseline, which itself is an analytically interesting finding 
consistent with baseball research on inning-level run variance.

---

## Features Used

**Starting Pitcher (opposing starter)**
ERA, FIP, xFIP, SIERA, K%, BB%, K-BB%, WHIP, HR/9, GB%, SwStr%, CSW%, 
HardHit%, xERA

**Team Offense (batting team)**
Team OPS, wRC+, BB%, K%

**Game Context**
Home/away, park factor, month, rolling 10-game first inning run average

---

## EV Calculator

For each game the app shows:
- **Model probability** of scoreless / scoring first inning
- **Sportsbook implied probability** from manually entered American odds
- **Edge** — model probability minus implied probability
- **EV per $100** — expected profit or loss at those odds

> ⚠️ For educational purposes only. Not financial or betting advice.

---

## Data Sources

- **Statcast** via [pybaseball](https://github.com/jldbc/pybaseball) — 
  pitch-level event data 2022–2025
- **FanGraphs** via pybaseball — pitcher and team season stats
- **MLB Stats API** — daily schedule and probable starters (free, no key required)

---

## How to Run Locally
```bash
git clone https://github.com/edmarinos/xFIR.git
cd xFIR
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## Project Structure
```
xFIR/
├── app.py                    # Streamlit application
├── requirements.txt
├── xfir_classifier.pkl       # Trained XGBoost classifier
├── xfir_regressor.pkl        # Trained Linear Regression model
├── scaler_v2.pkl             # Fitted StandardScaler
├── features_v2.json          # Feature list
├── pitcher_list_2025.csv     # 2025 pitcher stats
└── team_offense_2025.csv     # 2025 team offense stats
```

---

## Tech Stack

Python, scikit-learn, XGBoost, pybaseball, Streamlit, pandas, NumPy, requests

---

## Related Projects

- [Soccer xG Model](https://github.com/edmarinos/soccer-xg-model) — 
  Expected Goals model trained on FIFA World Cup 2022 data
