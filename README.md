# 📈 Stock Market Analysis and Price Prediction using Data Science

A complete, production-ready Data Science micro-project that analyzes historical
stock market data and predicts future closing prices — built end-to-end following
the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) lifecycle.

> Educational project. Nothing here is financial advice.

---

## Project Overview

This project pulls real historical OHLCV (Open, High, Low, Close, Volume) data for
any publicly traded stock ticker from Yahoo Finance, cleans and engineers features
from it, explores it visually, trains multiple regression models to predict the
closing price, evaluates them rigorously, and serves everything through an
interactive Streamlit dashboard.

### CRISP-DM Stages Covered

| Stage | Module |
|---|---|
| Business Understanding | This README / project objective |
| Data Understanding | [`src/data_collection.py`](src/data_collection.py) |
| Data Preparation | [`src/preprocessing.py`](src/preprocessing.py), [`src/feature_engineering.py`](src/feature_engineering.py) |
| Modeling | [`src/train_model.py`](src/train_model.py) |
| Evaluation | [`src/evaluate.py`](src/evaluate.py) |
| Deployment | [`src/predict.py`](src/predict.py), [`app.py`](app.py) |

---

## Features

- **Data Collection** — download 1–10 years of OHLCV history for any ticker via `yfinance`
- **Data Cleaning** — missing-value imputation, duplicate removal, dtype enforcement, IQR-based outlier flagging
- **Feature Engineering** — daily return, 20/50/100-day moving averages, rolling volatility, price difference, high-low spread
- **Exploratory Data Analysis** — summary statistics, correlation matrix, monthly averages, top gain/loss days
- **Technical Indicators** — RSI, MACD (+ signal & histogram), Bollinger Bands, and lagged-price features, used both as model inputs and as chart overlays
- **Visualization** — 15 charts (Plotly interactive + Matplotlib static) covering trend, volume, candlestick, distribution, correlation, autocorrelation, rolling volatility, and technical indicators
- **Machine Learning** — five regressors compared automatically (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, and a scaled linear SVR); the tree-based models are hyperparameter-tuned via `GridSearchCV` over a `TimeSeriesSplit` cross-validation; best model selected by R² and persisted with Joblib
- **Model Evaluation** — MAE, MSE, RMSE, R² comparison table; tuned-hyperparameter report; Actual-vs-Predicted, Residual, and Prediction Error plots
- **Prediction** — single and multi-day future closing price forecasts
- **Interactive Dashboard** — an 8-page Streamlit app: Dashboard, Stock Analysis, Technical Indicators, EDA, Prediction, Model Performance, Compare Tickers, Market Overview
- **Offline Dataset Mode** — an optional local snapshot of ~5,900 stocks and ~2,000 ETFs (Kaggle's `jacksoncrow/stock-market-dataset`) usable instead of live `yfinance` calls, plus a precomputed cross-sectional summary (CAGR, volatility, coverage) for market-wide analysis

---

## Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.12+ |
| Data | pandas, numpy |
| Data Source | yfinance (Yahoo Finance API) |
| Visualization | matplotlib, plotly |
| Machine Learning | scikit-learn |
| Model Persistence | joblib |
| Dashboard | streamlit |

All libraries used are free and open-source.

---

## Folder Structure

```
Stock-Market-Analysis/
│
├── data/
│   ├── raw/                     # Raw OHLCV CSVs downloaded from yfinance
│   ├── processed/               # Cleaned + feature-engineered CSVs, bulk summaries
│   ├── stocks/                  # (optional, gitignored) offline bulk dataset -- stocks
│   ├── etfs/                    # (optional, gitignored) offline bulk dataset -- ETFs
│   └── symbols_valid_meta.csv   # (optional, gitignored) symbol -> company name metadata
│
├── notebooks/                # (optional) exploratory notebooks
│
├── models/                   # Persisted best model per ticker (.joblib)
│
├── reports/
│   ├── figures/               # All generated charts (HTML + PNG)
│   └── *_model_metrics.csv    # Saved model comparison tables
│
├── src/
│   ├── config.py              # Paths, constants, defaults
│   ├── data_collection.py     # yfinance download logic + offline source option
│   ├── bulk_analysis.py       # Offline dataset loading + cross-sectional summary
│   ├── preprocessing.py       # Cleaning pipeline
│   ├── feature_engineering.py # Derived indicators
│   ├── eda.py                 # Summary statistics
│   ├── visualization.py       # Chart builders
│   ├── train_model.py         # Model training + selection
│   ├── evaluate.py            # Metrics + diagnostic plots
│   ├── predict.py             # Future price forecasting
│   └── utils.py                # Shared helpers (logging, I/O)
│
├── app.py                     # Streamlit dashboard
├── main.py                    # CLI pipeline orchestrator
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline from the command line

```bash
python main.py --ticker AAPL --years 5 --forecast-days 5
```

This runs data collection → preprocessing → feature engineering → EDA →
model training → evaluation → prediction, and prints a summary of each stage.

Supported tickers include any symbol Yahoo Finance recognizes, e.g.
`AAPL`, `MSFT`, `GOOG`, `TSLA`, `RELIANCE.NS`, `TCS.NS`, `INFY.NS`.

### Run individual stages

```bash
python -m src.data_collection --ticker AAPL --years 5
python -m src.preprocessing --ticker AAPL
python -m src.feature_engineering --ticker AAPL
python -m src.eda --ticker AAPL
python -m src.train_model --ticker AAPL
python -m src.evaluate --ticker AAPL
python -m src.predict --ticker AAPL --days 7
```

### Launch the interactive dashboard

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`), enter a
ticker in the sidebar, and explore the eight pages: **Dashboard**, **Stock
Analysis**, **Technical Indicators**, **EDA**, **Prediction**, **Model
Performance**, **Compare Tickers**, and **Market Overview**.

---

## Screenshots

> _Add screenshots of the dashboard pages here after running the app locally._

- `reports/screenshots/dashboard.png`
- `reports/screenshots/stock_analysis.png`
- `reports/screenshots/technical_indicators.png`
- `reports/screenshots/eda.png`
- `reports/screenshots/prediction.png`
- `reports/screenshots/model_performance.png`
- `reports/screenshots/compare_tickers.png`

---

## Modeling Notes

The train/test split is **chronological** (no shuffling) — the model is always
tested on days it has never seen, which is the correct evaluation protocol for
time series data. Hyperparameter tuning uses `GridSearchCV` with a
`TimeSeriesSplit` cross-validator, so even the inner validation folds never
peek at the future.

In practice this setup often favors the **linear models** (Linear Regression and the
linear SVR) over the tree-based ones: tree regressors cannot extrapolate beyond the
price range seen during training, so on a strongly trending stock they underperform
on a held-out future period even when they fit the training data well — a deliberately
retained, explainable contrast rather than a bug, and a good discussion point for the
project viva. The SVR uses a **linear kernel** (so it *can* extrapolate) and is wrapped
in a `TransformedTargetRegressor` so its epsilon-insensitive loss operates on a
standardized target rather than raw dollar prices; both choices keep it a sensible,
competitive member of the roster.

Multi-day forecasts in `src/predict.py` use a simplified iterative simulation
(next-day Open/High/Low are approximated from the previous predicted Close, and
Volume is held at its recent average) since true future OHLCV inputs are not
knowable in advance. This is clearly an approximation suited to a teaching
project, not a production forecasting system.

---

## Offline Dataset

In addition to live `yfinance` downloads, the project can run entirely offline
against a static local snapshot: Kaggle's
[`jacksoncrow/stock-market-dataset`](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
(~5,900 stock and ~2,000 ETF daily-history CSVs, schema-compatible with our
pipeline, ending **2020-04-01**). It is not bundled with the repo (~2.6 GB) —
download it yourself:

```python
import kagglehub
path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")
```

...then copy `stocks/`, `etfs/`, and `symbols_valid_meta.csv` into this
project's `data/` folder (`data/stocks/`, `data/etfs/`, `data/symbols_valid_meta.csv`).

**Using it:**

- **Per-ticker offline mode** — in the dashboard sidebar, set **Data Source**
  to "offline" (frozen at 2020-04-01) or "auto" (live `yfinance` with an
  offline fallback if the network call fails). From the CLI:
  ```bash
  python -m src.data_collection --ticker NFLX --years 5 --source offline
  ```
- **Market Overview page** — a cross-sectional view across the *entire*
  offline universe (median CAGR, return distribution, risk-vs-return
  scatter, top/bottom performers, searchable table). Computing statistics
  for ~8,000 tickers live would make the dashboard unusably slow, so this
  is precomputed once via a batch script, not recalculated on every page
  load:
  ```bash
  python -m src.bulk_analysis --kind stocks --build
  python -m src.bulk_analysis --kind etfs --build
  ```
  This writes `data/processed/bulk_summary_stocks.csv` and
  `bulk_summary_etfs.csv`. Re-run it any time the underlying dataset changes.

**A data-quality note:** a handful of legacy tickers in this dataset have
corrupted `Adj Close` values from decades of back-adjusting for stock splits
(including prices that underflow to near-zero, or even negative). `src/bulk_analysis.py`
filters these out (`MIN_VALID_PRICE`) and ranks tickers by **CAGR** (annualized
return) rather than raw total return — raw total return isn't comparable
across tickers whose histories span anywhere from 5 to 58 years, so a 50-year
blue chip would trivially dominate a "top gainers" list otherwise.

---

## Deployment

The dashboard can be deployed for free on **Streamlit Community Cloud**:

1. Push this project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select the repository, and set the main file to `app.py`.
4. Deploy. Streamlit Cloud auto-detects `requirements.txt` and installs everything.

No secrets or API keys are required — all data comes from the free `yfinance`
endpoint, so there is nothing to configure. The `data/`, `models/`, and
`reports/figures/` artifacts are regenerated on demand at runtime.

---

## Future Improvements

- Add LSTM/GRU deep learning models for sequential forecasting
- Add sentiment analysis from financial news/social media (needs a news API key)
- Add automated model retraining on a schedule
- Portfolio optimization / risk metrics (Sharpe ratio, max drawdown)

---

## License

This project is intended for educational use as part of a college Data Science
micro-project submission.
