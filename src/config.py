"""Central configuration for the Stock Market Analysis project.

All paths, default parameters, and shared constants live here so the
rest of the codebase never hardcodes a path or magic number twice.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

for _directory in (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data collection defaults
# ---------------------------------------------------------------------------
DEFAULT_TICKER: str = "AAPL"
DEFAULT_PERIOD_YEARS: int = 5
DATE_COLUMN: str = "Date"

OHLCV_COLUMNS: list[str] = [
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
]

EXAMPLE_TICKERS: list[str] = [
    "AAPL",
    "MSFT",
    "GOOG",
    "TSLA",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
]

# ---------------------------------------------------------------------------
# Offline bulk dataset (Kaggle "jacksoncrow/stock-market-dataset")
#
# A static, offline snapshot of ~8,000 US stock/ETF daily histories
# (schema-compatible with our yfinance output) ending 2020-04-01. Used
# as an alternative to live yfinance downloads -- useful when offline,
# for reproducible grading, or for cross-sectional analysis across
# thousands of tickers at once.
# ---------------------------------------------------------------------------
BULK_STOCKS_DIR: Path = DATA_DIR / "stocks"
BULK_ETFS_DIR: Path = DATA_DIR / "etfs"
BULK_SYMBOLS_META_PATH: Path = DATA_DIR / "symbols_valid_meta.csv"

BULK_SUMMARY_STOCKS_PATH: Path = PROCESSED_DATA_DIR / "bulk_summary_stocks.csv"
BULK_SUMMARY_ETFS_PATH: Path = PROCESSED_DATA_DIR / "bulk_summary_etfs.csv"

# ---------------------------------------------------------------------------
# Feature engineering defaults
# ---------------------------------------------------------------------------
MOVING_AVERAGE_WINDOWS: list[int] = [20, 50, 100]
VOLATILITY_WINDOW: int = 20

RSI_WINDOW: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BOLLINGER_WINDOW: int = 20
BOLLINGER_STD: float = 2.0
LAG_PERIODS: list[int] = [1, 2, 3]

# ---------------------------------------------------------------------------
# Machine learning defaults
# ---------------------------------------------------------------------------
TARGET_COLUMN: str = "Close"
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42
CV_SPLITS: int = 5

FEATURE_COLUMNS: list[str] = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Daily_Return",
    "MA_20",
    "MA_50",
    "MA_100",
    "Volatility",
    "Price_Diff",
    "HL_Spread",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "BB_Upper",
    "BB_Lower",
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
]

# Models in this dict get GridSearchCV hyperparameter tuning (via
# TimeSeriesSplit) in train_model.py; models not listed here are trained
# with fixed default parameters.
TUNED_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "Decision Tree Regressor": {"max_depth": [3, 5, 10, None]},
    "Random Forest Regressor": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
    },
    "Gradient Boosting Regressor": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
    },
}

MODEL_FILENAME_TEMPLATE: str = "{ticker}_best_model.joblib"
METRICS_FILENAME_TEMPLATE: str = "{ticker}_model_metrics.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_LEVEL: str = "INFO"
