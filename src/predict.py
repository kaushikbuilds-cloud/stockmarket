"""Prediction stage (CRISP-DM: Deployment).

Loads the persisted best model for a ticker and forecasts future
closing prices.

Note on methodology: the model is trained on same-day features (Open,
High, Low, Volume, and derived indicators). True multi-day forecasting
would require future Open/High/Low/Volume values, which are unknown in
advance. For educational forecasting beyond a single day, this module
uses a simplified iterative simulation: each predicted Close becomes
the basis for the next day's Open/High/Low (a persistence assumption),
Volume is held at its recent rolling average, and derived indicators
(moving averages, volatility, returns) are recomputed on the extended
series at each step. This is a reasonable approximation for a teaching
project, not a production-grade multivariate time-series forecast.
"""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.config import (
    DATE_COLUMN,
    MODEL_FILENAME_TEMPLATE,
    MODELS_DIR,
    MOVING_AVERAGE_WINDOWS,
    PROCESSED_DATA_DIR,
    VOLATILITY_WINDOW,
)
from src.feature_engineering import engineer_features
from src.utils import get_logger, load_dataframe, processed_csv_path, validate_ticker

logger = get_logger(__name__)


def load_model(ticker: str) -> dict[str, Any]:
    """Load the persisted best model bundle for `ticker`.

    Raises:
        FileNotFoundError: If no trained model exists for `ticker`.
    """
    ticker = validate_ticker(ticker)
    model_path = MODELS_DIR / MODEL_FILENAME_TEMPLATE.format(ticker=ticker)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found for '{ticker}' at {model_path}. "
            "Run train_model first."
        )
    return joblib.load(model_path)


def predict_next_day(ticker: str) -> float:
    """Predict the next trading day's closing price using the latest row.

    Raises:
        FileNotFoundError: If no processed data or trained model exists.
    """
    ticker = validate_ticker(ticker)
    bundle = load_model(ticker)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    df = load_dataframe(
        processed_csv_path(PROCESSED_DATA_DIR, ticker), parse_dates=[DATE_COLUMN]
    )
    latest_features = df[feature_columns].iloc[[-1]]
    prediction = float(model.predict(latest_features)[0])
    logger.info("Predicted next-day close for %s: %.2f", ticker, prediction)
    return prediction


def predict_future(ticker: str, days: int = 5) -> pd.DataFrame:
    """Iteratively forecast closing prices for the next `days` trading days.

    Raises:
        FileNotFoundError: If no processed data or trained model exists.
        ValueError: If `days` is not a positive integer.
    """
    if days < 1:
        raise ValueError("`days` must be a positive integer.")

    ticker = validate_ticker(ticker)
    bundle = load_model(ticker)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    history = load_dataframe(
        processed_csv_path(PROCESSED_DATA_DIR, ticker), parse_dates=[DATE_COLUMN]
    ).copy()

    recent_volume = history["Volume"].tail(VOLATILITY_WINDOW).mean()
    lookback = max(MOVING_AVERAGE_WINDOWS) + VOLATILITY_WINDOW
    rolling_window = history.tail(lookback).copy()

    predictions: list[dict[str, Any]] = []
    last_date = history[DATE_COLUMN].max()

    for step in range(1, days + 1):
        latest_row = rolling_window[feature_columns].iloc[[-1]]
        predicted_close = float(model.predict(latest_row)[0])

        next_date = last_date + pd.tseries.offsets.BDay(step)
        new_row = {
            DATE_COLUMN: next_date,
            "Open": rolling_window["Close"].iloc[-1],
            "High": predicted_close * 1.01,
            "Low": predicted_close * 0.99,
            "Close": predicted_close,
            "Adj Close": predicted_close,
            "Volume": recent_volume,
        }
        rolling_window = pd.concat(
            [rolling_window, pd.DataFrame([new_row])], ignore_index=True
        )
        rolling_window = engineer_features(rolling_window)

        predictions.append({DATE_COLUMN: next_date, "Predicted_Close": predicted_close})

    logger.info("Forecasted %d future closing prices for %s", days, ticker)
    return pd.DataFrame(predictions)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict future closing prices.")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--days", type=int, default=5)
    args = parser.parse_args()

    forecast = predict_future(args.ticker, days=args.days)
    print(f"\n=== {args.ticker} - {args.days}-Day Forecast ===")
    print(forecast.to_string(index=False))
