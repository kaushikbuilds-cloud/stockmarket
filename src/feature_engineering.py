"""Feature Engineering stage (CRISP-DM: Data Preparation).

Derives predictive features from cleaned OHLCV data: returns, moving
averages, volatility, and price-spread measures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BOLLINGER_STD,
    BOLLINGER_WINDOW,
    DATE_COLUMN,
    LAG_PERIODS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MOVING_AVERAGE_WINDOWS,
    PROCESSED_DATA_DIR,
    RSI_WINDOW,
    VOLATILITY_WINDOW,
)
from src.utils import get_logger, load_dataframe, processed_csv_path, validate_ticker

logger = get_logger(__name__)


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """Add the percentage daily return of the Close price."""
    df["Daily_Return"] = df["Close"].pct_change() * 100
    return df


def add_moving_averages(
    df: pd.DataFrame, windows: list[int] = MOVING_AVERAGE_WINDOWS
) -> pd.DataFrame:
    """Add simple moving averages of the Close price for each window."""
    for window in windows:
        df[f"MA_{window}"] = df["Close"].rolling(window=window, min_periods=1).mean()
    return df


def add_volatility(
    df: pd.DataFrame, window: int = VOLATILITY_WINDOW
) -> pd.DataFrame:
    """Add rolling standard deviation of daily returns as a volatility proxy."""
    if "Daily_Return" not in df.columns:
        df = add_daily_return(df)
    df["Volatility"] = df["Daily_Return"].rolling(window=window, min_periods=1).std()
    return df


def add_price_difference(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-over-day absolute Close price difference."""
    df["Price_Diff"] = df["Close"].diff()
    return df


def add_high_low_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Add the intraday High-Low spread."""
    df["HL_Spread"] = df["High"] - df["Low"]
    return df


def add_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    """Add the Relative Strength Index (RSI) of the Close price.

    RSI is bounded [0, 100]; a flat window (no gains or losses) is
    scored as neutral (50) rather than left undefined, and a window
    with gains but zero losses is scored as maximally overbought (100).
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    default_rsi = 100 - (100 / (1 + rs))
    conditions = [(avg_gain == 0) & (avg_loss == 0), avg_loss == 0]
    choices = [50.0, 100.0]
    df["RSI"] = pd.Series(
        np.select(conditions, choices, default=default_rsi), index=df.index
    ).fillna(50.0)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """Add MACD, its signal line, and the MACD histogram."""
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame, window: int = BOLLINGER_WINDOW, num_std: float = BOLLINGER_STD
) -> pd.DataFrame:
    """Add upper/lower Bollinger Bands around the `window`-day moving average."""
    ma_col = f"MA_{window}"
    middle = df[ma_col] if ma_col in df.columns else df["Close"].rolling(
        window=window, min_periods=1
    ).mean()
    rolling_std = df["Close"].rolling(window=window, min_periods=1).std().fillna(0)
    df["BB_Upper"] = middle + num_std * rolling_std
    df["BB_Lower"] = middle - num_std * rolling_std
    return df


def add_lag_features(df: pd.DataFrame, periods: list[int] = LAG_PERIODS) -> pd.DataFrame:
    """Add lagged Close prices as autoregressive features.

    Rows before a lag exists default to the current Close (i.e. "no
    momentum yet") rather than NaN or zero.
    """
    for period in periods:
        df[f"Close_Lag_{period}"] = df["Close"].shift(period).fillna(df["Close"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature engineering pipeline on a processed DataFrame."""
    df = df.copy()
    df = add_daily_return(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_price_difference(df)
    df = add_high_low_spread(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_lag_features(df)

    # Rolling/diff operations leave NaNs only in the first rows; fill
    # with 0 so downstream models receive a complete feature matrix.
    engineered_cols = (
        ["Daily_Return", "Volatility", "Price_Diff", "RSI", "MACD", "MACD_Signal", "MACD_Hist"]
        + [f"MA_{w}" for w in MOVING_AVERAGE_WINDOWS]
    )
    df[engineered_cols] = df[engineered_cols].fillna(0)

    logger.info("Engineered %d feature columns", len(engineered_cols) + 1)
    return df


def engineer_and_save(ticker: str) -> str:
    """Load processed data for `ticker`, engineer features, and overwrite it.

    Raises:
        FileNotFoundError: If no processed CSV exists for `ticker`.
    """
    ticker = validate_ticker(ticker)
    path = processed_csv_path(PROCESSED_DATA_DIR, ticker)
    df = load_dataframe(path, parse_dates=[DATE_COLUMN])

    df = engineer_features(df)
    df.to_csv(path, index=False)
    logger.info("Saved feature-engineered data to %s", path)
    return str(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Engineer features for a stock.")
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()

    saved_path = engineer_and_save(args.ticker)
    print(f"Saved feature-engineered data to: {saved_path}")
