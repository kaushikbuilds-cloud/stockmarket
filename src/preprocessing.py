"""Data Preprocessing stage (CRISP-DM: Data Preparation).

Cleans a raw OHLCV CSV: handles missing values, removes duplicates,
enforces datatypes, sorts chronologically, and flags outliers using
the IQR method. The cleaned result is saved to `data/processed/`.
"""

from __future__ import annotations

import pandas as pd

from src.config import DATE_COLUMN, OHLCV_COLUMNS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.utils import (
    get_logger,
    load_dataframe,
    processed_csv_path,
    raw_csv_path,
    save_dataframe,
    validate_ticker,
)

logger = get_logger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing OHLCV values via forward-fill then back-fill.

    Forward-fill is appropriate for time series prices (carry the last
    known price forward); back-fill catches a possible gap at the very
    start of the series.
    """
    before = df[OHLCV_COLUMNS].isna().sum().sum()
    df[OHLCV_COLUMNS] = df[OHLCV_COLUMNS].ffill().bfill()
    after = df[OHLCV_COLUMNS].isna().sum().sum()
    logger.info("Missing values: %d -> %d after fill", before, after)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows based on the Date column, keeping the first."""
    before = len(df)
    df = df.drop_duplicates(subset=[DATE_COLUMN], keep="first")
    logger.info("Duplicates removed: %d", before - len(df))
    return df


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Date to datetime and OHLCV columns to numeric."""
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    for col in OHLCV_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows chronologically and reset the index."""
    return df.sort_values(DATE_COLUMN).reset_index(drop=True)


def flag_outliers(df: pd.DataFrame, column: str = "Close") -> pd.DataFrame:
    """Flag statistical outliers in `column` using the IQR method.

    Adds a boolean `Is_Outlier` column rather than dropping rows, since
    genuine market moves (crashes, rallies) are valid data points that
    should stay visible in EDA rather than being silently discarded.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df["Is_Outlier"] = ~df[column].between(lower_bound, upper_bound)
    n_outliers = int(df["Is_Outlier"].sum())
    logger.info("Flagged %d outlier rows on column '%s'", n_outliers, column)
    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline on a raw OHLCV DataFrame."""
    df = df.copy()
    df = enforce_dtypes(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = sort_by_date(df)
    df = flag_outliers(df)
    return df


def preprocess_and_save(ticker: str) -> str:
    """Load a raw ticker CSV, clean it, and save the processed version.

    Raises:
        FileNotFoundError: If no raw CSV exists for `ticker`.
    """
    ticker = validate_ticker(ticker)
    raw_path = raw_csv_path(RAW_DATA_DIR, ticker)
    df = load_dataframe(raw_path, parse_dates=[DATE_COLUMN])

    logger.info("Preprocessing %s (%d raw rows)", ticker, len(df))
    df = preprocess_dataframe(df)

    out_path = save_dataframe(df, processed_csv_path(PROCESSED_DATA_DIR, ticker))
    logger.info("Saved processed data to %s (%d rows)", out_path, len(df))
    return str(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a raw stock CSV.")
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()

    saved_path = preprocess_and_save(args.ticker)
    print(f"Saved processed data to: {saved_path}")
