"""DataFrame Operations Demo — Stock Market Analysis Project.

A single, runnable walkthrough of every pandas DataFrame operation used
in this project: creation (from CSV), inspection, cleaning, feature
engineering, filtering, grouping, and sorting. Uses the project's own
`src/` functions on real data, so the output matches what the pipeline
actually produces.

Run with:
    python notebooks/dataframe_demo.py --ticker AAPL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATE_COLUMN, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_collection import collect_and_save
from src.feature_engineering import engineer_and_save
from src.preprocessing import preprocess_and_save
from src.utils import load_dataframe, processed_csv_path, raw_csv_path

pd.set_option("display.max_columns", 12)
pd.set_option("display.width", 120)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main(ticker: str) -> None:
    # ---------------------------------------------------------------
    # 1. CREATE a DataFrame -- download (or reuse) raw data, then load
    #    the CSV straight into a pandas DataFrame.
    # ---------------------------------------------------------------
    section("1. CREATE - load raw OHLCV data into a DataFrame")
    collect_and_save(ticker=ticker, years=5)
    df: pd.DataFrame = pd.read_csv(raw_csv_path(RAW_DATA_DIR, ticker), parse_dates=[DATE_COLUMN])
    print(df.head())

    # ---------------------------------------------------------------
    # 2. INSPECT the DataFrame -- shape, dtypes, summary statistics.
    # ---------------------------------------------------------------
    section("2. INSPECT - shape, dtypes, describe()")
    print("shape:", df.shape)
    print("\ndtypes:\n", df.dtypes)
    print("\ndescribe():\n", df.describe())

    # ---------------------------------------------------------------
    # 3. CLEAN the DataFrame -- missing values, duplicates, dtypes,
    #    sort, outlier flag. (src/preprocessing.py)
    # ---------------------------------------------------------------
    section("3. CLEAN - preprocess_and_save() -> DataFrame with Is_Outlier column")
    preprocess_and_save(ticker)
    df = load_dataframe(processed_csv_path(PROCESSED_DATA_DIR, ticker), parse_dates=[DATE_COLUMN])
    print(df[[DATE_COLUMN, "Close", "Is_Outlier"]].tail())

    # ---------------------------------------------------------------
    # 4. ADD COLUMNS (feature engineering) -- new DataFrame columns
    #    derived from existing ones. (src/feature_engineering.py)
    # ---------------------------------------------------------------
    section("4. ADD COLUMNS - engineer_and_save() -> 20 engineered features")
    engineer_and_save(ticker)
    df = load_dataframe(processed_csv_path(PROCESSED_DATA_DIR, ticker), parse_dates=[DATE_COLUMN])
    print("columns now:", list(df.columns))
    print(df[[DATE_COLUMN, "Close", "MA_20", "RSI", "MACD"]].tail())

    # ---------------------------------------------------------------
    # 5. FILTER rows -- boolean indexing.
    # ---------------------------------------------------------------
    section("5. FILTER - boolean indexing (RSI > 70, i.e. overbought days)")
    overbought = df[df["RSI"] > 70]
    print(f"{len(overbought)} overbought days out of {len(df)} total")
    print(overbought[[DATE_COLUMN, "Close", "RSI"]].tail())

    # ---------------------------------------------------------------
    # 6. GROUP BY -- aggregate rows into a smaller summary DataFrame.
    # ---------------------------------------------------------------
    section("6. GROUP BY - monthly average closing price")
    monthly = (
        df.assign(Month=df[DATE_COLUMN].dt.to_period("M").astype(str))
        .groupby("Month", as_index=False)["Close"]
        .mean()
    )
    print(monthly.tail())

    # ---------------------------------------------------------------
    # 7. SORT / SELECT -- rank rows and pick the top N.
    # ---------------------------------------------------------------
    section("7. SORT - top 5 single-day gains (nlargest on Daily_Return)")
    top_gains = df.nlargest(5, "Daily_Return")[[DATE_COLUMN, "Close", "Daily_Return"]]
    print(top_gains)

    # ---------------------------------------------------------------
    # 8. TRAIN/TEST SPLIT -- slicing a DataFrame chronologically,
    #    the way src/train_model.py prepares data for scikit-learn.
    # ---------------------------------------------------------------
    section("8. SPLIT - chronological 80/20 train/test slice")
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"train: {train_df.shape}, test: {test_df.shape}")
    print("train ends:", train_df[DATE_COLUMN].max().date())
    print("test starts:", test_df[DATE_COLUMN].min().date())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataFrame operations demo.")
    parser.add_argument("--ticker", type=str, default="AAPL")
    args = parser.parse_args()
    main(args.ticker)
