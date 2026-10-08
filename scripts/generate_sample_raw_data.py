"""Generates a synthetic OHLCV CSV for offline pipeline demonstration.

This sandbox's network policy blocks outbound requests to Yahoo Finance,
so this script fabricates a realistic 5-year daily price series (geometric
random walk with mild drift/volatility) in the exact schema
`collect_and_save` would otherwise produce, purely to demonstrate that the
rest of the pipeline (preprocessing -> feature engineering -> EDA ->
training -> evaluation -> prediction) executes correctly end-to-end.

This is NOT real market data. Do not cite these numbers as the project's
reported results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RAW_DATA_DIR
from src.utils import raw_csv_path, save_dataframe


def generate_synthetic_ohlcv(ticker: str, years: int = 5, start_price: float = 150.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_days = years * 252
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    daily_returns = rng.normal(loc=0.0006, scale=0.018, size=n_days)
    close = start_price * np.cumprod(1 + daily_returns)

    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days)))
    volume = rng.integers(20_000_000, 90_000_000, n_days)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": open_.round(2),
            "High": high.round(2),
            "Low": low.round(2),
            "Close": close.round(2),
            "Adj Close": close.round(2),
            "Volume": volume,
        }
    )
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()

    df = generate_synthetic_ohlcv(args.ticker, years=args.years)
    path = save_dataframe(df, raw_csv_path(RAW_DATA_DIR, args.ticker))
    print(f"Synthetic raw data ({len(df)} rows) saved to: {path}")
