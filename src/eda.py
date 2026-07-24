"""Exploratory Data Analysis stage (CRISP-DM: Data Understanding/Modeling prep).

Computes descriptive statistics and summary tables, and triggers
generation of the full visualization suite via `src.visualization`.
"""

from __future__ import annotations

import pandas as pd

from src.config import DATE_COLUMN, PROCESSED_DATA_DIR
from src.utils import get_logger, load_dataframe, processed_csv_path, validate_ticker
from src.visualization import generate_all_figures

logger = get_logger(__name__)


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics (count, mean, std, quartiles, etc.)."""
    return df.select_dtypes(include="number").describe().transpose()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the Pearson correlation matrix of numeric columns."""
    return df.select_dtypes(include="number").corr()


def monthly_average_closing_price(df: pd.DataFrame) -> pd.DataFrame:
    """Return average Close price grouped by calendar month."""
    monthly = df.assign(
        Month=pd.to_datetime(df[DATE_COLUMN]).dt.to_period("M").astype(str)
    )
    return monthly.groupby("Month", as_index=False)["Close"].mean()


def top_gain_days(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the `top_n` days with the highest positive daily return."""
    return (
        df.dropna(subset=["Daily_Return"])
        .nlargest(top_n, "Daily_Return")[[DATE_COLUMN, "Close", "Daily_Return"]]
        .reset_index(drop=True)
    )


def top_loss_days(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the `top_n` days with the largest negative daily return."""
    return (
        df.dropna(subset=["Daily_Return"])
        .nsmallest(top_n, "Daily_Return")[[DATE_COLUMN, "Close", "Daily_Return"]]
        .reset_index(drop=True)
    )


def run_eda(ticker: str) -> dict[str, object]:
    """Run the full EDA workflow for `ticker`: stats + all visualizations.

    Raises:
        FileNotFoundError: If no processed CSV exists for `ticker`.

    Returns:
        A dict containing summary tables and the paths to saved figures.
    """
    ticker = validate_ticker(ticker)
    path = processed_csv_path(PROCESSED_DATA_DIR, ticker)
    df = load_dataframe(path, parse_dates=[DATE_COLUMN])

    logger.info("Running EDA for %s (%d rows)", ticker, len(df))

    results = {
        "summary_statistics": summary_statistics(df),
        "correlation_matrix": correlation_matrix(df),
        "monthly_average_closing_price": monthly_average_closing_price(df),
        "top_gain_days": top_gain_days(df),
        "top_loss_days": top_loss_days(df),
        "figures": generate_all_figures(df, ticker),
    }
    logger.info("EDA complete for %s", ticker)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EDA for a stock.")
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()

    output = run_eda(args.ticker)
    print("\n=== Summary Statistics ===")
    print(output["summary_statistics"])
    print("\n=== Top Gain Days ===")
    print(output["top_gain_days"])
    print("\n=== Top Loss Days ===")
    print(output["top_loss_days"])
    print(f"\nFigures saved: {list(output['figures'].values())}")
