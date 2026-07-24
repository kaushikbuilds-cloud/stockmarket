"""Offline bulk dataset stage (CRISP-DM: Data Understanding, extended).

Wraps the local Kaggle "jacksoncrow/stock-market-dataset" snapshot:
~5,884 stock and ~2,165 ETF daily-history CSVs (schema-compatible with
our yfinance output) plus a symbol/company-name metadata file. Ending
date is 2020-04-01 -- this is a frozen snapshot, not live data.

Two things live here:

1. Per-ticker offline loading (`load_offline_ticker`), used as an
   alternative to `src.data_collection.download_stock_data` when
   working offline or reproducing results without network access.
2. A one-time cross-sectional summary builder (`build_bulk_summary`)
   that computes total return, volatility, and coverage for every
   ticker in the bulk dataset. Computing this on the fly for ~8,000
   files is too slow for an interactive dashboard, so it is built
   once via the CLI entry point below and cached to CSV.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BULK_ETFS_DIR,
    BULK_STOCKS_DIR,
    BULK_SUMMARY_ETFS_PATH,
    BULK_SUMMARY_STOCKS_PATH,
    BULK_SYMBOLS_META_PATH,
    DATE_COLUMN,
    OHLCV_COLUMNS,
)
from src.utils import get_logger, save_dataframe, validate_ticker

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


def _bulk_dir_for(kind: str) -> Path:
    if kind == "stocks":
        return BULK_STOCKS_DIR
    if kind == "etfs":
        return BULK_ETFS_DIR
    raise ValueError(f"kind must be 'stocks' or 'etfs', got '{kind}'")


def is_offline_dataset_available() -> bool:
    """Return True if the bulk Kaggle dataset has been downloaded locally."""
    return BULK_STOCKS_DIR.is_dir() and any(BULK_STOCKS_DIR.glob("*.csv"))


def list_offline_tickers(kind: str = "stocks") -> list[str]:
    """List tickers available in the offline bulk dataset for `kind`.

    Args:
        kind: Either "stocks" or "etfs".
    """
    directory = _bulk_dir_for(kind)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.csv"))


def load_symbol_meta() -> pd.DataFrame:
    """Load the symbol -> company name / exchange metadata table.

    Returns an empty DataFrame (with the expected columns) if the
    metadata file is not present, so callers don't need to special-case
    a missing file.
    """
    if not BULK_SYMBOLS_META_PATH.exists():
        return pd.DataFrame(columns=["Symbol", "Security Name", "ETF", "Listing Exchange"])
    return pd.read_csv(BULK_SYMBOLS_META_PATH)


def find_offline_ticker_path(ticker: str) -> Path | None:
    """Locate a ticker's CSV in the offline dataset, checking stocks then ETFs."""
    ticker = validate_ticker(ticker)
    for kind in ("stocks", "etfs"):
        candidate = _bulk_dir_for(kind) / f"{ticker}.csv"
        if candidate.exists():
            return candidate
    return None


def load_offline_ticker(ticker: str, years: int | None = None) -> pd.DataFrame:
    """Load a ticker's OHLCV history from the offline bulk dataset.

    Args:
        ticker: Stock or ETF symbol, e.g. "AAPL".
        years: If given, keep only the most recent `years` of the
            available history (the dataset ends 2020-04-01, so "recent"
            means relative to that end date, not today).

    Raises:
        FileNotFoundError: If `ticker` is not present in the offline
            stocks or ETF collections.
    """
    path = find_offline_ticker_path(ticker)
    if path is None:
        raise FileNotFoundError(
            f"'{ticker}' was not found in the offline bulk dataset "
            f"({BULK_STOCKS_DIR} or {BULK_ETFS_DIR})."
        )

    df = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    df = df[[DATE_COLUMN, *OHLCV_COLUMNS]].sort_values(DATE_COLUMN).reset_index(drop=True)

    if years is not None:
        cutoff = df[DATE_COLUMN].max() - pd.Timedelta(days=365 * years)
        df = df[df[DATE_COLUMN] >= cutoff].reset_index(drop=True)

    logger.info("Loaded %d offline rows for %s from %s", len(df), ticker, path.name)
    return df


MIN_VALID_PRICE = 0.01  # 1 cent -- below this, treat Adj Close as a data artifact


def _summarize_one_file(path: Path) -> dict[str, object] | None:
    """Compute cross-sectional summary stats for a single ticker CSV.

    Returns None (skips the ticker) if the file is malformed, too
    short, or contains invalid price data. A handful of tickers in this
    dataset have corrupted `Adj Close` values from decades of back-
    adjusting for stock splits -- including negative and near-zero
    prices that are not economically meaningful (see e.g. NVO, whose
    early-1980s Adj Close underflows to ~1e-9 and briefly goes
    negative). Rows below `MIN_VALID_PRICE` are dropped before any
    statistics are computed, rather than merely filtered by their
    first/last value, since the corruption can appear mid-series.
    """
    try:
        df = pd.read_csv(path, usecols=[DATE_COLUMN, "Adj Close", "Volume"], parse_dates=[DATE_COLUMN])
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        logger.warning("Skipping unreadable file %s: %s", path.name, exc)
        return None

    df = df.dropna(subset=["Adj Close"]).sort_values(DATE_COLUMN)
    df = df[df["Adj Close"] >= MIN_VALID_PRICE]
    if len(df) < 30:
        return None

    first_date = df[DATE_COLUMN].iloc[0]
    last_date = df[DATE_COLUMN].iloc[-1]
    first_price = df["Adj Close"].iloc[0]
    last_price = df["Adj Close"].iloc[-1]

    daily_returns = df["Adj Close"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return_pct = (last_price / first_price - 1) * 100
    annualized_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100

    # Raw total return isn't comparable across tickers whose histories
    # span wildly different lengths (5 years vs 50+ years here) -- a
    # multi-decade blue chip will always dwarf a recent IPO on total
    # return alone. CAGR (annualized return) normalizes for holding
    # period and is what "top gainers" should actually rank by.
    years_held = (last_date - first_date).days / 365.25
    cagr_pct = (
        ((last_price / first_price) ** (1 / years_held) - 1) * 100
        if years_held >= 1
        else np.nan
    )

    return {
        "Symbol": path.stem,
        "First_Date": first_date.date(),
        "Last_Date": last_date.date(),
        "Years_Held": round(float(years_held), 1),
        "Num_Rows": len(df),
        "First_Price": round(float(first_price), 4),
        "Last_Price": round(float(last_price), 4),
        "Total_Return_Pct": round(float(total_return_pct), 2),
        "CAGR_Pct": round(float(cagr_pct), 2) if pd.notna(cagr_pct) else None,
        "Annualized_Volatility_Pct": round(float(annualized_volatility), 2),
        "Avg_Volume": int(df["Volume"].mean()),
    }


def build_bulk_summary(kind: str = "stocks", limit: int | None = None) -> pd.DataFrame:
    """Compute and save cross-sectional summary stats for every ticker in `kind`.

    This reads every CSV in the offline dataset once -- for the full
    ~5,884 stocks this takes a few minutes, which is why it is a
    separate batch step rather than something the dashboard computes
    on every page load.

    Args:
        kind: Either "stocks" or "etfs".
        limit: If given, only process the first `limit` files (useful
            for a quick smoke test before running the full batch).

    Raises:
        FileNotFoundError: If the offline dataset directory for `kind`
            does not exist (i.e. it hasn't been downloaded).
    """
    directory = _bulk_dir_for(kind)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Offline dataset directory not found: {directory}. "
            "Download the Kaggle 'jacksoncrow/stock-market-dataset' into data/."
        )

    paths = sorted(directory.glob("*.csv"))
    if limit is not None:
        paths = paths[:limit]

    meta = load_symbol_meta().set_index("Symbol") if BULK_SYMBOLS_META_PATH.exists() else None

    rows: list[dict[str, object]] = []
    start_time = time.monotonic()
    for i, path in enumerate(paths, start=1):
        row = _summarize_one_file(path)
        if row is not None:
            if meta is not None and row["Symbol"] in meta.index:
                row["Security_Name"] = meta.loc[row["Symbol"], "Security Name"]
            else:
                row["Security_Name"] = ""
            rows.append(row)

        if i % 500 == 0 or i == len(paths):
            elapsed = time.monotonic() - start_time
            logger.info("Summarized %d/%d %s files (%.1fs elapsed)", i, len(paths), kind, elapsed)

    summary = pd.DataFrame(rows)
    out_path = BULK_SUMMARY_STOCKS_PATH if kind == "stocks" else BULK_SUMMARY_ETFS_PATH
    save_dataframe(summary, out_path)
    logger.info("Saved %s bulk summary (%d tickers) to %s", kind, len(summary), out_path)
    return summary


def load_bulk_summary(kind: str = "stocks") -> pd.DataFrame:
    """Load a previously built bulk summary table.

    Raises:
        FileNotFoundError: If the summary hasn't been built yet. Run
            `python -m src.bulk_analysis --kind {kind} --build` first.
    """
    path = BULK_SUMMARY_STOCKS_PATH if kind == "stocks" else BULK_SUMMARY_ETFS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No bulk summary found at {path}. Build it first with:\n"
            f"    python -m src.bulk_analysis --kind {kind} --build"
        )
    return pd.read_csv(path, parse_dates=["First_Date", "Last_Date"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the offline bulk-dataset cross-sectional summary."
    )
    parser.add_argument("--kind", choices=["stocks", "etfs"], default="stocks")
    parser.add_argument("--build", action="store_true", help="Build and save the summary.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N files (smoke test)."
    )
    args = parser.parse_args()

    if not args.build:
        parser.error("Pass --build to run the summary build (this reads every file).")

    result = build_bulk_summary(kind=args.kind, limit=args.limit)
    print(f"Built summary for {len(result)} {args.kind} tickers.")
    print(result.sort_values("CAGR_Pct", ascending=False).head(10).to_string(index=False))
