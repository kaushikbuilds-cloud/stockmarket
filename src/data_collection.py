"""Data Collection stage (CRISP-DM: Data Understanding).

Downloads historical OHLCV data for a given stock ticker from Yahoo
Finance via the `yfinance` API and persists it as a raw CSV under
`data/raw/`. Can also source from the local offline bulk dataset (see
`src.bulk_analysis`) when working without network access or when
reproducing results deterministically.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from src.bulk_analysis import is_offline_dataset_available, load_offline_ticker
from src.config import DEFAULT_PERIOD_YEARS, DEFAULT_TICKER, RAW_DATA_DIR
from src.utils import get_logger, raw_csv_path, save_dataframe, validate_ticker

logger = get_logger(__name__)


def download_stock_data(
    ticker: str,
    years: int = DEFAULT_PERIOD_YEARS,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """Download historical OHLCV data for `ticker` from Yahoo Finance.

    Args:
        ticker: Stock symbol, e.g. "AAPL" or "RELIANCE.NS".
        years: Number of years of history to fetch, ending at `end_date`.
        end_date: End of the date range (defaults to today).

    Returns:
        A DataFrame with columns Date, Open, High, Low, Close,
        Adj Close, Volume.

    Raises:
        ValueError: If the ticker is invalid or yfinance returns no data.
    """
    ticker = validate_ticker(ticker)
    end_date = end_date or datetime.today()
    start_date = end_date - timedelta(days=365 * years)

    logger.info(
        "Downloading %s data from %s to %s",
        ticker,
        start_date.date(),
        end_date.date(),
    )

    raw = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}'. Check that the "
            "symbol is correct and that it exists on Yahoo Finance."
        )

    # yfinance may return a MultiIndex column frame (Price, Ticker) for
    # single tickers depending on version; flatten to a simple index.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index()
    raw = raw.rename(columns={"index": "Date"})

    expected_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in expected_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing expected columns: {missing}")

    df = raw[expected_cols].copy()
    logger.info("Downloaded %d rows for %s", len(df), ticker)
    return df


def collect_and_save(
    ticker: str = DEFAULT_TICKER,
    years: int = DEFAULT_PERIOD_YEARS,
    source: str = "yfinance",
) -> str:
    """Fetch data for `ticker` and save it under data/raw/.

    Args:
        ticker: Stock ticker symbol.
        years: Number of years of history.
        source: One of:
            - "yfinance" (default): live download, matches all prior
              behavior of this function exactly.
            - "offline": read from the local bulk Kaggle dataset (see
              `src.bulk_analysis`); no network access required, but
              data ends 2020-04-01.
            - "auto": try "yfinance" first, and if that raises (e.g.
              no network), fall back to "offline" if the ticker is
              available there.

    Returns:
        The path to the saved CSV file, as a string.

    Raises:
        ValueError: If `source` is not one of the values above.
        FileNotFoundError: If source="offline" and `ticker` isn't in
            the bulk dataset.
    """
    ticker = validate_ticker(ticker)

    if source == "yfinance":
        df = download_stock_data(ticker, years=years)
    elif source == "offline":
        df = load_offline_ticker(ticker, years=years)
    elif source == "auto":
        try:
            df = download_stock_data(ticker, years=years)
        except Exception as exc:  # network failure, rate limit, etc.
            if not is_offline_dataset_available():
                raise
            logger.warning(
                "yfinance download failed for %s (%s); falling back to offline dataset",
                ticker,
                exc,
            )
            df = load_offline_ticker(ticker, years=years)
    else:
        raise ValueError(f"source must be 'yfinance', 'offline', or 'auto', got '{source}'")

    path = save_dataframe(df, raw_csv_path(RAW_DATA_DIR, ticker))
    logger.info("Saved raw data to %s", path)
    return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download historical stock data.")
    parser.add_argument(
        "--ticker",
        type=str,
        default=DEFAULT_TICKER,
        help="Stock ticker symbol, e.g. AAPL, TSLA, RELIANCE.NS",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=DEFAULT_PERIOD_YEARS,
        help="Number of years of historical data to fetch.",
    )
    parser.add_argument(
        "--source",
        choices=["yfinance", "offline", "auto"],
        default="yfinance",
        help="Data source: live yfinance download, the offline bulk "
        "dataset, or auto (yfinance with offline fallback).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    saved_path = collect_and_save(ticker=args.ticker, years=args.years, source=args.source)
    print(f"Saved raw data to: {saved_path}")
