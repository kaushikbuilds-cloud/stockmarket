"""Shared utility helpers used across the pipeline modules."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.config import LOG_FORMAT, LOG_LEVEL

_TICKER_PATTERN = re.compile(r"^[A-Za-z0-9\.\-]{1,15}$")


def get_logger(name: str) -> logging.Logger:
    """Return a configured module-level logger.

    Reuses handlers if the logger already has one attached, so importing
    a module multiple times (e.g. in Streamlit reruns) never duplicates
    log lines.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
    return logger


def validate_ticker(ticker: str) -> str:
    """Normalize and validate a stock ticker symbol.

    Raises:
        ValueError: If the ticker is empty or contains invalid characters.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker symbol cannot be empty.")
    if not _TICKER_PATTERN.match(ticker):
        raise ValueError(
            f"Invalid ticker symbol: '{ticker}'. Only letters, digits, "
            "dots, and hyphens are allowed."
        )
    return ticker


def raw_csv_path(raw_dir: Path, ticker: str) -> Path:
    """Build the standard file path for a raw ticker CSV."""
    return raw_dir / f"{ticker}.csv"


def processed_csv_path(processed_dir: Path, ticker: str) -> Path:
    """Build the standard file path for a processed ticker CSV."""
    return processed_dir / f"{ticker}_processed.csv"


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Save a DataFrame to CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_dataframe(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist at `path`.
    """
    if not path.exists():
        raise FileNotFoundError(f"No file found at {path}")
    return pd.read_csv(path, parse_dates=parse_dates)
