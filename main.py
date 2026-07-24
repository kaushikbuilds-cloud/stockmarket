"""Pipeline orchestrator for the Stock Market Analysis and Price
Prediction project.

Runs the complete CRISP-DM workflow end-to-end for a given ticker:

    Data Collection -> Preprocessing -> Feature Engineering -> EDA ->
    Model Training -> Model Evaluation -> Prediction

Usage:
    python main.py --ticker AAPL --years 5 --forecast-days 5
"""

from __future__ import annotations

import argparse
import sys

from src.config import DEFAULT_PERIOD_YEARS, DEFAULT_TICKER
from src.data_collection import collect_and_save
from src.eda import run_eda
from src.evaluate import evaluate_model
from src.feature_engineering import engineer_and_save
from src.predict import predict_future
from src.preprocessing import preprocess_and_save
from src.train_model import train_and_select_best_model
from src.utils import get_logger

logger = get_logger(__name__)


def run_pipeline(ticker: str, years: int, forecast_days: int) -> None:
    """Execute the full CRISP-DM pipeline for `ticker`."""
    print(f"\n{'=' * 60}")
    print(f"STOCK MARKET ANALYSIS AND PRICE PREDICTION - {ticker}")
    print(f"{'=' * 60}")

    print("\n[1/7] Data Collection...")
    raw_path = collect_and_save(ticker=ticker, years=years)
    print(f"      Raw data saved to: {raw_path}")

    print("\n[2/7] Data Preprocessing...")
    processed_path = preprocess_and_save(ticker)
    print(f"      Processed data saved to: {processed_path}")

    print("\n[3/7] Feature Engineering...")
    engineer_and_save(ticker)
    print("      Features engineered and merged into processed data.")

    print("\n[4/7] Exploratory Data Analysis + Visualization...")
    eda_output = run_eda(ticker)
    print(f"      Generated {len(eda_output['figures'])} figures in reports/figures/")

    print("\n[5/7] Model Training (5 models: Linear, Tree, Random Forest, Gradient Boosting, SVR)...")
    training_result = train_and_select_best_model(ticker)
    for name, score in training_result.r2_scores.items():
        print(f"      {name}: R2 = {score:.4f}")
    print(f"      Best model: {training_result.best_model_name}")

    print("\n[6/7] Model Evaluation...")
    evaluation = evaluate_model(training_result)
    print(evaluation["metrics_table"].to_string())

    print(f"\n[7/7] Prediction ({forecast_days}-day forecast)...")
    forecast = predict_future(ticker, days=forecast_days)
    print(forecast.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full stock market analysis and prediction pipeline."
    )
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
        "--forecast-days",
        type=int,
        default=5,
        help="Number of future trading days to forecast.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        run_pipeline(args.ticker, args.years, args.forecast_days)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Pipeline failed: %s", exc)
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
