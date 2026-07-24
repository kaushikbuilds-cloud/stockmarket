"""Model Evaluation stage (CRISP-DM: Evaluation).

Computes MAE, MSE, RMSE, and R² for each trained model, builds a
comparison table, and produces diagnostic plots (Actual vs Predicted,
Residuals, Prediction Error) for the best model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FIGURES_DIR, METRICS_FILENAME_TEMPLATE, REPORTS_DIR
from src.train_model import TrainingResult
from src.utils import get_logger

logger = get_logger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, MSE, RMSE, and R² for a set of predictions."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def build_metrics_table(result: TrainingResult) -> pd.DataFrame:
    """Compute test-set metrics for every trained candidate model."""
    rows = []
    for name, model in result.models.items():
        y_pred = model.predict(result.X_test)
        metrics = compute_metrics(result.y_test, y_pred)
        metrics["Model"] = name
        rows.append(metrics)

    table = pd.DataFrame(rows).set_index("Model")[["MAE", "MSE", "RMSE", "R2"]]
    table = table.sort_values("R2", ascending=False)

    out_path = REPORTS_DIR / METRICS_FILENAME_TEMPLATE.format(ticker=result.ticker)
    table.to_csv(out_path)
    logger.info("Saved metrics table to %s", out_path)
    return table


def plot_actual_vs_predicted(result: TrainingResult) -> Path:
    """Scatter plot of actual vs. predicted closing prices for the best model."""
    y_pred = result.best_model.predict(result.X_test)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(result.y_test, y_pred, alpha=0.6, color="steelblue", edgecolor="k")
    lims = [
        min(result.y_test.min(), y_pred.min()),
        max(result.y_test.max(), y_pred.max()),
    ]
    ax.plot(lims, lims, "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual Close Price")
    ax.set_ylabel("Predicted Close Price")
    ax.set_title(f"{result.ticker} — Actual vs Predicted ({result.best_model_name})")
    ax.legend()

    path = FIGURES_DIR / f"{result.ticker}_actual_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_residuals(result: TrainingResult) -> Path:
    """Residual plot (prediction error vs. predicted value) for the best model."""
    y_pred = result.best_model.predict(result.X_test)
    residuals = result.y_test.values - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, color="darkorange", edgecolor="k")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Close Price")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"{result.ticker} — Residual Plot ({result.best_model_name})")

    path = FIGURES_DIR / f"{result.ticker}_residual_plot.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction_error(result: TrainingResult) -> Path:
    """Line chart comparing actual vs. predicted prices across the test period."""
    y_pred = result.best_model.predict(result.X_test)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.y_test.values, label="Actual", color="steelblue")
    ax.plot(y_pred, label="Predicted", color="darkorange", linestyle="--")
    ax.set_xlabel("Test Sample Index (chronological)")
    ax.set_ylabel("Close Price")
    ax.set_title(f"{result.ticker} — Prediction Error Over Time ({result.best_model_name})")
    ax.legend()

    path = FIGURES_DIR / f"{result.ticker}_prediction_error.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluate_model(result: TrainingResult) -> dict[str, object]:
    """Run the full evaluation workflow for a `TrainingResult`.

    Returns:
        A dict with the metrics table and paths to the diagnostic plots.
    """
    logger.info("Evaluating models for %s", result.ticker)
    metrics_table = build_metrics_table(result)
    figures = {
        "actual_vs_predicted": plot_actual_vs_predicted(result),
        "residual_plot": plot_residuals(result),
        "prediction_error": plot_prediction_error(result),
    }
    logger.info("Evaluation complete for %s", result.ticker)
    return {"metrics_table": metrics_table, "figures": figures}


if __name__ == "__main__":
    import argparse

    from src.train_model import train_and_select_best_model

    parser = argparse.ArgumentParser(description="Evaluate trained models for a stock.")
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()

    training_result = train_and_select_best_model(args.ticker)
    evaluation = evaluate_model(training_result)

    print("\n=== Model Comparison ===")
    print(evaluation["metrics_table"])
    print(f"\nFigures saved: {list(evaluation['figures'].values())}")
