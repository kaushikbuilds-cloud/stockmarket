"""Machine Learning stage (CRISP-DM: Modeling).

Trains and compares five regressors to predict the stock's closing
price -- Linear Regression, Decision Tree, Random Forest, Gradient
Boosting, and a scaled SVR -- tunes the tree-based models via
GridSearchCV over a time-series-aware cross-validation split, selects
the best by held-out R² score, and persists it with Joblib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    CV_SPLITS,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    MODEL_FILENAME_TEMPLATE,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TUNED_PARAM_GRIDS,
)
from src.utils import get_logger, load_dataframe, processed_csv_path, validate_ticker

logger = get_logger(__name__)


@dataclass
class TrainingResult:
    """Container for a full model-training run."""

    ticker: str
    models: dict[str, Any]
    r2_scores: dict[str, float]
    best_model_name: str
    best_model: Any
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]
    model_path: str
    best_params: dict[str, dict | None] = field(default_factory=dict)


def build_feature_matrix(
    df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a processed DataFrame into feature matrix `X` and target `y`.

    Raises:
        ValueError: If any required feature column is missing from `df`.
    """
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required feature columns: {missing}. "
            "Run feature_engineering before training."
        )
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def get_candidate_models() -> dict[str, Any]:
    """Return the untrained candidate regressors to compare.

    Random Forest and Gradient Boosting are constructed with `n_jobs=1`
    where applicable: they get GridSearchCV-tuned below, and the outer
    GridSearchCV already parallelizes across folds/params, so giving
    the estimator its own parallelism too would oversubscribe cores
    without adding speed.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
        # A linear-kernel SVR is used (rather than RBF) so the model can
        # extrapolate to prices above the training range -- essential on
        # a trending stock evaluated on a held-out future window. Inputs
        # are standardized via a Pipeline, and the target is standardized
        # via TransformedTargetRegressor so SVR's epsilon-insensitive loss
        # (tuned for unit-scale data) is meaningful against dollar prices.
        "SVR (linear)": TransformedTargetRegressor(
            regressor=Pipeline(
                [("scaler", StandardScaler()), ("svr", SVR(kernel="linear", C=10))]
            ),
            transformer=StandardScaler(),
        ),
    }


def tune_model(
    name: str,
    estimator: Any,
    param_grid: dict[str, list],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Any, dict, float]:
    """Tune `estimator` via GridSearchCV over a chronological CV split.

    `TimeSeriesSplit` never shuffles, so each fold is still validated
    on data that comes after its training slice -- the same "no peeking
    at the future" protocol used for the outer train/test split.

    Returns:
        (best_estimator, best_params, best_cv_r2_score)
    """
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        scoring="r2",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info(
        "%s tuned: best_params=%s, cv_r2=%.4f",
        name,
        search.best_params_,
        search.best_score_,
    )
    return search.best_estimator_, search.best_params_, search.best_score_


def train_and_select_best_model(ticker: str) -> TrainingResult:
    """Train all candidate models for `ticker` and select the best by R².

    The train/test split preserves chronological order (no shuffling),
    which is the correct evaluation protocol for time-series data: a
    model must be tested on future prices it has never seen, not on
    randomly interleaved historical days.

    Raises:
        FileNotFoundError: If no processed CSV exists for `ticker`.
        ValueError: If the processed data lacks engineered features.
    """
    ticker = validate_ticker(ticker)
    path = processed_csv_path(PROCESSED_DATA_DIR, ticker)
    df = load_dataframe(path, parse_dates=[DATE_COLUMN])

    X, y = build_feature_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )

    logger.info(
        "Training on %d rows, testing on %d rows for %s",
        len(X_train),
        len(X_test),
        ticker,
    )

    candidates = get_candidate_models()
    models: dict[str, Any] = {}
    r2_scores: dict[str, float] = {}
    best_params: dict[str, dict | None] = {}

    for name, estimator in candidates.items():
        if name in TUNED_PARAM_GRIDS:
            model, params, _ = tune_model(
                name, estimator, TUNED_PARAM_GRIDS[name], X_train, y_train
            )
            best_params[name] = params
        else:
            model = estimator.fit(X_train, y_train)
            best_params[name] = None

        score = model.score(X_test, y_test)
        models[name] = model
        r2_scores[name] = score
        logger.info("%s R2 score: %.4f", name, score)

    best_model_name = max(r2_scores, key=r2_scores.get)
    best_model = models[best_model_name]
    logger.info("Best model: %s (R2=%.4f)", best_model_name, r2_scores[best_model_name])

    model_path = MODELS_DIR / MODEL_FILENAME_TEMPLATE.format(ticker=ticker)
    joblib.dump(
        {
            "model": best_model,
            "model_name": best_model_name,
            "feature_columns": FEATURE_COLUMNS,
            "ticker": ticker,
        },
        model_path,
    )
    logger.info("Saved best model to %s", model_path)

    return TrainingResult(
        ticker=ticker,
        models=models,
        r2_scores=r2_scores,
        best_model_name=best_model_name,
        best_model=best_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=FEATURE_COLUMNS,
        model_path=str(model_path),
        best_params=best_params,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train models for a stock.")
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()

    result = train_and_select_best_model(args.ticker)
    print("\n=== R2 Scores ===")
    for model_name, score in result.r2_scores.items():
        tuned = result.best_params.get(model_name)
        suffix = f" (tuned: {tuned})" if tuned else ""
        print(f"{model_name}: {score:.4f}{suffix}")
    print(f"\nBest model: {result.best_model_name}")
    print(f"Saved to: {result.model_path}")
