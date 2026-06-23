"""
src/model_training.py
---------------------
Trains, tunes, and selects among XGBoost, LightGBM, and
Random Forest regressors using TimeSeriesSplit cross-validation.

Classes
-------
ModelTrainer
    Orchestrates hyperparameter search, model comparison, and
    persistence of the best model.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from config import (
    MODELS_DIR,
    TIMESERIES_SPLIT_FOLDS,
    RANDOM_STATE,
    XGBOOST_PARAM_GRID,
    LGBM_PARAM_GRID,
    RF_PARAM_GRID,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains XGBoost, LightGBM, and Random Forest models with
    time-series cross-validation and hyperparameter tuning.

    After ``train_all()``, the best model is persisted to disk
    and accessible via ``self.best_model``.

    Parameters
    ----------
    models_dir : Path, optional
        Directory for saving trained models.
    n_splits : int, optional
        Number of TimeSeriesSplit folds.
    random_state : int, optional
        Random seed for reproducibility.
    n_iter : int, optional
        Number of iterations for RandomizedSearchCV.
    """

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        n_splits: int = TIMESERIES_SPLIT_FOLDS,
        random_state: int = RANDOM_STATE,
        n_iter: int = 10,
    ) -> None:
        self.models_dir = models_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_iter = n_iter
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Any = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def train_all(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tune and train all three model families.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.
        y_train : pd.Series
            Training target values.

        Returns
        -------
        dict
            ``{model_name: {"model": fitted_model, "cv_rmse": float, ...}}``
        """
        model_configs = [
            ("XGBoost",      self._build_xgboost(),       XGBOOST_PARAM_GRID),
            ("LightGBM",     self._build_lgbm(),          LGBM_PARAM_GRID),
            ("RandomForest", self._build_random_forest(), RF_PARAM_GRID),
        ]

        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        for name, estimator, param_grid in model_configs:
            logger.info("-" * 55)
            logger.info("Training: %s  (n_iter=%d, folds=%d)", name, self.n_iter, self.n_splits)
            start = time.time()

            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                scoring="neg_root_mean_squared_error",
                cv=tscv,
                n_jobs=-1,
                random_state=self.random_state,
                refit=True,
                verbose=0,
            )

            try:
                search.fit(X_train, y_train)
                best = search.best_estimator_
                cv_rmse = -search.best_score_

                elapsed = time.time() - start
                logger.info(
                    "%s -> CV RMSE: %.2f  |  Best params: %s  |  Time: %.1fs",
                    name, cv_rmse, search.best_params_, elapsed,
                )

                self.results[name] = {
                    "model": best,
                    "cv_rmse": cv_rmse,
                    "best_params": search.best_params_,
                    "training_time": elapsed,
                    "feature_names": list(X_train.columns),
                }

                # Persist individual model
                model_path = self.models_dir / f"{name.lower()}_model.joblib"
                joblib.dump(best, model_path)
                logger.info("Model saved: %s", model_path)

            except Exception as exc:
                logger.error("Failed to train %s: %s", name, exc)

        self._select_best_model()
        return self.results

    def get_feature_importance(self, model_name: str) -> Optional[pd.DataFrame]:
        """
        Extract feature importance from a trained model.

        Parameters
        ----------
        model_name : str
            Key in ``self.results`` (e.g. "XGBoost").

        Returns
        -------
        pd.DataFrame or None
            DataFrame with columns ['feature', 'importance'] sorted
            descending, or None if not available.
        """
        if model_name not in self.results:
            logger.warning("Model '%s' not found in results.", model_name)
            return None

        model = self.results[model_name]["model"]
        feature_names = self.results[model_name].get("feature_names", [])

        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                logger.warning("Model '%s' has no feature_importances_.", model_name)
                return None

            fi = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)
            return fi

        except Exception as exc:
            logger.error("Could not extract feature importance for %s: %s", model_name, exc)
            return None

    def load_best_model(self) -> Any:
        """Load the persisted best model from disk."""
        path = self.models_dir / "best_model.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Best model not found at: {path}")
        model = joblib.load(path)
        logger.info("Loaded best model from: %s", path)
        return model

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _select_best_model(self) -> None:
        """Select and persist the model with the lowest CV RMSE."""
        if not self.results:
            logger.error("No models trained; cannot select best.")
            return

        best_name = min(self.results, key=lambda k: self.results[k]["cv_rmse"])
        self.best_model_name = best_name
        self.best_model = self.results[best_name]["model"]

        best_path = self.models_dir / "best_model.joblib"
        joblib.dump(self.best_model, best_path)

        logger.info("=" * 55)
        logger.info("Best model selected: %s (CV RMSE=%.2f)", best_name,
                    self.results[best_name]["cv_rmse"])
        logger.info("Best model saved -> %s", best_path)
        logger.info("=" * 55)

    def _build_xgboost(self) -> XGBRegressor:
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
        )

    def _build_lgbm(self) -> LGBMRegressor:
        return LGBMRegressor(
            objective="regression",
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
            force_row_wise=True,
        )

    def _build_random_forest(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1,
        )
