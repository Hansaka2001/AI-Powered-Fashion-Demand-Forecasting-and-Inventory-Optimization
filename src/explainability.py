"""
src/explainability.py
---------------------
SHAP (SHapley Additive exPlanations) analysis for model
interpretability.

Produces:
- outputs/visualizations/17_shap_summary_bar.png
- outputs/visualizations/18_shap_beeswarm.png
- outputs/visualizations/19_shap_waterfall_sample.png

Classes
-------
SHAPExplainer
    Wrap a fitted tree-based model and generate global + local
    SHAP explanations.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SHAP_CONFIG, VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"


def _dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": GRID_COLOR,
    })


class SHAPExplainer:
    """
    Computes SHAP values for a trained tree-based model and
    generates global and local interpretability plots.

    Parameters
    ----------
    model : fitted estimator
        Trained XGBoost, LightGBM, or Random Forest model.
    feature_names : list of str
        Feature column names.
    viz_dir : Path
        Directory for saving plots.
    sample_size : int
        Number of rows to sample for SHAP computation.
    max_display : int
        Maximum features to show in summary plots.
    """

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        viz_dir: Path = VISUALIZATIONS_DIR,
        sample_size: int = None,
        max_display: int = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.viz_dir = viz_dir
        self.sample_size = sample_size or SHAP_CONFIG["sample_size"]
        self.max_display = max_display or SHAP_CONFIG["max_display"]
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self._shap_values: Optional[np.ndarray] = None
        self._X_sample: Optional[pd.DataFrame] = None
        _dark_style()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(self, X: pd.DataFrame) -> None:
        """
        Compute SHAP values and generate all explanation plots.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (a sample will be drawn).
        """
        try:
            import shap
        except ImportError:
            logger.error(
                "SHAP not installed. Run: pip install shap"
            )
            return

        logger.info("Computing SHAP values (sample_size=%d) …", self.sample_size)

        # Sample for speed
        if len(X) > self.sample_size:
            X_sample = X.sample(self.sample_size, random_state=42).copy()
        else:
            X_sample = X.copy()

        self._X_sample = X_sample

        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)

            # For multi-output models take the first output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            self._shap_values = shap_values
            logger.info("SHAP values computed. Shape: %s", shap_values.shape)

            self._plot_bar_summary(shap, explainer, X_sample)
            self._plot_beeswarm(shap, explainer, X_sample)
            self._plot_waterfall(shap, explainer, X_sample)
            self._save_importance_csv()

        except Exception as exc:
            logger.error("SHAP computation failed: %s", exc)

    def get_importance_df(self) -> Optional[pd.DataFrame]:
        """
        Return mean absolute SHAP values as a feature importance table.

        Returns
        -------
        pd.DataFrame or None
        """
        if self._shap_values is None:
            return None
        mean_abs = np.mean(np.abs(self._shap_values), axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs)],
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)
        return df

    # ----------------------------------------------------------
    # Private plot methods
    # ----------------------------------------------------------

    def _plot_bar_summary(self, shap, explainer, X_sample: pd.DataFrame) -> None:
        """Global SHAP bar chart (mean |SHAP value|)."""
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        shap.summary_plot(
            self._shap_values,
            X_sample,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=self.max_display,
            show=False,
            color="#58a6ff",
        )
        plt.title("SHAP Feature Importance (Mean |SHAP|)", color=TEXT_COLOR, fontsize=14)
        self._save(fig, "17_shap_summary_bar.png")

    def _plot_beeswarm(self, shap, explainer, X_sample: pd.DataFrame) -> None:
        """SHAP beeswarm (dot) plot — global feature impact."""
        fig, ax = plt.subplots(figsize=(11, 8))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)

        shap.summary_plot(
            self._shap_values,
            X_sample,
            feature_names=self.feature_names,
            max_display=self.max_display,
            show=False,
            plot_size=None,
        )
        plt.title("SHAP Beeswarm — Feature Impact Distribution", color=TEXT_COLOR, fontsize=14)
        self._save(fig, "18_shap_beeswarm.png")

    def _plot_waterfall(self, shap, explainer, X_sample: pd.DataFrame) -> None:
        """Waterfall explanation for a single representative prediction."""
        try:
            import shap as shap_lib
            # Pick the sample with median predicted sales
            sample_idx = len(X_sample) // 2
            row = X_sample.iloc[[sample_idx]]

            explanation = explainer(row)

            fig, ax = plt.subplots(figsize=(12, 7))
            fig.patch.set_facecolor(BG_COLOR)
            ax.set_facecolor(BG_COLOR)

            shap_lib.waterfall_plot(explanation[0], max_display=self.max_display, show=False)
            plt.title("SHAP Waterfall — Individual Prediction Explanation",
                      color=TEXT_COLOR, fontsize=13)
            self._save(fig, "19_shap_waterfall_sample.png")
        except Exception as exc:
            logger.warning("Waterfall plot failed (non-critical): %s", exc)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _save_importance_csv(self) -> None:
        """Persist SHAP importance to CSV."""
        from config import REPORTS_DIR
        df = self.get_importance_df()
        if df is not None:
            path = REPORTS_DIR / "shap_feature_importance.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            logger.info("SHAP importance saved → %s", path)

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.viz_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        logger.info("Saved SHAP plot: %s", path.name)
