"""
src/model_evaluation.py
------------------------
Evaluates trained models on the held-out test set using:
  • MAE   – Mean Absolute Error
  • RMSE  – Root Mean Squared Error
  • MAPE  – Mean Absolute Percentage Error
  • R²    – Coefficient of Determination

Produces:
  - outputs/reports/model_evaluation_report.csv
  - outputs/visualizations/09_model_comparison.png
  - outputs/visualizations/10_actual_vs_predicted.png
  - outputs/visualizations/11_residual_plot.png

Classes
-------
ModelEvaluator
    Compute metrics and generate evaluation visualisations.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import REPORTS_DIR, VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"
ACCENT_COLOR = "#58a6ff"
HIGHLIGHT_COLOR = "#f78166"
GREEN_COLOR = "#3fb950"


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
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
    })


class ModelEvaluator:
    """
    Computes evaluation metrics for multiple trained models and
    generates comparison visualisations.

    Parameters
    ----------
    reports_dir : Path
        Directory for CSV report output.
    viz_dir : Path
        Directory for visualisation output.
    """

    def __init__(
        self,
        reports_dir: Path = REPORTS_DIR,
        viz_dir: Path = VISUALIZATIONS_DIR,
    ) -> None:
        self.reports_dir = reports_dir
        self.viz_dir = viz_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_df: Optional[pd.DataFrame] = None
        _dark_style()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def evaluate_all(
        self,
        model_results: Dict[str, Dict[str, Any]],
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> pd.DataFrame:
        """
        Evaluate all trained models against the test set.

        Parameters
        ----------
        model_results : dict
            Output of ``ModelTrainer.train_all()``.
        X_test : pd.DataFrame
        y_test : pd.Series

        Returns
        -------
        pd.DataFrame
            Metrics for all models (MAE, RMSE, MAPE, R²).
        """
        rows = []
        predictions: Dict[str, np.ndarray] = {}

        for name, info in model_results.items():
            model = info["model"]
            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, 0, None)   # Sales cannot be negative

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = self._mape(y_test.values, y_pred)
            r2 = r2_score(y_test, y_pred)
            cv_rmse = info.get("cv_rmse", np.nan)

            rows.append({
                "Model": name,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "MAPE (%)": round(mape, 2),
                "R²": round(r2, 4),
                "CV_RMSE": round(cv_rmse, 2),
                "Training_Time_s": round(info.get("training_time", 0), 1),
            })
            predictions[name] = y_pred

            logger.info(
                "%-15s → MAE=%.2f | RMSE=%.2f | MAPE=%.2f%% | R²=%.4f",
                name, mae, rmse, mape, r2,
            )

        self.metrics_df = pd.DataFrame(rows).sort_values("RMSE")

        # Persist report
        report_path = self.reports_dir / "model_evaluation_report.csv"
        self.metrics_df.to_csv(report_path, index=False)
        logger.info("Evaluation report saved → %s", report_path)

        # Visualisations
        self.plot_model_comparison(self.metrics_df)
        best_model_name = self.metrics_df.iloc[0]["Model"]
        if best_model_name in predictions:
            self.plot_actual_vs_predicted(
                y_test.values, predictions[best_model_name], best_model_name
            )
            self.plot_residuals(
                y_test.values, predictions[best_model_name], best_model_name
            )

        return self.metrics_df

    # ----------------------------------------------------------
    # Metric helpers
    # ----------------------------------------------------------

    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute MAPE, safely ignoring zero-truth rows."""
        mask = y_true > 0
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    # ----------------------------------------------------------
    # Visualisations
    # ----------------------------------------------------------

    def plot_model_comparison(self, metrics_df: pd.DataFrame) -> None:
        """Grouped bar chart comparing all metric values per model."""
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        fig.patch.set_facecolor(BG_COLOR)
        fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

        metrics = [("MAE", "lower=better"), ("RMSE", "lower=better"),
                   ("MAPE (%)", "lower=better"), ("R²", "higher=better")]
        colors = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff"]

        for ax, (metric, note), color in zip(axes, metrics, colors):
            vals = metrics_df[metric].values
            bars = ax.bar(metrics_df["Model"], vals, color=color, alpha=0.85,
                          edgecolor="none", width=0.55)
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=9, color=TEXT_COLOR,
                )
            ax.set_title(f"{metric}\n({note})")
            ax.set_ylabel(metric)
            ax.grid(True, axis="y")
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

        self._save(fig, "09_model_comparison.png")

    def plot_actual_vs_predicted(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> None:
        """Scatter plot of actual vs. predicted sales (best model)."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(BG_COLOR)

        # Scatter
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.15, s=6, color=ACCENT_COLOR)
        lim = max(y_true.max(), y_pred.max()) * 1.05
        ax.plot([0, lim], [0, lim], color=HIGHLIGHT_COLOR, linewidth=1.5,
                linestyle="--", label="Perfect prediction")
        ax.set_title(f"{model_name}: Actual vs. Predicted")
        ax.set_xlabel("Actual Sales (€)")
        ax.set_ylabel("Predicted Sales (€)")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend()
        ax.grid(True)

        # Time-series slice (first 120 test points for clarity)
        ax2 = axes[1]
        n = min(120, len(y_true))
        ax2.plot(range(n), y_true[:n], color=ACCENT_COLOR, linewidth=1.2,
                 label="Actual", alpha=0.9)
        ax2.plot(range(n), y_pred[:n], color=HIGHLIGHT_COLOR, linewidth=1.2,
                 label="Predicted", alpha=0.9)
        ax2.set_title(f"{model_name}: Forecast vs. Actual (first {n} test pts)")
        ax2.set_xlabel("Test Sample Index")
        ax2.set_ylabel("Sales (€)")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax2.legend()
        ax2.grid(True)

        fig.suptitle(f"{model_name} — Prediction Quality", fontsize=15, fontweight="bold")
        self._save(fig, "10_actual_vs_predicted.png")

    def plot_residuals(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> None:
        """Residual distribution and residuals vs. predicted scatter."""
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(BG_COLOR)

        # Residual distribution
        axes[0].hist(residuals, bins=60, color=GREEN_COLOR, alpha=0.75,
                     edgecolor="none")
        axes[0].axvline(0, color=HIGHLIGHT_COLOR, linewidth=1.5, linestyle="--")
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residual (Actual − Predicted)")
        axes[0].set_ylabel("Count")
        axes[0].grid(True, axis="y")

        # Residuals vs. predicted
        axes[1].scatter(y_pred, residuals, alpha=0.12, s=5, color=ACCENT_COLOR)
        axes[1].axhline(0, color=HIGHLIGHT_COLOR, linewidth=1.5, linestyle="--")
        axes[1].set_title("Residuals vs. Predicted Values")
        axes[1].set_xlabel("Predicted Sales (€)")
        axes[1].set_ylabel("Residual")
        axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        axes[1].grid(True)

        fig.suptitle(f"{model_name} — Residual Analysis", fontsize=15, fontweight="bold")
        self._save(fig, "11_residual_plot.png")

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.viz_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        logger.info("Saved plot: %s", path.name)
