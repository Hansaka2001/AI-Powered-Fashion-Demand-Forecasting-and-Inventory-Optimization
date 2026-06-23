"""
src/forecasting.py
------------------
Generates 7-day and 30-day demand forecasts using the best
trained model.  Builds the future feature matrix iteratively
to propagate lag/rolling values forward.

Outputs
-------
- outputs/reports/forecast_7day.csv
- outputs/reports/forecast_30day.csv
- outputs/visualizations/12_forecast_7day.png
- outputs/visualizations/13_forecast_30day.png

Classes
-------
DemandForecaster
    Load the best model and generate multi-step-ahead forecasts.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from config import (
    MODELS_DIR,
    REPORTS_DIR,
    VISUALIZATIONS_DIR,
    FEATURE_COLUMNS,
    LAG_PERIODS,
    ROLLING_WINDOWS,
    FORECAST_HORIZONS,
    FORECAST_STORE_SAMPLE,
)

logger = logging.getLogger(__name__)

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"
ACCENT_COLOR = "#58a6ff"
HIGHLIGHT_COLOR = "#f78166"
BAND_COLOR = "#1f6feb"


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


class DemandForecaster:
    """
    Uses the best saved model to forecast future demand for
    a sample of stores over 7-day and 30-day horizons.

    Parameters
    ----------
    models_dir : Path
        Directory containing ``best_model.joblib``.
    reports_dir : Path
        Directory for CSV output.
    viz_dir : Path
        Directory for plot output.
    store_sample : int
        Number of stores to generate forecasts for.
    forecast_horizons : list of int
        Forecast horizon lengths in days.
    """

    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        reports_dir: Path = REPORTS_DIR,
        viz_dir: Path = VISUALIZATIONS_DIR,
        store_sample: int = FORECAST_STORE_SAMPLE,
        forecast_horizons: List[int] = None,
    ) -> None:
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.viz_dir = viz_dir
        self.store_sample = store_sample
        self.forecast_horizons = forecast_horizons or FORECAST_HORIZONS
        self.model: Any = None
        self.feature_cols: List[str] = []
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        _dark_style()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def load_model(self, model: Any = None, feature_cols: List[str] = None) -> None:
        """
        Load the best trained model, either from disk or passed directly.

        Parameters
        ----------
        model : fitted estimator, optional
            If provided, uses this model directly.
        feature_cols : list of str, optional
            Feature column names used during training.
        """
        if model is not None:
            self.model = model
        else:
            path = self.models_dir / "best_model.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            self.model = joblib.load(path)
            logger.info("Best model loaded from: %s", path)

        self.feature_cols = feature_cols or [
            c for c in FEATURE_COLUMNS
        ]

    def forecast_all_horizons(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Generate forecasts for each horizon in ``self.forecast_horizons``.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature-engineered historical DataFrame.

        Returns
        -------
        dict
            ``{horizon_days: forecast_DataFrame}``
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        stores = df["Store"].unique()
        if len(stores) > self.store_sample:
            stores = stores[: self.store_sample]

        results = {}
        for horizon in self.forecast_horizons:
            logger.info("Generating %d-day forecast for %d stores …", horizon, len(stores))
            all_store_forecasts = []

            for store_id in stores:
                store_df = df[df["Store"] == store_id].sort_values("Date").copy()
                forecast_df = self._forecast_store(store_id, store_df, horizon)
                all_store_forecasts.append(forecast_df)

            combined = pd.concat(all_store_forecasts, ignore_index=True)
            results[horizon] = combined

            # Save CSV
            out_path = self.reports_dir / f"forecast_{horizon}day.csv"
            combined.to_csv(out_path, index=False)
            logger.info("%d-day forecast saved → %s", horizon, out_path)

            # Plot (for first store)
            first_store = stores[0]
            self._plot_forecast(
                df[df["Store"] == first_store].sort_values("Date"),
                combined[combined["Store"] == first_store],
                horizon,
                first_store,
            )

        return results

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _forecast_store(
        self, store_id: int, store_df: pd.DataFrame, horizon: int
    ) -> pd.DataFrame:
        """
        Build an iterative multi-step forecast for one store.

        For each future day we:
        1. Construct the feature row using known static/calendar features
        2. Fill lag features from the running history (real + predicted)
        3. Fill rolling features from the same history
        4. Predict and append to history
        """
        last_date = store_df["Date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        # Running history for lag/rolling computation
        history = store_df["Sales"].tolist()
        history_dates = store_df["Date"].tolist()

        static_cols = {
            "Store": store_id,
            "StoreType_enc":           store_df["StoreType_enc"].iloc[-1] if "StoreType_enc" in store_df else 1,
            "Assortment_enc":          store_df["Assortment_enc"].iloc[-1] if "Assortment_enc" in store_df else 1,
            "CompetitionDistance":     store_df["CompetitionDistance"].iloc[-1] if "CompetitionDistance" in store_df else 1000,
            "CompetitionOpenMonths":   store_df["CompetitionOpenMonths"].iloc[-1] if "CompetitionOpenMonths" in store_df else 12,
            "HasCompetition":          store_df["HasCompetition"].iloc[-1] if "HasCompetition" in store_df else 1,
            "CompetitionDistanceLog":  store_df["CompetitionDistanceLog"].iloc[-1] if "CompetitionDistanceLog" in store_df else 6.9,
            "Promo2":                  store_df["Promo2"].iloc[-1] if "Promo2" in store_df else 0,
        }

        forecast_records = []
        for future_date in future_dates:
            row = self._build_feature_row(
                future_date, history, history_dates, static_cols
            )
            feat_values = {col: row.get(col, 0) for col in self.feature_cols}
            X_row = pd.DataFrame([feat_values])

            # Align columns
            for col in self.feature_cols:
                if col not in X_row.columns:
                    X_row[col] = 0
            X_row = X_row[self.feature_cols]

            pred_sales = float(self.model.predict(X_row)[0])
            pred_sales = max(0, pred_sales)

            # Simple confidence band (±15% of prediction)
            lower = pred_sales * 0.85
            upper = pred_sales * 1.15

            forecast_records.append({
                "Store": store_id,
                "Date": future_date.strftime("%Y-%m-%d"),
                "Forecast_Sales": round(pred_sales, 2),
                "Lower_CI_85": round(lower, 2),
                "Upper_CI_85": round(upper, 2),
                "DayOfWeek": future_date.dayofweek + 1,
                "IsWeekend": int(future_date.dayofweek >= 5),
            })

            # Append prediction to history for next step's lags
            history.append(pred_sales)
            history_dates.append(future_date)

        return pd.DataFrame(forecast_records)

    def _build_feature_row(
        self,
        date: pd.Timestamp,
        history: List[float],
        history_dates: list,
        static_cols: dict,
    ) -> dict:
        """Build a single feature row for a future date."""
        row = dict(static_cols)

        # Calendar features
        row["DayOfWeek"] = date.dayofweek + 1
        row["Day"] = date.day
        row["Week"] = date.isocalendar()[1]
        row["Month"] = date.month
        row["Quarter"] = date.quarter
        row["Year"] = date.year
        row["DayOfYear"] = date.timetuple().tm_yday
        row["IsWeekend"] = int(date.dayofweek >= 5)
        row["IsMonthStart"] = int(date.day == 1)
        row["IsMonthEnd"] = int(date.day == pd.Timestamp(date.year, date.month, 1).days_in_month)

        # Seasonal
        m = date.month
        row["IsSpring"] = int(m in (3, 4, 5))
        row["IsSummer"] = int(m in (6, 7, 8))
        row["IsAutumn"] = int(m in (9, 10, 11))
        row["IsWinter"] = int(m in (12, 1, 2))
        row["IsHolidaySeason"] = int(m == 12 or (m == 11 and date.day >= 15))

        # Promo (alternating weekly heuristic)
        row["Promo"] = int(date.isocalendar()[1] % 2 == 0)
        row["PromoActive"] = int(row["Promo"] or static_cols.get("Promo2", 0))
        row["SchoolHoliday"] = int(m in (7, 8) or (m == 12 and date.day >= 23))
        row["StateHoliday_enc"] = 0

        # Lag features from history
        n = len(history)
        for lag in LAG_PERIODS:
            col = f"Sales_lag_{lag}"
            row[col] = history[n - lag] if n >= lag else np.nan

        # Rolling features
        for window in ROLLING_WINDOWS:
            window_data = history[max(0, n - window):n]
            if window_data:
                row[f"Sales_rolling_mean_{window}"] = float(np.mean(window_data))
                row[f"Sales_rolling_std_{window}"] = float(np.std(window_data)) if len(window_data) > 1 else 0.0
            else:
                row[f"Sales_rolling_mean_{window}"] = 0.0
                row[f"Sales_rolling_std_{window}"] = 0.0

        return row

    def _plot_forecast(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        horizon: int,
        store_id: int,
    ) -> None:
        """Plot historical sales + forecast with confidence band."""
        # Show last 60 days of history
        hist = historical_df.tail(60).copy()
        hist["Date"] = pd.to_datetime(hist["Date"])
        forecast_df = forecast_df.copy()
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor(BG_COLOR)

        ax.plot(
            hist["Date"], hist["Sales"],
            color=ACCENT_COLOR, linewidth=1.5, label="Historical Sales",
        )
        ax.plot(
            forecast_df["Date"], forecast_df["Forecast_Sales"],
            color=HIGHLIGHT_COLOR, linewidth=2.0, linestyle="--",
            label=f"{horizon}-Day Forecast", marker="o", markersize=4,
        )
        ax.fill_between(
            forecast_df["Date"],
            forecast_df["Lower_CI_85"],
            forecast_df["Upper_CI_85"],
            alpha=0.25, color=HIGHLIGHT_COLOR, label="85% Confidence Band",
        )

        # Vertical divider
        transition = hist["Date"].max()
        ax.axvline(x=transition, color=TEXT_COLOR, linewidth=1.0,
                   linestyle=":", alpha=0.7)

        ax.set_title(f"Store {store_id} — {horizon}-Day Demand Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales (€)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend()
        ax.grid(True)

        filename = f"1{'2' if horizon == 7 else '3'}_forecast_{horizon}day.png"
        path = self.viz_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        logger.info("Forecast plot saved: %s", path.name)
