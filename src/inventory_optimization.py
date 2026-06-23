"""
src/inventory_optimization.py
------------------------------
Calculates inventory planning recommendations from demand
forecasts using classical inventory theory.

Metrics computed per store / period
------------------------------------
- Safety Stock    : buffer against demand variability
- Reorder Point   : trigger level for re-ordering
- Recommended Stock: forecast demand + safety stock
- Stockout Risk   : probability that demand exceeds current_stock
- EOQ             : Economic Order Quantity
- Days of Supply  : recommended_stock / avg_daily_demand

Classes
-------
InventoryOptimizer
    Accept forecast DataFrames and compute inventory recommendations.
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

from config import INVENTORY_CONFIG, REPORTS_DIR, VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)

BG_COLOR = "#0d1117"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"
ACCENT_COLOR = "#58a6ff"
HIGHLIGHT_COLOR = "#f78166"
GREEN_COLOR = "#3fb950"
WARN_COLOR = "#e3b341"


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


class InventoryOptimizer:
    """
    Generates inventory recommendations from demand forecasts.

    Uses standard inventory theory formulas:
        Safety Stock  = Z × σ_demand × √(lead_time)
        Reorder Point = μ_demand × lead_time + Safety_Stock
        EOQ           = √(2 × D × S / H)

    Parameters
    ----------
    config : dict, optional
        Inventory configuration.  Defaults to ``config.INVENTORY_CONFIG``.
    reports_dir : Path, optional
    viz_dir : Path, optional
    """

    def __init__(
        self,
        config: dict = None,
        reports_dir: Path = REPORTS_DIR,
        viz_dir: Path = VISUALIZATIONS_DIR,
    ) -> None:
        self.cfg = config or INVENTORY_CONFIG
        self.reports_dir = reports_dir
        self.viz_dir = viz_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        _dark_style()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def optimize(
        self,
        forecast_map: Dict[int, pd.DataFrame],
        historical_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute inventory recommendations for all stores and horizons.

        Parameters
        ----------
        forecast_map : dict
            ``{horizon: forecast_DataFrame}`` from DemandForecaster.
        historical_df : pd.DataFrame
            Full historical DataFrame for computing demand std dev.

        Returns
        -------
        pd.DataFrame
            Inventory recommendations, one row per (Store, Horizon).
        """
        logger.info("Running inventory optimisation …")

        # Pre-compute historical demand stats per store
        demand_stats = self._compute_demand_stats(historical_df)

        records = []
        for horizon, forecast_df in forecast_map.items():
            for store_id, group in forecast_df.groupby("Store"):
                rec = self._compute_store_inventory(
                    store_id, horizon, group, demand_stats
                )
                records.append(rec)

        result_df = pd.DataFrame(records).sort_values(
            ["Store", "Forecast_Horizon_Days"]
        )

        out_path = self.reports_dir / "inventory_recommendations.csv"
        result_df.to_csv(out_path, index=False)
        logger.info("Inventory recommendations saved → %s", out_path)

        # Visualisations
        self._plot_safety_stock(result_df)
        self._plot_stockout_risk(result_df)
        self._plot_reorder_dashboard(result_df)

        return result_df

    # ----------------------------------------------------------
    # Core inventory calculations
    # ----------------------------------------------------------

    def _compute_demand_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-store mean and std of daily Sales."""
        stats_df = (
            df.groupby("Store")["Sales"]
            .agg(mean_demand="mean", std_demand="std")
            .fillna(0)
            .reset_index()
        )
        return stats_df

    def _compute_store_inventory(
        self,
        store_id: int,
        horizon: int,
        forecast_df: pd.DataFrame,
        demand_stats: pd.DataFrame,
    ) -> dict:
        """
        Compute all inventory metrics for one store × horizon pair.
        """
        lead_time = self.cfg["lead_time_days"]
        z = self.cfg["z_score"]

        # Forecasted demand totals over the horizon
        total_forecast = forecast_df["Forecast_Sales"].sum()
        avg_daily_forecast = forecast_df["Forecast_Sales"].mean()

        # Historical demand stats for this store
        stat_row = demand_stats[demand_stats["Store"] == store_id]
        if len(stat_row) == 0:
            hist_std = avg_daily_forecast * 0.15
        else:
            hist_std = float(stat_row["std_demand"].iloc[0])

        # Inventory formulas
        safety_stock = z * hist_std * np.sqrt(lead_time)
        reorder_point = avg_daily_forecast * lead_time + safety_stock
        recommended_stock = total_forecast + safety_stock

        # EOQ: Economic Order Quantity
        # EOQ = sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)
        annual_demand = avg_daily_forecast * 365
        holding_cost = self.cfg["holding_cost_rate"] * self.cfg["unit_cost"]
        if holding_cost > 0 and annual_demand > 0:
            eoq = np.sqrt(2 * annual_demand * self.cfg["ordering_cost"] / holding_cost)
        else:
            eoq = 0

        # Days of supply covered by recommended stock
        days_of_supply = (
            recommended_stock / avg_daily_forecast if avg_daily_forecast > 0 else 0
        )

        # Stockout risk: P(demand > recommended_stock) assuming Normal demand
        if hist_std > 0 and avg_daily_forecast > 0:
            z_score_stockout = (recommended_stock - total_forecast) / (
                hist_std * np.sqrt(horizon)
            )
            stockout_risk_pct = max(0.0, (1 - stats.norm.cdf(z_score_stockout)) * 100)
        else:
            stockout_risk_pct = 0.0

        # Risk label
        if stockout_risk_pct < 5:
            risk_label = "LOW"
        elif stockout_risk_pct < 15:
            risk_label = "MEDIUM"
        else:
            risk_label = "HIGH"

        return {
            "Store": store_id,
            "Forecast_Horizon_Days": horizon,
            "Total_Forecast_Sales": round(total_forecast, 2),
            "Avg_Daily_Forecast": round(avg_daily_forecast, 2),
            "Safety_Stock_Units": round(safety_stock, 0),
            "Reorder_Point": round(reorder_point, 0),
            "Recommended_Stock_Level": round(recommended_stock, 0),
            "EOQ_Units": round(eoq, 0),
            "Days_of_Supply": round(days_of_supply, 1),
            "Stockout_Risk_Pct": round(stockout_risk_pct, 2),
            "Risk_Level": risk_label,
            "Lead_Time_Days": lead_time,
            "Service_Level_Pct": round(self.cfg["service_level"] * 100, 1),
        }

    # ----------------------------------------------------------
    # Visualisations
    # ----------------------------------------------------------

    def _plot_safety_stock(self, df: pd.DataFrame) -> None:
        """Bar chart of safety stock per store for the 7-day horizon."""
        subset = df[df["Forecast_Horizon_Days"] == 7].head(15)

        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor(BG_COLOR)

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(subset)))
        bars = ax.barh(
            subset["Store"].astype(str), subset["Safety_Stock_Units"],
            color=colors, edgecolor="none",
        )
        for bar, val in zip(bars, subset["Safety_Stock_Units"]):
            ax.text(
                bar.get_width() + subset["Safety_Stock_Units"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"€{val:,.0f}",
                va="center", fontsize=9, color=TEXT_COLOR,
            )

        ax.set_title("Safety Stock Requirements by Store (7-Day Horizon)")
        ax.set_xlabel("Safety Stock (€-value units)")
        ax.set_ylabel("Store ID")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.grid(True, axis="x")
        self._save(fig, "14_safety_stock.png")

    def _plot_stockout_risk(self, df: pd.DataFrame) -> None:
        """Scatter plot of stockout risk vs. recommended stock."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(BG_COLOR)

        color_map = {"LOW": GREEN_COLOR, "MEDIUM": WARN_COLOR, "HIGH": HIGHLIGHT_COLOR}

        for ax, horizon in zip(axes, [7, 30]):
            subset = df[df["Forecast_Horizon_Days"] == horizon]
            if subset.empty:
                continue

            colors_scatter = subset["Risk_Level"].map(color_map).fillna(ACCENT_COLOR)
            sc = ax.scatter(
                subset["Recommended_Stock_Level"],
                subset["Stockout_Risk_Pct"],
                c=colors_scatter,
                s=80, alpha=0.85, edgecolors="none",
            )

            # Legend
            for label, color in color_map.items():
                ax.scatter([], [], c=color, label=label, s=60)
            ax.legend(title="Risk Level", title_fontsize=9)

            ax.set_title(f"Stockout Risk vs. Stock Level ({horizon}-Day)")
            ax.set_xlabel("Recommended Stock Level (€)")
            ax.set_ylabel("Stockout Risk (%)")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
            ax.grid(True)

        fig.suptitle("Stockout Risk Analysis", fontsize=15, fontweight="bold")
        self._save(fig, "15_stockout_risk.png")

    def _plot_reorder_dashboard(self, df: pd.DataFrame) -> None:
        """Summary dashboard: reorder points, EOQ, days of supply."""
        subset_7 = df[df["Forecast_Horizon_Days"] == 7].head(10)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor(BG_COLOR)
        fig.suptitle("Inventory Optimisation Dashboard (Top 10 Stores, 7-Day)",
                     fontsize=14, fontweight="bold")

        stores = subset_7["Store"].astype(str)
        metrics = [
            ("Reorder_Point",  "Reorder Point (€)", ACCENT_COLOR),
            ("EOQ_Units",      "EOQ (units)",        "#d2a8ff"),
            ("Days_of_Supply", "Days of Supply",     GREEN_COLOR),
        ]

        for ax, (col, label, color) in zip(axes, metrics):
            ax.barh(stores, subset_7[col], color=color, edgecolor="none")
            ax.set_title(label)
            ax.set_xlabel(label)
            ax.set_ylabel("Store ID")
            ax.grid(True, axis="x")
            if "€" in label or "Point" in label:
                ax.xaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}")
                )

        self._save(fig, "16_inventory_dashboard.png")

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self.viz_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        logger.info("Saved plot: %s", path.name)
