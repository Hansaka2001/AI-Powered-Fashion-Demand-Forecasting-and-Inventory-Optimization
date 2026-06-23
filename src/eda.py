"""
src/eda.py
----------
Exploratory Data Analysis module.

Produces publication-quality visualisations saved to
``outputs/visualizations/`` covering:

- Sales trend over time (rolling average)
- Monthly seasonality bar chart
- Day-of-week demand distribution
- Sales distribution (histogram + KDE)
- Correlation heatmap
- Store-type comparison
- Promotion impact analysis
- Quarterly decomposition

Classes
-------
EDAAnalyzer
    Run the full EDA suite and save all plots.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Shared plot style
# ──────────────────────────────────────────────────────────────
PALETTE = "viridis"
BG_COLOR = "#0d1117"
GRID_COLOR = "#21262d"
TEXT_COLOR = "#c9d1d9"
ACCENT_COLOR = "#58a6ff"
HIGHLIGHT_COLOR = "#f78166"


def _apply_dark_style() -> None:
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
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
    })


class EDAAnalyzer:
    """
    Produces a comprehensive suite of EDA visualisations.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed and feature-engineered DataFrame.
    output_dir : Path, optional
        Directory to save visualisations.  Defaults to
        ``config.VISUALIZATIONS_DIR``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: Path = VISUALIZATIONS_DIR,
    ) -> None:
        self.df = df.copy()
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _apply_dark_style()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run_all(self) -> None:
        """Execute all EDA plots."""
        logger.info("Running full EDA suite …")
        self.plot_sales_trend()
        self.plot_monthly_seasonality()
        self.plot_day_of_week_demand()
        self.plot_sales_distribution()
        self.plot_correlation_heatmap()
        self.plot_store_type_comparison()
        self.plot_promotion_impact()
        self.plot_quarterly_decomposition()
        logger.info("EDA complete. Plots saved to: %s", self.output_dir)

    # ----------------------------------------------------------
    # Individual plots
    # ----------------------------------------------------------

    def plot_sales_trend(self) -> None:
        """Aggregate daily sales + 30-day rolling average trend."""
        daily = (
            self.df.groupby("Date")["Sales"]
            .sum()
            .reset_index()
            .sort_values("Date")
        )
        daily["Rolling30"] = daily["Sales"].rolling(30, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor(BG_COLOR)

        ax.fill_between(
            daily["Date"], daily["Sales"],
            alpha=0.25, color=ACCENT_COLOR, label="Daily Sales",
        )
        ax.plot(
            daily["Date"], daily["Sales"],
            color=ACCENT_COLOR, linewidth=0.6, alpha=0.6,
        )
        ax.plot(
            daily["Date"], daily["Rolling30"],
            color=HIGHLIGHT_COLOR, linewidth=2.0, label="30-Day Rolling Avg",
        )

        ax.set_title("Total Sales Trend Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales (€)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.legend()
        ax.grid(True)

        self._save(fig, "01_sales_trend.png")

    def plot_monthly_seasonality(self) -> None:
        """Average sales by month — seasonal pattern."""
        month_labels = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        monthly = (
            self.df.groupby("Month")["Sales"].mean().reindex(range(1, 13))
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(BG_COLOR)

        colors = plt.cm.viridis(np.linspace(0.2, 0.9, 12))
        bars = ax.bar(range(1, 13), monthly.values, color=colors, edgecolor="none", width=0.7)

        # Value labels
        for bar, val in zip(bars, monthly.values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + monthly.max() * 0.01,
                    f"€{val:,.0f}",
                    ha="center", va="bottom", fontsize=9, color=TEXT_COLOR,
                )

        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels)
        ax.set_title("Average Daily Sales by Month (Seasonality)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Avg Sales (€)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.grid(True, axis="y")

        self._save(fig, "02_monthly_seasonality.png")

    def plot_day_of_week_demand(self) -> None:
        """Violin + box plot of Sales by day of week."""
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        df_plot = self.df[self.df["DayOfWeek"].between(1, 7)].copy()
        df_plot["DayLabel"] = df_plot["DayOfWeek"].map(
            {i + 1: l for i, l in enumerate(dow_labels)}
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(BG_COLOR)

        palette = sns.color_palette("viridis", 7)
        sns.violinplot(
            data=df_plot,
            x="DayLabel", y="Sales",
            order=dow_labels,
            palette=palette,
            inner="box",
            ax=ax,
            linewidth=0.8,
        )
        ax.set_title("Sales Distribution by Day of Week")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Sales (€)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.grid(True, axis="y")

        self._save(fig, "03_day_of_week_demand.png")

    def plot_sales_distribution(self) -> None:
        """Histogram + KDE of Sales values."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(BG_COLOR)

        sales = self.df["Sales"]

        # Raw distribution
        axes[0].hist(
            sales, bins=80, color=ACCENT_COLOR, alpha=0.75, edgecolor="none",
        )
        axes[0].set_title("Sales Distribution (Raw)")
        axes[0].set_xlabel("Sales (€)")
        axes[0].set_ylabel("Frequency")
        axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        axes[0].grid(True, axis="y")

        # Log-transformed
        log_sales = np.log1p(sales)
        axes[1].hist(
            log_sales, bins=80, color=HIGHLIGHT_COLOR, alpha=0.75, edgecolor="none",
        )
        axes[1].set_title("Sales Distribution (Log-Transformed)")
        axes[1].set_xlabel("log(1 + Sales)")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, axis="y")

        fig.suptitle("Demand Distribution Analysis", fontsize=16, fontweight="bold")
        self._save(fig, "04_sales_distribution.png")

    def plot_correlation_heatmap(self) -> None:
        """Correlation heatmap of numerical features vs. Sales."""
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Keep a manageable subset
        priority_cols = [
            "Sales", "DayOfWeek", "Month", "Quarter", "IsWeekend",
            "Promo", "PromoActive", "SchoolHoliday", "StateHoliday_enc",
            "CompetitionDistance", "CompetitionDistanceLog",
            "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_30",
            "Sales_rolling_mean_7", "Sales_rolling_mean_14", "Sales_rolling_mean_30",
            "IsHolidaySeason", "StoreType_enc",
        ]
        cols = [c for c in priority_cols if c in num_cols][:20]
        corr = self.df[cols].corr()

        fig, ax = plt.subplots(figsize=(14, 11))
        fig.patch.set_facecolor(BG_COLOR)

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            linecolor=GRID_COLOR,
            annot_kws={"size": 8},
        )
        ax.set_title("Feature Correlation Heatmap")

        self._save(fig, "05_correlation_heatmap.png")

    def plot_store_type_comparison(self) -> None:
        """Box plot comparing sales across store types."""
        if "StoreType" not in self.df.columns:
            logger.warning("StoreType column not found, skipping store-type plot.")
            return

        type_map = {1: "Type A", 2: "Type B", 3: "Type C", 4: "Type D"}
        if "StoreType_enc" in self.df.columns:
            df_plot = self.df.copy()
            df_plot["StoreTypeLabel"] = df_plot["StoreType_enc"].map(type_map).fillna("Other")
        else:
            df_plot = self.df.copy()
            df_plot["StoreTypeLabel"] = df_plot["StoreType"].str.upper().apply(
                lambda x: f"Type {x}"
            )

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(BG_COLOR)

        palette = sns.color_palette("viridis", df_plot["StoreTypeLabel"].nunique())
        sns.boxplot(
            data=df_plot,
            x="StoreTypeLabel", y="Sales",
            palette=palette,
            ax=ax,
            fliersize=2,
            linewidth=0.8,
        )
        ax.set_title("Sales Distribution by Store Type")
        ax.set_xlabel("Store Type")
        ax.set_ylabel("Daily Sales (€)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
        ax.grid(True, axis="y")

        self._save(fig, "06_store_type_comparison.png")

    def plot_promotion_impact(self) -> None:
        """Bar chart comparing average sales with/without promotions."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(BG_COLOR)

        for ax, col, title in zip(
            axes,
            ["Promo", "SchoolHoliday"],
            ["Regular Promo Impact", "School Holiday Impact"],
        ):
            if col not in self.df.columns:
                continue
            grouped = self.df.groupby(col)["Sales"].mean()
            labels = ["No", "Yes"]
            values = [grouped.get(0, 0), grouped.get(1, 0)]
            colors = [GRID_COLOR, ACCENT_COLOR]

            bars = ax.bar(labels, values, color=colors, edgecolor="none", width=0.5)
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"€{val:,.0f}",
                    ha="center", va="bottom", fontsize=11, color=TEXT_COLOR,
                )

            lift = (values[1] / values[0] - 1) * 100 if values[0] else 0
            ax.set_title(f"{title}\n(Lift: {lift:+.1f}%)")
            ax.set_ylabel("Avg Daily Sales (€)")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))
            ax.grid(True, axis="y")

        fig.suptitle("Promotion & Holiday Impact on Sales", fontsize=15, fontweight="bold")
        self._save(fig, "07_promotion_impact.png")

    def plot_quarterly_decomposition(self) -> None:
        """Total sales by quarter across years."""
        df_q = self.df.copy()
        df_q["YearQuarter"] = (
            df_q["Year"].astype(str) + " Q" + df_q["Quarter"].astype(str)
        )
        quarterly = df_q.groupby("YearQuarter")["Sales"].sum().reset_index()
        quarterly = quarterly.sort_values("YearQuarter")

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor(BG_COLOR)

        colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(quarterly)))
        bars = ax.bar(
            quarterly["YearQuarter"], quarterly["Sales"],
            color=colors, edgecolor="none", width=0.7,
        )
        ax.set_title("Total Sales by Quarter")
        ax.set_xlabel("Year-Quarter")
        ax.set_ylabel("Total Sales (€)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x/1e6:.1f}M"))
        plt.xticks(rotation=45, ha="right")
        ax.grid(True, axis="y")

        self._save(fig, "08_quarterly_decomposition.png")

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> None:
        """Save figure and close it."""
        path = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        logger.info("Saved plot: %s", path.name)
