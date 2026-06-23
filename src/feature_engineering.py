"""
src/feature_engineering.py
---------------------------
Builds all time-series features required for the ML models.

Features generated
------------------
- Date features: day, week, month, quarter, year, day_of_year,
  is_weekend, is_month_start, is_month_end
- Seasonal indicators: spring, summer, autumn, winter, holiday_season
- Lag features: Sales at t-1, t-7, t-14, t-30
- Rolling statistics: mean & std over 7, 14, 30-day windows
- Promotion flags: Promo, Promo2, promo_active
- Holiday indicators: StateHoliday, SchoolHoliday, is_holiday_season
- Competition features (derived in preprocessing, reused here)

Classes
-------
FeatureEngineer
    Transform a clean DataFrame into a feature matrix.
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import (
    LAG_PERIODS,
    ROLLING_WINDOWS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Generates the complete feature set for demand forecasting.

    Parameters
    ----------
    lag_periods : list of int
        Lag offsets in days.  Defaults to ``config.LAG_PERIODS``.
    rolling_windows : list of int
        Rolling window sizes in days.  Defaults to ``config.ROLLING_WINDOWS``.
    """

    def __init__(
        self,
        lag_periods: List[int] = None,
        rolling_windows: List[int] = None,
    ) -> None:
        self.lag_periods = lag_periods or LAG_PERIODS
        self.rolling_windows = rolling_windows or ROLLING_WINDOWS

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps and return the
        enriched DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Clean DataFrame from DataPreprocessor (must have
            'Date', 'Sales', 'Store' columns).

        Returns
        -------
        pd.DataFrame
            DataFrame with all engineered features added.
        """
        logger.info("Starting feature engineering …")
        df = df.copy()

        df = self._add_date_features(df)
        df = self._add_seasonal_features(df)
        df = self._add_holiday_season_flag(df)
        df = self._add_promo_active(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)

        # Drop rows where lags/rolling produce NaN (first N days per store)
        before = len(df)
        df.dropna(subset=self._lag_col_names() + self._rolling_col_names(), inplace=True)
        logger.info(
            "Dropped %d rows with NaN lag/rolling features. Remaining: %d",
            before - len(df), len(df),
        )
        logger.info("Feature engineering complete. Shape: %s", df.shape)
        return df

    def get_feature_matrix(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract the final (X, y) split from the engineered DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Engineered DataFrame.

        Returns
        -------
        (X, y) : Tuple[pd.DataFrame, pd.Series]
        """
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            logger.warning("Missing expected feature columns: %s", missing)

        X = df[available].copy()
        y = df[TARGET_COLUMN].copy()
        logger.info("Feature matrix X: %s  |  Target y: %d rows", X.shape, len(y))
        return X, y

    # ----------------------------------------------------------
    # Date & calendar features
    # ----------------------------------------------------------

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract calendar components from the Date column."""
        dt = df["Date"]
        df["DayOfWeek"] = dt.dt.dayofweek + 1   # 1=Mon … 7=Sun (align with raw data)
        df["Day"] = dt.dt.day
        df["Week"] = dt.dt.isocalendar().week.astype(int)
        df["Month"] = dt.dt.month
        df["Quarter"] = dt.dt.quarter
        df["Year"] = dt.dt.year
        df["DayOfYear"] = dt.dt.dayofyear
        df["IsWeekend"] = (dt.dt.dayofweek >= 5).astype(int)
        df["IsMonthStart"] = dt.dt.is_month_start.astype(int)
        df["IsMonthEnd"] = dt.dt.is_month_end.astype(int)
        logger.debug("Date features added.")
        return df

    # ----------------------------------------------------------
    # Seasonal features
    # ----------------------------------------------------------

    def _add_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add meteorological season indicator columns."""
        m = df["Month"]
        df["IsSpring"] = m.isin([3, 4, 5]).astype(int)
        df["IsSummer"] = m.isin([6, 7, 8]).astype(int)
        df["IsAutumn"] = m.isin([9, 10, 11]).astype(int)
        df["IsWinter"] = m.isin([12, 1, 2]).astype(int)
        logger.debug("Seasonal features added.")
        return df

    def _add_holiday_season_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag the key holiday shopping window (Nov 15 – Dec 31)."""
        df["IsHolidaySeason"] = (
            ((df["Month"] == 12) | ((df["Month"] == 11) & (df["Day"] >= 15)))
        ).astype(int)
        return df

    # ----------------------------------------------------------
    # Promotion features
    # ----------------------------------------------------------

    def _add_promo_active(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive a combined promotion-active flag.

        PromoActive = 1 if either Promo or Promo2 is active.
        """
        promo = df.get("Promo", pd.Series(0, index=df.index))
        promo2 = df.get("Promo2", pd.Series(0, index=df.index))
        df["PromoActive"] = ((promo == 1) | (promo2 == 1)).astype(int)
        return df

    # ----------------------------------------------------------
    # Lag features
    # ----------------------------------------------------------

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add shifted Sales values as lag features per store."""
        df = df.sort_values(["Store", "Date"])
        for lag in self.lag_periods:
            col_name = f"Sales_lag_{lag}"
            df[col_name] = df.groupby("Store")["Sales"].shift(lag)
            logger.debug("Lag feature added: %s", col_name)
        return df

    def _lag_col_names(self) -> List[str]:
        return [f"Sales_lag_{lag}" for lag in self.lag_periods]

    # ----------------------------------------------------------
    # Rolling features
    # ----------------------------------------------------------

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling mean and std of Sales per store."""
        df = df.sort_values(["Store", "Date"])
        for window in self.rolling_windows:
            # Shift by 1 to avoid data leakage (use only past values)
            grouped = df.groupby("Store")["Sales"]
            df[f"Sales_rolling_mean_{window}"] = grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f"Sales_rolling_std_{window}"] = grouped.transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
            )
            logger.debug("Rolling features added for window=%d.", window)
        return df

    def _rolling_col_names(self) -> List[str]:
        cols = []
        for w in self.rolling_windows:
            cols += [f"Sales_rolling_mean_{w}", f"Sales_rolling_std_{w}"]
        return cols
