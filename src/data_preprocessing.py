"""
src/data_preprocessing.py
--------------------------
Handles all data loading, cleaning, merging, and encoding
steps to produce a clean, analysis-ready DataFrame.

Classes
-------
DataPreprocessor
    Load → merge → clean → encode → return clean DataFrame.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import PREPROCESSING_CONFIG, STORE_CSV, SALES_CSV

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Loads, merges, cleans, and encodes the raw sales and store
    metadata CSVs into a single analysis-ready DataFrame.

    Parameters
    ----------
    sales_path : Path, optional
        Path to the sales CSV.  Defaults to ``config.SALES_CSV``.
    store_path : Path, optional
        Path to the store metadata CSV.  Defaults to ``config.STORE_CSV``.
    config : dict, optional
        Preprocessing configuration.  Defaults to ``config.PREPROCESSING_CONFIG``.
    """

    # Categorical encoding maps
    _STORE_TYPE_MAP = {"a": 1, "b": 2, "c": 3, "d": 4}
    _ASSORTMENT_MAP = {"a": 1, "b": 2, "c": 3}
    _STATE_HOLIDAY_MAP = {"0": 0, 0: 0, "a": 1, "b": 2, "c": 3}

    def __init__(
        self,
        sales_path: Path = SALES_CSV,
        store_path: Path = STORE_CSV,
        config: dict = None,
    ) -> None:
        self.sales_path = sales_path
        self.store_path = store_path
        self.config = config or PREPROCESSING_CONFIG
        self._df: pd.DataFrame | None = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """
        Execute the full preprocessing pipeline.

        Returns
        -------
        pd.DataFrame
            Clean, encoded, date-processed DataFrame.
        """
        logger.info("Starting preprocessing pipeline …")
        df = self._load_and_merge()
        df = self._parse_dates(df)
        df = self._handle_missing_values(df)
        df = self._filter_closed_stores(df)
        df = self._encode_categoricals(df)
        df = self._compute_competition_features(df)
        df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
        self._df = df
        logger.info(
            "Preprocessing complete. Shape: %s. Date range: %s → %s",
            df.shape,
            df["Date"].min().date(),
            df["Date"].max().date(),
        )
        return df

    def get_train_test_split(
        self, df: pd.DataFrame, test_ratio: float = 0.20
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporally split the dataset into training and test sets.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed DataFrame with a 'Date' column.
        test_ratio : float
            Fraction of the *time range* to reserve for testing.

        Returns
        -------
        (train_df, test_df) : Tuple[pd.DataFrame, pd.DataFrame]
        """
        dates = df["Date"].sort_values()
        cutoff_idx = int(len(dates.unique()) * (1 - test_ratio))
        cutoff_date = sorted(dates.unique())[cutoff_idx]
        train_df = df[df["Date"] < cutoff_date].copy()
        test_df = df[df["Date"] >= cutoff_date].copy()
        logger.info(
            "Train split: %d rows (%s → %s)",
            len(train_df), train_df["Date"].min().date(), train_df["Date"].max().date(),
        )
        logger.info(
            "Test split : %d rows (%s → %s)",
            len(test_df), test_df["Date"].min().date(), test_df["Date"].max().date(),
        )
        return train_df, test_df

    # ----------------------------------------------------------
    # Private steps
    # ----------------------------------------------------------

    def _load_and_merge(self) -> pd.DataFrame:
        """Load sales + store CSVs and merge on Store."""
        logger.info("Loading sales data from: %s", self.sales_path)
        sales = pd.read_csv(self.sales_path, low_memory=False)
        logger.info("Sales rows loaded: %d", len(sales))

        logger.info("Loading store metadata from: %s", self.store_path)
        store = pd.read_csv(self.store_path, low_memory=False)
        logger.info("Store rows loaded: %d", len(store))

        df = sales.merge(store, on="Store", how="left")
        logger.info("Merged DataFrame shape: %s", df.shape)
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the Date column to datetime."""
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute and fill missing values across key columns."""
        # CompetitionDistance: fill with median
        if "CompetitionDistance" in df.columns:
            median_dist = df["CompetitionDistance"].median()
            missing_dist = df["CompetitionDistance"].isna().sum()
            df["CompetitionDistance"].fillna(median_dist, inplace=True)
            logger.info(
                "CompetitionDistance: filled %d NaNs with median=%.0f",
                missing_dist, median_dist,
            )

        # CompetitionOpenSince: fill with sensible defaults
        for col in ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]:
            if col in df.columns:
                df[col].fillna(0, inplace=True)

        # Promo2Since columns
        for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
            if col in df.columns:
                df[col].fillna(0, inplace=True)

        # PromoInterval
        if "PromoInterval" in df.columns:
            df["PromoInterval"].fillna("", inplace=True)

        # StateHoliday
        if "StateHoliday" in df.columns:
            df["StateHoliday"].fillna("0", inplace=True)
            df["StateHoliday"] = df["StateHoliday"].astype(str)

        # SchoolHoliday / Promo
        for col in ["SchoolHoliday", "Promo", "Promo2"]:
            if col in df.columns:
                df[col].fillna(0, inplace=True)

        # Customers
        if "Customers" in df.columns:
            df["Customers"].fillna(0, inplace=True)

        logger.info("Missing value handling complete.")
        return df

    def _filter_closed_stores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows where the store was closed (Open=0, Sales=0)."""
        if "Open" in df.columns:
            before = len(df)
            df = df[df["Open"] == 1].copy()
            logger.info(
                "Filtered closed-store rows: %d → %d (removed %d)",
                before, len(df), before - len(df),
            )
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode StoreType, Assortment, and StateHoliday."""
        if "StoreType" in df.columns:
            df["StoreType_enc"] = (
                df["StoreType"].str.lower().map(self._STORE_TYPE_MAP).fillna(0).astype(int)
            )

        if "Assortment" in df.columns:
            df["Assortment_enc"] = (
                df["Assortment"].str.lower().map(self._ASSORTMENT_MAP).fillna(0).astype(int)
            )

        if "StateHoliday" in df.columns:
            df["StateHoliday_enc"] = (
                df["StateHoliday"].map(self._STATE_HOLIDAY_MAP).fillna(0).astype(int)
            )

        logger.info("Categorical encoding complete.")
        return df

    def _compute_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive competition-related engineered features."""
        # Months since competition opened
        if all(c in df.columns for c in
               ["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]):
            comp_year = df["CompetitionOpenSinceYear"].replace(0, np.nan)
            comp_month = df["CompetitionOpenSinceMonth"].replace(0, np.nan)

            # Competition open date in months
            comp_open_months = (
                (df["Date"].dt.year - comp_year) * 12
                + (df["Date"].dt.month - comp_month)
            ).clip(lower=0)
            df["CompetitionOpenMonths"] = comp_open_months.fillna(-1)
        else:
            df["CompetitionOpenMonths"] = -1

        # Binary: has a nearby competitor
        df["HasCompetition"] = (df["CompetitionDistance"] < 5000).astype(int)

        # Log-transformed distance (reduces skewness)
        df["CompetitionDistanceLog"] = np.log1p(
            df["CompetitionDistance"].clip(lower=0)
        )

        logger.info("Competition features engineered.")
        return df
