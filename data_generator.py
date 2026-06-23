"""
data_generator.py
-----------------
Generates a realistic synthetic sales dataset that mirrors the
Rossmann Store Sales structure, enriched with fashion/retail
seasonality patterns.

Outputs:  dataset/sales.csv

Columns produced
----------------
Store            : int     - Store ID (1..num_stores)
Date             : str     - ISO date string
DayOfWeek        : int     - 1=Mon … 7=Sun
Sales            : float   - Daily revenue (€)
Customers        : int     - Foot traffic
Open             : int     - 1 if store open, 0 closed
Promo            : int     - 1 if regular promo running
StateHoliday     : str     - '0', 'a'(public), 'b'(Easter), 'c'(Christmas)
SchoolHoliday    : int     - 1 if school holiday
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DATASET_DIR, GENERATION_CONFIG, LOG_FORMAT, LOG_DATE_FORMAT

# ──────────────────────────────────────────────────────────────
# Logger setup
# ──────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class SalesDataGenerator:
    """
    Generates a realistic synthetic daily-sales dataset for
    multiple retail stores spanning multiple years.

    The generated data mimics the Rossmann Store Sales
    Kaggle dataset with added fashion-retail seasonality
    (summer slumps, back-to-school peaks, Christmas surges).

    Parameters
    ----------
    config : dict
        Configuration dictionary from ``config.GENERATION_CONFIG``.
    store_csv_path : Path
        Path to the existing ``store.csv`` metadata file.
    """

    # German public holiday months (simplified)
    _PUBLIC_HOLIDAY_DATES = {
        (1, 1), (5, 1), (10, 3), (12, 25), (12, 26),
    }
    # Easter approximate dates 2013-2015 (month, day)
    _EASTER_DATES = {(3, 31), (4, 1), (4, 20), (4, 21), (4, 5), (4, 6)}
    # Christmas window
    _CHRISTMAS_WINDOW = list(range(12, 26))  # Dec 12–25

    def __init__(self, config: dict, store_csv_path: Path) -> None:
        self.config = config
        self.store_csv_path = store_csv_path
        self._rng = np.random.default_rng(config["random_seed"])
        random.seed(config["random_seed"])

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate the full synthetic sales dataset.

        Returns
        -------
        pd.DataFrame
            Raw sales DataFrame ready for preprocessing.
        """
        logger.info("Loading store metadata …")
        store_meta = self._load_store_meta()

        logger.info(
            "Generating sales data for %d stores (%s → %s) …",
            self.config["num_stores"],
            self.config["start_date"],
            self.config["end_date"],
        )

        date_range = pd.date_range(
            self.config["start_date"], self.config["end_date"], freq="D"
        )

        records = []
        for store_id in tqdm(store_meta["Store"].values, desc="Stores"):
            store_info = store_meta[store_meta["Store"] == store_id].iloc[0]
            store_records = self._generate_store_series(store_id, store_info, date_range)
            records.extend(store_records)

        df = pd.DataFrame(records)
        logger.info("Generated %d total rows.", len(df))
        return df

    def save(self, df: pd.DataFrame, output_path: Path) -> None:
        """Persist the generated DataFrame to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved synthetic sales data → %s", output_path)

    # ----------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------

    def _load_store_meta(self) -> pd.DataFrame:
        """Load and sample stores from store.csv."""
        meta = pd.read_csv(self.store_csv_path)
        n = min(self.config["num_stores"], len(meta))
        return meta.head(n)[["Store", "StoreType", "Assortment", "Promo2"]].copy()

    def _generate_store_series(
        self, store_id: int, store_info: pd.Series, date_range: pd.DatetimeIndex
    ) -> list:
        """Generate a daily time-series for a single store."""
        # Base demand level varies by store type
        base_demand_map = {"a": 6500, "b": 4000, "c": 8500, "d": 9500}
        base_sales = base_demand_map.get(
            str(store_info.get("StoreType", "a")).lower(), 6000
        )
        noise_scale = base_sales * 0.12

        records = []
        for date in date_range:
            dow = date.dayofweek + 1  # 1=Mon … 7=Sun
            # Stores closed on Sundays (German retail norm)
            is_open = 0 if dow == 7 else 1
            # Random ~3% closed days (maintenance)
            if is_open and self._rng.random() < 0.03:
                is_open = 0

            state_holiday = self._state_holiday(date)
            school_holiday = self._school_holiday(date)

            # Promo schedule (approx every other week)
            promo = int((date.isocalendar()[1] % 2) == 0)
            promo2 = int(store_info.get("Promo2", 0))

            if not is_open:
                sales, customers = 0, 0
            else:
                sales, customers = self._compute_sales(
                    base_sales, noise_scale, date, dow,
                    promo, promo2, state_holiday, school_holiday,
                )

            records.append({
                "Store": store_id,
                "Date": date.strftime("%Y-%m-%d"),
                "DayOfWeek": dow,
                "Sales": round(sales, 2),
                "Customers": customers,
                "Open": is_open,
                "Promo": promo,
                "StateHoliday": state_holiday,
                "SchoolHoliday": school_holiday,
            })
        return records

    def _compute_sales(
        self,
        base: float,
        noise_scale: float,
        date: pd.Timestamp,
        dow: int,
        promo: int,
        promo2: int,
        state_holiday: str,
        school_holiday: int,
    ) -> tuple[float, int]:
        """Apply all multiplicative factors to base sales."""
        # 1. Day-of-week factor (Mon–Sat)
        dow_factors = {1: 1.05, 2: 0.95, 3: 0.90, 4: 0.95, 5: 1.10, 6: 1.25}
        s = base * dow_factors.get(dow, 1.0)

        # 2. Monthly seasonality (fashion retail peaks)
        month_factors = {
            1: 0.80,  # Post-holiday dip
            2: 0.82,
            3: 1.00,  # Spring collection launch
            4: 1.05,
            5: 1.08,
            6: 0.95,  # Summer: slight dip before sales
            7: 0.88,  # Summer sale period
            8: 1.10,  # Back-to-school
            9: 1.12,
            10: 1.05,
            11: 1.20,  # Black Friday / pre-holiday
            12: 1.35,  # Christmas peak
        }
        s *= month_factors.get(date.month, 1.0)

        # 3. Yearly trend (slight growth)
        s *= 1.0 + (date.year - 2013) * 0.03

        # 4. Holiday effects
        if state_holiday in ("b", "c"):
            s *= 1.30  # Public/Christmas holidays boost
        if school_holiday:
            s *= 1.08

        # 5. Promotion boost
        if promo:
            s *= 1.18
        if promo2:
            s *= 1.05

        # 6. Gaussian noise
        noise = self._rng.normal(0, noise_scale)
        s = max(100, s + noise)

        # 7. Customers proportional to sales with some variance
        customers = int(max(10, s / self._rng.normal(8.5, 0.8)))

        return s, customers

    def _state_holiday(self, date: pd.Timestamp) -> str:
        """Return state holiday code for a given date."""
        if (date.month, date.day) in self._EASTER_DATES:
            return "b"
        if date.month == 12 and date.day in range(20, 27):
            return "c"
        if (date.month, date.day) in self._PUBLIC_HOLIDAY_DATES:
            return "a"
        return "0"

    def _school_holiday(self, date: pd.Timestamp) -> int:
        """Heuristic school holiday flag for German states."""
        m, d = date.month, date.day
        # Summer holidays (July–Aug), Christmas (Dec 23 – Jan 5)
        if m in (7, 8):
            return 1
        if m == 12 and d >= 23:
            return 1
        if m == 1 and d <= 5:
            return 1
        if m == 4 and d in range(10, 25):  # Easter school break
            return 1
        return 0


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    """Generate and save the synthetic sales dataset."""
    from config import SALES_CSV, STORE_CSV

    gen = SalesDataGenerator(GENERATION_CONFIG, STORE_CSV)
    df = gen.generate()
    gen.save(df, SALES_CSV)

    # Quick summary
    print("\n" + "=" * 60)
    print("  Synthetic Sales Dataset Summary")
    print("=" * 60)
    print(f"  Rows          : {len(df):,}")
    print(f"  Stores        : {df['Store'].nunique()}")
    print(f"  Date range    : {df['Date'].min()} → {df['Date'].max()}")
    print(f"  Avg daily sales: €{df[df['Open']==1]['Sales'].mean():,.0f}")
    print(f"  Saved to      : {SALES_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
