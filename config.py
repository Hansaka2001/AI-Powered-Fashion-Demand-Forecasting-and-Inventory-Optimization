"""
config.py
---------
Global configuration for the Fashion Demand Forecasting &
Inventory Optimization pipeline.

All paths, hyperparameters, and domain constants live here.
"""

from pathlib import Path
from typing import Dict, Any, List

# ──────────────────────────────────────────────────────────────
# ROOT PATHS
# ──────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).parent
DATASET_DIR: Path = ROOT_DIR / "dataset"
SRC_DIR: Path = ROOT_DIR / "src"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
VISUALIZATIONS_DIR: Path = OUTPUTS_DIR / "visualizations"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"

# ──────────────────────────────────────────────────────────────
# DATASET PATHS
# ──────────────────────────────────────────────────────────────
STORE_CSV: Path = DATASET_DIR / "store.csv"
SALES_CSV: Path = DATASET_DIR / "sales.csv"

# ──────────────────────────────────────────────────────────────
# DATA GENERATION SETTINGS
# ──────────────────────────────────────────────────────────────
GENERATION_CONFIG: Dict[str, Any] = {
    "num_stores": 50,           # Number of stores to simulate (from 1..1115)
    "start_date": "2013-01-01",
    "end_date": "2015-07-31",
    "random_seed": 42,
}

# ──────────────────────────────────────────────────────────────
# PREPROCESSING SETTINGS
# ──────────────────────────────────────────────────────────────
PREPROCESSING_CONFIG: Dict[str, Any] = {
    "min_sales_threshold": 0,       # Drop rows with Sales < threshold if Open=1
    "competition_distance_fill": "median",  # How to fill missing CompetitionDistance
    "target_column": "Sales",
    "date_column": "Date",
    "store_column": "Store",
}

# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING SETTINGS
# ──────────────────────────────────────────────────────────────
LAG_PERIODS: List[int] = [1, 7, 14, 30]
ROLLING_WINDOWS: List[int] = [7, 14, 30]

FEATURE_COLUMNS: List[str] = [
    # Date features
    "DayOfWeek", "Day", "Week", "Month", "Quarter", "Year",
    "DayOfYear", "IsWeekend", "IsMonthStart", "IsMonthEnd",
    # Seasonal
    "IsSpring", "IsSummer", "IsAutumn", "IsWinter",
    "IsHolidaySeason",
    # Store features
    "StoreType_enc", "Assortment_enc",
    "CompetitionDistance", "CompetitionOpenMonths",
    "HasCompetition", "CompetitionDistanceLog",
    "Promo2",
    # Promotion & Holiday
    "Promo", "StateHoliday_enc", "SchoolHoliday",
    "PromoActive",
    # Lag features
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_30",
    # Rolling features
    "Sales_rolling_mean_7", "Sales_rolling_mean_14", "Sales_rolling_mean_30",
    "Sales_rolling_std_7", "Sales_rolling_std_14", "Sales_rolling_std_30",
]

TARGET_COLUMN: str = "Sales"

# ──────────────────────────────────────────────────────────────
# MODEL TRAINING SETTINGS
# ──────────────────────────────────────────────────────────────
TIMESERIES_SPLIT_FOLDS: int = 5
RANDOM_STATE: int = 42
TEST_SPLIT_RATIO: float = 0.2   # Fraction of data held out for final evaluation

# XGBoost hyperparameter grid
XGBOOST_PARAM_GRID: Dict[str, Any] = {
    "n_estimators": [300, 500],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
}

# LightGBM hyperparameter grid
LGBM_PARAM_GRID: Dict[str, Any] = {
    "n_estimators": [300, 500],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "num_leaves": [31, 63],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

# Random Forest hyperparameter grid
RF_PARAM_GRID: Dict[str, Any] = {
    "n_estimators": [200, 400],
    "max_depth": [8, 12, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.5],
}

# ──────────────────────────────────────────────────────────────
# FORECASTING SETTINGS
# ──────────────────────────────────────────────────────────────
FORECAST_HORIZONS: List[int] = [7, 30]   # days ahead
FORECAST_STORE_SAMPLE: int = 10          # Number of stores to generate forecasts for

# ──────────────────────────────────────────────────────────────
# INVENTORY OPTIMIZATION SETTINGS
# ──────────────────────────────────────────────────────────────
INVENTORY_CONFIG: Dict[str, Any] = {
    "lead_time_days": 7,              # Supplier lead time
    "service_level": 0.95,           # 95% service level → Z ≈ 1.645
    "z_score": 1.645,                # Z-score for 95% service level
    "holding_cost_rate": 0.25,       # Annual holding cost as fraction of item value
    "ordering_cost": 50.0,           # Fixed cost per order (€)
    "unit_cost": 15.0,               # Average unit cost (€)
    "stockout_cost_multiplier": 3.0, # Stockout cost = multiplier × unit_cost
}

# ──────────────────────────────────────────────────────────────
# SHAP SETTINGS
# ──────────────────────────────────────────────────────────────
SHAP_CONFIG: Dict[str, Any] = {
    "sample_size": 5000,     # Rows to sample for SHAP computation
    "max_display": 20,       # Top N features to display
}

# ──────────────────────────────────────────────────────────────
# LOGGING SETTINGS
# ──────────────────────────────────────────────────────────────
LOG_FILE: Path = LOGS_DIR / "pipeline.log"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
