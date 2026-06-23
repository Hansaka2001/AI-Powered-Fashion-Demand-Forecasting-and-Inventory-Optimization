"""
main.py
-------
End-to-end pipeline orchestrator for the Fashion Demand
Forecasting & Inventory Optimization system.

Usage
-----
    python main.py                    # Full pipeline
    python main.py --skip-eda         # Skip EDA plots
    python main.py --skip-training    # Load existing model
    python main.py --skip-shap        # Skip SHAP (faster)

Pipeline stages
---------------
1. Data Generation     - Create synthetic sales dataset
2. Preprocessing       - Load, clean, merge, encode
3. Feature Engineering - Lag/rolling/calendar features
4. EDA                 - Exploratory visualisations
5. Model Training      - XGBoost / LightGBM / Random Forest
6. Model Evaluation    - MAE, RMSE, MAPE, R2
7. Forecasting         - 7-day & 30-day demand forecasts
8. Inventory Opt.      - Stock levels, reorder, EOQ
9. SHAP Explainability - Feature importance
"""

import argparse
import io
import logging
import logging.handlers
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Bootstrap – ensure project root on sys.path
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    LOGS_DIR,
    LOG_FILE,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    SALES_CSV,
    STORE_CSV,
    MODELS_DIR,
    FEATURE_COLUMNS,
    TEST_SPLIT_RATIO,
)


# ──────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """Configure console + rotating file logging."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler — force UTF-8 to avoid CP1252 issues on Windows
    ch = logging.StreamHandler(
        stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)
        if hasattr(sys.stdout, 'fileno') else sys.stdout
    )
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    # Rotating file handler (5MB × 3 backups)
    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    return logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# CLI Arguments
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fashion Demand Forecasting & Inventory Optimization Pipeline"
    )
    parser.add_argument(
        "--skip-data-gen", action="store_true",
        help="Skip synthetic data generation (use existing sales.csv)",
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA visualisations",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Skip model training (load existing best_model.joblib)",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip SHAP explainability (faster runs)",
    )
    parser.add_argument(
        "--n-iter", type=int, default=8,
        help="Number of RandomizedSearchCV iterations per model (default: 8)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
# Pipeline Stages
# ──────────────────────────────────────────────────────────────

def stage_data_generation(logger: logging.Logger) -> None:
    """Stage 1: Generate synthetic sales dataset."""
    from data_generator import SalesDataGenerator
    from config import GENERATION_CONFIG

    logger.info("=" * 50)
    logger.info("  STAGE 1 -- Synthetic Data Generation")
    logger.info("=" * 50)

    gen = SalesDataGenerator(GENERATION_CONFIG, STORE_CSV)
    df = gen.generate()
    gen.save(df, SALES_CSV)
    logger.info("[OK] Sales data generated: %d rows", len(df))


def stage_preprocessing(logger: logging.Logger):
    """Stage 2: Load, clean, merge, and encode data."""
    from src.data_preprocessing import DataPreprocessor

    logger.info("=" * 50)
    logger.info("  STAGE 2 -- Data Preprocessing")
    logger.info("=" * 50)

    preprocessor = DataPreprocessor()
    df_clean = preprocessor.run()
    train_df, test_df = preprocessor.get_train_test_split(df_clean, TEST_SPLIT_RATIO)
    logger.info("[OK] Preprocessing complete. Train: %d | Test: %d", len(train_df), len(test_df))
    return df_clean, train_df, test_df


def stage_feature_engineering(
    train_df, test_df, df_clean, logger: logging.Logger
):
    """Stage 3: Generate all ML features."""
    from src.feature_engineering import FeatureEngineer

    logger.info("=" * 50)
    logger.info("  STAGE 3 -- Feature Engineering")
    logger.info("=" * 50)

    fe = FeatureEngineer()

    # Full dataset features (for forecasting history)
    df_feat = fe.fit_transform(df_clean)

    # Train/test feature matrices
    train_feat = fe.fit_transform(train_df)
    test_feat  = fe.fit_transform(test_df)

    X_train, y_train = fe.get_feature_matrix(train_feat)
    X_test,  y_test  = fe.get_feature_matrix(test_feat)

    logger.info("[OK] Features: X_train=%s | X_test=%s", X_train.shape, X_test.shape)
    return X_train, y_train, X_test, y_test, df_feat, fe


def stage_eda(df_feat, logger: logging.Logger) -> None:
    """Stage 4: Exploratory Data Analysis."""
    from src.eda import EDAAnalyzer

    logger.info("=" * 50)
    logger.info("  STAGE 4 -- Exploratory Data Analysis")
    logger.info("=" * 50)

    eda = EDAAnalyzer(df_feat)
    eda.run_all()
    logger.info("[OK] EDA plots saved.")


def stage_model_training(X_train, y_train, n_iter, logger):
    """Stage 5: Train and tune all three models."""
    from src.model_training import ModelTrainer

    logger.info("=" * 50)
    logger.info("  STAGE 5 -- Model Training")
    logger.info("=" * 50)

    trainer = ModelTrainer(n_iter=n_iter)
    results = trainer.train_all(X_train, y_train)
    logger.info("[OK] Training complete. Best model: %s", trainer.best_model_name)
    return trainer, results


def stage_evaluation(trainer, results, X_test, y_test, logger):
    """Stage 6: Evaluate all models on the test set."""
    from src.model_evaluation import ModelEvaluator

    logger.info("=" * 50)
    logger.info("  STAGE 6 -- Model Evaluation")
    logger.info("=" * 50)

    evaluator = ModelEvaluator()
    metrics_df = evaluator.evaluate_all(results, X_test, y_test)
    logger.info("[OK] Evaluation complete.")
    logger.info("\n%s", metrics_df.to_string(index=False))
    return metrics_df


def stage_forecasting(trainer, df_feat, fe, logger):
    """Stage 7: Generate 7-day and 30-day demand forecasts."""
    from src.forecasting import DemandForecaster

    logger.info("=" * 50)
    logger.info("  STAGE 7 -- Demand Forecasting")
    logger.info("=" * 50)

    feature_cols = [c for c in FEATURE_COLUMNS if c in df_feat.columns]
    forecaster = DemandForecaster()
    forecaster.load_model(
        model=trainer.best_model,
        feature_cols=feature_cols,
    )
    forecast_map = forecaster.forecast_all_horizons(df_feat)
    logger.info("[OK] Forecasts generated for horizons: %s", list(forecast_map.keys()))
    return forecast_map


def stage_inventory(forecast_map, df_feat, logger):
    """Stage 8: Inventory optimization."""
    from src.inventory_optimization import InventoryOptimizer

    logger.info("=" * 50)
    logger.info("  STAGE 8 -- Inventory Optimization")
    logger.info("=" * 50)

    optimizer = InventoryOptimizer()
    inv_df = optimizer.optimize(forecast_map, df_feat)
    logger.info("[OK] Inventory recommendations: %d rows", len(inv_df))

    # Summary stats
    logger.info(
        "Risk distribution:\n%s",
        inv_df.groupby(["Forecast_Horizon_Days", "Risk_Level"]).size().to_string(),
    )
    return inv_df


def stage_shap(trainer, X_train, logger):
    """Stage 9: SHAP explainability."""
    from src.explainability import SHAPExplainer

    logger.info("=" * 50)
    logger.info("  STAGE 9 -- SHAP Explainability")
    logger.info("=" * 50)

    feature_cols = list(X_train.columns)
    explainer = SHAPExplainer(trainer.best_model, feature_names=feature_cols)
    explainer.run(X_train)
    logger.info("[OK] SHAP analysis complete.")


def print_summary(start_time: float, args, logger: logging.Logger) -> None:
    """Print a final pipeline summary."""
    from config import REPORTS_DIR, VISUALIZATIONS_DIR, MODELS_DIR

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

    logger.info("")
    logger.info("=" * 62)
    logger.info("  PIPELINE COMPLETE [SUCCESS]")
    logger.info("=" * 62)
    logger.info("  Total runtime : %02dm %02ds", mins, secs)
    logger.info("-" * 62)
    logger.info("  Models        -> %s", MODELS_DIR)
    logger.info("  Reports       -> %s", REPORTS_DIR)
    logger.info("  Visualisations-> %s", VISUALIZATIONS_DIR)
    logger.info("=" * 62)

    # List output files
    for report in sorted(REPORTS_DIR.glob("*.csv")):
        logger.info("  [CSV] %s", report.name)
    for viz in sorted(VISUALIZATIONS_DIR.glob("*.png")):
        logger.info("  [PNG] %s", viz.name)


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logger = setup_logging()
    pipeline_start = time.time()

    logger.info("=" * 62)
    logger.info("  Fashion Demand Forecasting & Inventory Optimization")
    logger.info("  Production ML Pipeline  --  Starting ...")
    logger.info("=" * 62)

    try:
        # ── Stage 1: Data Generation ──────────────────────────────
        if not args.skip_data_gen and not SALES_CSV.exists():
            t = time.time()
            stage_data_generation(logger)
            logger.info("[Stage 1 done in %.1fs]", time.time() - t)
        elif SALES_CSV.exists():
            logger.info("[Stage 1 skipped] Using existing sales.csv")
        else:
            logger.info("[Stage 1 skipped] --skip-data-gen flag set")

        # ── Stage 2: Preprocessing ────────────────────────────────
        t = time.time()
        df_clean, train_df, test_df = stage_preprocessing(logger)
        logger.info("[Stage 2 done in %.1fs]", time.time() - t)

        # ── Stage 3: Feature Engineering ─────────────────────────
        t = time.time()
        X_train, y_train, X_test, y_test, df_feat, fe = stage_feature_engineering(
            train_df, test_df, df_clean, logger
        )
        logger.info("[Stage 3 done in %.1fs]", time.time() - t)

        # ── Stage 4: EDA ──────────────────────────────────────────
        if not args.skip_eda:
            t = time.time()
            stage_eda(df_feat, logger)
            logger.info("[Stage 4 done in %.1fs]", time.time() - t)
        else:
            logger.info("[Stage 4 skipped] --skip-eda flag set")

        # ── Stage 5: Model Training ───────────────────────────────
        if not args.skip_training:
            t = time.time()
            trainer, results = stage_model_training(X_train, y_train, args.n_iter, logger)
            logger.info("[Stage 5 done in %.1fs]", time.time() - t)
        else:
            # Load existing model
            logger.info("[Stage 5 skipped] Loading saved model …")
            from src.model_training import ModelTrainer
            trainer = ModelTrainer()
            trainer.best_model = trainer.load_best_model()
            trainer.best_model_name = "Loaded"
            results = {"Loaded": {"model": trainer.best_model, "cv_rmse": 0,
                                  "best_params": {}, "training_time": 0,
                                  "feature_names": list(X_train.columns)}}

        # ── Stage 6: Evaluation ───────────────────────────────────
        t = time.time()
        metrics_df = stage_evaluation(trainer, results, X_test, y_test, logger)
        logger.info("[Stage 6 done in %.1fs]", time.time() - t)

        # ── Stage 7: Forecasting ──────────────────────────────────
        t = time.time()
        forecast_map = stage_forecasting(trainer, df_feat, fe, logger)
        logger.info("[Stage 7 done in %.1fs]", time.time() - t)

        # ── Stage 8: Inventory Optimization ──────────────────────
        t = time.time()
        inv_df = stage_inventory(forecast_map, df_feat, logger)
        logger.info("[Stage 8 done in %.1fs]", time.time() - t)

        # ── Stage 9: SHAP ─────────────────────────────────────────
        if not args.skip_shap:
            t = time.time()
            stage_shap(trainer, X_train, logger)
            logger.info("[Stage 9 done in %.1fs]", time.time() - t)
        else:
            logger.info("[Stage 9 skipped] --skip-shap flag set")

        # ── Summary ───────────────────────────────────────────────
        print_summary(pipeline_start, args, logger)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Pipeline failed with error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
