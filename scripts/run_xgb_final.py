"""Evaluate the final tuned XGBoost configuration per target through evaluate_folds, with the full bootstrap CI.

Usage:
    venv/bin/python3 scripts/run_xgb_final.py
"""

import sys  # module search path
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the snapshots

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import evaluate_folds, make_folds, split_holdout, year_distribution  # folds and per-fold report
from models import FINAL_XGB_PARAMS, make_xgboost  # locked configurations and the reusable pair, both in src/models.py

snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached snapshots
development = split_holdout(snapshots)["development"]  # the holdout stays sealed
low_volume_years = year_distribution(snapshots).query("low_volume").index  # years no test tranche may touch
pd.set_option("display.width", 250)
for target, params in FINAL_XGB_PARAMS.items():
    folds = make_folds(development, target=target, low_volume_years=low_volume_years)  # same folds as every earlier step
    report = evaluate_folds(folds, target, *make_xgboost(target, **params))  # 1,000-draw artist-clustered bootstrap
    report.to_csv(ROOT / f"results/trials/xgb_final_{target}.csv")  # kept next to the trial logs
    print(target, params); print(report.round(4).to_string()); print(flush=True)
