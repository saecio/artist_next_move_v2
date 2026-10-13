"""Run the XGBoost grid search for one or both targets and append every combination to results/trials/.

Usage:
    venv/bin/python3 scripts/run_xgb_grid_search.py next_1y next_2y
"""

import sys  # command line arguments and module search path
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the snapshots

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import make_folds, split_holdout, year_distribution  # fold generation, same as the random search
from tuning import prepare_fold_data, run_grid_search  # tuning helpers

snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached snapshots
development = split_holdout(snapshots)["development"]  # the holdout stays sealed
low_volume_years = year_distribution(snapshots).query("low_volume").index  # years no test tranche may touch
for target in sys.argv[1:]:  # one session per target, tuned separately
    folds = make_folds(development, target=target, low_volume_years=low_volume_years)  # same folds as the random search
    fold_data = prepare_fold_data(folds, target)  # features and DMatrix built once for all combos
    run_grid_search(fold_data, target, ROOT / f"results/trials/xgb_{target}.jsonl")
