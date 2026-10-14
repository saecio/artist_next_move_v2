"""Recompute the intervals of the final tuned XGBoost configuration at the project confidence level.

Same folds, locked parameters and seed as scripts/run_xgb_final.py, which is not run again. The point estimates
must equal the stored results/trials/xgb_final_{target}.csv exactly. If they do not, nothing is written.
The output goes to new files and no existing file is opened for writing.

Usage:
    venv/bin/python3 scripts/run_xgb_final_ci95.py
"""

import sys  # module search path and exit
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the snapshots and the stored results

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import evaluate_folds, make_folds, split_holdout, year_distribution  # folds and per-fold report
from models import FINAL_XGB_PARAMS, make_xgboost  # locked configurations and the reusable pair, both in src/models.py

POINT_COLUMNS = ["test_years", "train_rows", "test_rows", "positive_rate", "pr_auc", "roc_auc"]  # everything except the interval bounds
new_paths = {target: ROOT / f"results/trials/xgb_final_{target}_ci95.csv" for target in FINAL_XGB_PARAMS}  # one new file per target
if any(path.exists() for path in new_paths.values()):  # never overwrite
    sys.exit("a ci95 result file already exists, nothing is overwritten")

snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached snapshots
development = split_holdout(snapshots)["development"]  # the holdout stays sealed
low_volume_years = year_distribution(snapshots).query("low_volume").index  # years no test tranche may touch
pd.set_option("display.width", 250)  # readable printed tables
reports = {}  # new report per target, written only after both pass the check
for target, params in FINAL_XGB_PARAMS.items():
    folds = make_folds(development, target=target, low_volume_years=low_volume_years)  # same folds as every earlier step
    reports[target] = evaluate_folds(folds, target, *make_xgboost(target, **params))  # 1,000-draw artist-clustered bootstrap at the default confidence
    stored = pd.read_csv(ROOT / f"results/trials/xgb_final_{target}.csv", index_col="fold", float_precision="round_trip")  # stored report, read only
    same_point = reports[target][POINT_COLUMNS].equals(stored[POINT_COLUMNS])  # exact equality of every point estimate
    wider = bool(((reports[target]["pr_auc_lo"] <= stored["pr_auc_lo"]) & (reports[target]["pr_auc_hi"] >= stored["pr_auc_hi"])).all())  # the new interval contains the stored one
    print(target, FINAL_XGB_PARAMS[target], "| point estimates identical:", same_point, "| new interval contains stored:", wider, flush=True)
    print(reports[target].round(4).to_string(), flush=True)
    if not (same_point and wider):  # stop before writing anything
        sys.exit(f"{target}: the check failed, nothing was written")
for target, report in reports.items():
    report.to_csv(new_paths[target])  # new file, next to the stored one
    print("written", new_paths[target].name, flush=True)
