"""One-time final evaluation on the sealed holdout (2018-2023): locked XGBoost and the historical baseline.

Refuses to run twice: the result files are written once and never overwritten.

Usage:
    venv/bin/python3 scripts/run_holdout.py
"""

import json  # result files
import sys  # module search path
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the snapshots

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import evaluate_holdout, holdout_gap, split_holdout  # holdout split, label gap and final evaluation
from baselines import fit_when_baseline, score_when_baseline  # historical WHEN baseline, the reference floor
from models import FINAL_XGB_PARAMS, make_xgboost  # locked configurations and the reusable pair

out_dir = ROOT / "results/holdout"  # one JSON file per model and target
if out_dir.exists() and any(out_dir.iterdir()):  # the holdout is a one-time number
    sys.exit("results/holdout already has results, the sealed holdout is evaluated once")
out_dir.mkdir(parents=True, exist_ok=True)

snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached snapshots
parts = split_holdout(snapshots)  # development, holdout, excluded_recent, demo
final_train, _ = holdout_gap(parts["development"])  # development without the rows whose label reaches into the holdout
for target in ("next_1y", "next_2y"):
    models = {"xgboost_final": make_xgboost(target, **FINAL_XGB_PARAMS[target]), "baseline": (fit_when_baseline, score_when_baseline)}  # locked model and its reference
    for name, (fit_fn, score_fn) in models.items():
        result = evaluate_holdout(final_train, parts["holdout"], target, fit_fn, score_fn)  # fit once, score the holdout once
        (out_dir / f"{name}_{target}.json").write_text(json.dumps(result, indent=1))  # written once
        print(name, target, round(result["pr_auc"], 4), [round(result["pr_auc_lo"], 4), round(result["pr_auc_hi"], 4)], round(result["roc_auc"], 4), flush=True)
