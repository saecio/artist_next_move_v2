"""Recompute the sealed-holdout intervals of the WHEN final model and baseline at the project confidence level.

Same final training rows, locked parameters, seed and bootstrap as scripts/run_holdout.py, which is not run again.
Each model is fitted once and the holdout is scored once, solely to obtain the new bounds. Every point estimate
must equal the stored results/holdout/*.json exactly. If anything differs, nothing is written. The output goes to
results/holdout_ci95/ and no existing file is opened for writing.

Usage:
    venv/bin/python3 scripts/run_holdout_ci95.py
"""

import json  # result files
import sys  # module search path and exit
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the snapshots

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import evaluate_holdout, holdout_gap, split_holdout  # holdout split, label gap and final evaluation
from baselines import fit_when_baseline, score_when_baseline  # historical WHEN baseline, the reference floor
from models import FINAL_XGB_PARAMS, make_xgboost  # locked configurations and the reusable pair

BOUNDS = ["pr_auc_lo", "pr_auc_hi", "roc_auc_lo", "roc_auc_hi"]  # the only keys expected to change with the confidence level
stored_dir = ROOT / "results/holdout"  # stored results, read only
out_dir = ROOT / "results/holdout_ci95"  # new results
if out_dir.exists() and any(out_dir.iterdir()):  # never overwrite
    sys.exit("results/holdout_ci95 already has results, nothing is overwritten")

snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached snapshots
parts = split_holdout(snapshots)  # development, holdout, excluded_recent, demo
final_train, _ = holdout_gap(parts["development"])  # development without the rows whose label reaches into the holdout
results = {}  # all four results, written only after every check passes
for target in ("next_1y", "next_2y"):
    models = {"xgboost_final": make_xgboost(target, **FINAL_XGB_PARAMS[target]), "baseline": (fit_when_baseline, score_when_baseline)}  # locked model and its reference
    for name, (fit_fn, score_fn) in models.items():
        result = json.loads(json.dumps(evaluate_holdout(final_train, parts["holdout"], target, fit_fn, score_fn)))  # fit once, score the holdout once, plain JSON types
        stored = json.loads((stored_dir / f"{name}_{target}.json").read_text())  # stored result, read only
        differs = [key for key in stored if key not in BOUNDS and stored[key] != result[key]]  # keys whose value is not identical
        wider = result["pr_auc_lo"] <= stored["pr_auc_lo"] and result["pr_auc_hi"] >= stored["pr_auc_hi"] and result["roc_auc_lo"] <= stored["roc_auc_lo"] and result["roc_auc_hi"] >= stored["roc_auc_hi"]  # the new intervals contain the stored ones
        print(name, target, "| point estimates identical:", not differs, "| new intervals contain stored:", wider, flush=True)
        print("  PR-AUC", round(result["pr_auc"], 4), [round(result["pr_auc_lo"], 4), round(result["pr_auc_hi"], 4)], "ROC-AUC", round(result["roc_auc"], 4), [round(result["roc_auc_lo"], 4), round(result["roc_auc_hi"], 4)], flush=True)
        if differs or not wider:  # stop before writing anything
            sys.exit(f"{name} {target}: the check failed ({differs}), nothing was written")
        results[f"{name}_{target}"] = result
out_dir.mkdir(parents=True, exist_ok=True)  # created only after every check passed
for name, result in results.items():
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1))  # same layout as the stored files
    print("written", f"{name}.json", flush=True)
