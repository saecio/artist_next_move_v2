"""Read-only calibration check of the final WHAT and WHEN models on the final holdout. Nothing is refitted or recalibrated.

Refuses to run if any output exists. Usage:
    venv/bin/python3 scripts/run_calibration_check.py
"""

import json  # model columns file
import sys  # module search path and exit
from pathlib import Path  # project root

import numpy as np  # arrays
import pandas as pd  # tables
import yaml  # demo settings
from sklearn.metrics import average_precision_score  # PR-AUC control
from xgboost import XGBClassifier  # saved final models

ROOT = Path(__file__).resolve().parent.parent  # project root
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import holdout_gap, split_holdout  # same holdout split as the final evaluation
from calibration import CAL, reliability_table, summarize  # calibration functions and settings
from models import build_features  # same feature builder as the fitted pair

out_dir = ROOT / CAL["out_dir"]  # output folder
if out_dir.exists() and any(out_dir.iterdir()):  # outputs are written once
    sys.exit("results/calibration already has results, refusing to overwrite")
out_dir.mkdir(parents=True, exist_ok=True)  # create the folder

def check(name: str, value: float, expected: float, tol: float) -> None:
    """Stop the run when a control differs from the published figure."""
    print(f"control {name}: {value:.4f} vs {expected:.4f}")  # show the control
    if abs(value - expected) > tol:  # failed control
        sys.exit(f"control failed: {name}")  # stop, no forcing

results = {}  # target -> (scores, labels, years)
what = pd.read_csv(ROOT / "results/holdout_what/row_scores.csv")  # WHAT holdout scores
check("what rows", len(what), CAL["controls"]["what_rows"], 0)  # row count
check("what positive rate", what["label"].mean(), CAL["controls"]["what_positive_rate"], 5e-5)  # positive rate
check("what pr_auc", average_precision_score(what["label"], what["stage0_score"]), CAL["published_pr_auc"]["what"], CAL["tolerance"])  # PR-AUC
results["what"] = (what["stage0_score"].to_numpy(), what["label"].to_numpy(), what["year"].to_numpy())  # WHAT inputs

live = yaml.safe_load((ROOT / "app/demo_config.yaml").read_text())["live"]  # model file names
folder = ROOT / live["folder"]  # folder of the saved models
columns = json.loads((folder / live["columns"]).read_text())  # ordered model columns
holdout = split_holdout(pd.read_parquet(ROOT / "data/processed/snapshots.parquet"))["holdout"]  # sealed holdout snapshots
for target in ("next_1y", "next_2y"):
    model = XGBClassifier()  # empty model
    model.load_model(folder / live["when_models"][target])  # saved final model
    test = holdout[holdout[target].notna()]  # resolved targets only
    scores = model.predict_proba(build_features(test)[columns[f"when_{target}"]])[:, 1]  # saved model scores
    labels = test[target].to_numpy()  # resolved labels
    check(f"{target} rows", len(test), CAL["controls"]["when_rows"], 0)  # row count
    check(f"{target} artists", test["artist_id"].nunique(), CAL["controls"]["when_artists"], 0)  # artist count
    check(f"{target} pr_auc", average_precision_score(labels, scores), CAL["published_pr_auc"][target], CAL["tolerance"])  # PR-AUC
    results[target] = (scores, labels, test["year"].to_numpy())  # WHEN inputs

summary = []  # one row per target
for target, (scores, labels, years) in results.items():
    reliability_table(scores, labels).to_csv(out_dir / f"reliability_{target}.csv", index=False, mode="x")  # reliability table, exclusive
    summary.append(summarize(target, scores, labels, years))  # summary row
summary = pd.DataFrame(summary)  # summary table
summary.to_csv(out_dir / "summary.csv", index=False, mode="x")  # summary, exclusive
pd.set_option("display.width", 250)  # wide print
print(summary.drop(columns="ece_by_year").round(4).to_string(index=False))  # summary
print(summary[["target", "ece_by_year"]].to_string(index=False))  # ECE by year
