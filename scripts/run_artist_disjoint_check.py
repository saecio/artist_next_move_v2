"""Descriptive check: score the final models on test artists never seen in training, on the development folds only.

For every fold, the train rows of the artists that appear in that fold's test set are removed and the test set is kept
exactly as it was. The final models and their baselines are refitted and re-scored with the same bootstrap settings as
the original folds. Nothing recorded elsewhere changes. The holdout is never read.

Usage:
    venv/bin/python3 scripts/run_artist_disjoint_check.py
"""

import sys  # module search path and exit
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the tables and build the report

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import HOLDOUT_START_YEAR, evaluate_folds, make_folds, make_what_folds, split_holdout, year_distribution  # development folds and per-fold report
from baselines import KILL_GATE_CONFIDENCE, N_BOOTSTRAP, SEED, add_what_buckets, fit_what_lookup_baseline, fit_when_baseline, score_what_lookup_baseline, score_when_baseline  # baselines and bootstrap settings
from models import FINAL_XGB_PARAMS, make_what_xgboost, make_xgboost  # locked WHEN configurations and the WHAT stage 0 pair
from what_target import WHAT_BASELINE_C_PR_AUC, build_artist_years, build_what_features, build_what_table  # WHAT table, inputs and logged baseline (c) values

OUT_DIR = ROOT / "results/artist_disjoint"  # new folder, never reused
if OUT_DIR.exists():  # refuse to run twice
    sys.exit("results/artist_disjoint already exists, nothing was run")

def disjoint(folds: list) -> list:  # same folds without the train rows of the artists in the test set
    out = []  # new fold dicts
    for f in folds:  # one pass per fold
        train, test = f["train"], f["test"]  # original dataframes
        assert train["year"].max() < HOLDOUT_START_YEAR and test["year"].max() < HOLDOUT_START_YEAR, "a holdout row is in the fold"  # development years only
        out.append({**f, "train": train[~train["artist_id"].isin(set(test["artist_id"]))]})  # test object is kept as it is
    return out

def control(folds: list, stored_rows: pd.Series, label: str) -> None:  # test rows must match the stored report row for row
    got = [len(f["test"]) for f in folds]  # rows scored in each fold
    assert got == stored_rows.tolist(), f"{label}: test rows {got} differ from the stored {stored_rows.tolist()}"  # same folds as the original

def rows_for(target: str, folds: list, new_folds: list, models: dict, stored: pd.DataFrame, stored_col: str, original_baseline) -> list:  # one dict per fold
    reports = {}  # report of each model
    for name, (fit_fn, score_fn) in models.items():  # model, then baseline
        print(f"{target} {name}", flush=True)
        reports[name] = evaluate_folds(new_folds, target, fit_fn, score_fn, n_bootstrap=N_BOOTSTRAP, confidence=KILL_GATE_CONFIDENCE, seed=SEED)  # same bootstrap as the originals
    model, base = reports["model"], reports["baseline"]  # short names
    rows = []  # report rows
    for i, (f, g) in enumerate(zip(folds, new_folds)):  # fold by fold
        k = i + 1  # 1-based fold number
        orig = stored[stored_col].iloc[i]  # original model PR-AUC
        orig_base = original_baseline[i] if original_baseline is not None else float("nan")  # original baseline PR-AUC, only where it is recorded
        rows.append({
            "target": target, "fold": k, "test_rows": len(g["test"]), "test_artists": g["test"]["artist_id"].nunique(),
            "train_rows_original": len(f["train"]), "train_rows_disjoint": len(g["train"]),
            "share_of_train_rows_removed": 1 - len(g["train"]) / len(f["train"]),
            "model_pr_auc": model.loc[k, "pr_auc"], "model_lo": model.loc[k, "pr_auc_lo"], "model_hi": model.loc[k, "pr_auc_hi"],
            "baseline_pr_auc": base.loc[k, "pr_auc"], "baseline_lo": base.loc[k, "pr_auc_lo"], "baseline_hi": base.loc[k, "pr_auc_hi"],
            "difference_points": (model.loc[k, "pr_auc"] - base.loc[k, "pr_auc"]) * 100,  # model minus baseline
            "original_model_pr_auc": orig, "original_difference_points": (orig - orig_base) * 100,  # empty for WHEN, its baseline is not stored per fold
        })
        print(rows[-1], flush=True)
    return rows

# WHEN: development table and its folds, built as run_when_xgb_final_ci95.py builds them.
snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached snapshots
development = split_holdout(snapshots)["development"]  # the holdout stays sealed
low_volume_years = year_distribution(snapshots).query("low_volume").index  # years no test tranche may touch
when = {}  # target -> (original folds, disjoint folds, stored report)
for target in FINAL_XGB_PARAMS:
    folds = make_folds(development, target=target, low_volume_years=low_volume_years)  # same folds as every earlier step
    stored = pd.read_csv(ROOT / f"results/trials/xgb_final_{target}_ci95.csv", index_col="fold")  # stored report, read only
    control(folds, stored["test_rows"], target)  # row for row against the original
    when[target] = (folds, disjoint(folds), stored)

# WHAT: table, inputs and folds, built as run_what_gate builds them.
works = pd.read_parquet(ROOT / "data/processed/works_checkpoint.parquet")  # cleaned Works table
table = build_what_table(build_artist_years(works))  # labelled artist-year rows
features = build_what_features(works)  # model inputs
what_folds = make_what_folds(table)  # five development folds
for f in what_folds:  # bucket columns baseline (c) needs
    f["train"], f["test"] = add_what_buckets(f["train"]), add_what_buckets(f["test"])
what_stored = pd.read_csv(ROOT / "results/trials/xgb_what_compare.csv", index_col="fold")  # stored report, read only
control(what_folds, what_stored["test_rows"], "new_style_year")  # row for row against the original
print("control passed: every fold has the stored test rows", flush=True)

OUT_DIR.mkdir(parents=True)  # created only after the control, fails if it exists
report = []  # all rows
what_models = {"model": make_what_xgboost(features), "baseline": (lambda train: fit_what_lookup_baseline(train, ["history_bucket", "volume_bucket"]), score_what_lookup_baseline)}  # stage 0 and baseline (c)
report += rows_for("new_style_year", what_folds, disjoint(what_folds), what_models, what_stored, "stage0_pr_auc", WHAT_BASELINE_C_PR_AUC)
for target, (folds, new_folds, stored) in when.items():
    models = {"model": make_xgboost(target, **FINAL_XGB_PARAMS[target]), "baseline": (fit_when_baseline, score_when_baseline)}  # locked model and its reference
    report += rows_for(target, folds, new_folds, models, stored, "pr_auc", None)
result = pd.DataFrame(report)  # one row per target and fold
with open(OUT_DIR / "artist_disjoint.csv", "x") as handle:  # exclusive creation
    result.to_csv(handle, index=False)
pd.set_option("display.width", 250)  # readable printed table
print(result[["target", "fold", "share_of_train_rows_removed", "model_pr_auc", "baseline_pr_auc", "difference_points", "original_model_pr_auc", "original_difference_points"]].round(4).to_string(index=False), flush=True)
