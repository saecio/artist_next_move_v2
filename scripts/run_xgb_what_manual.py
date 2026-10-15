"""Manual tuning phase for the WHAT XGBoost (phase 3): five hand-picked trials, appended to results/trials/.

The log is append-only and shared by the three phases. Trials already in it are skipped, so a rerun resumes.
The base configuration is read from the log (best non-stage0 trial of phases 1 and 2), not typed in.

Usage:
    venv/bin/python3 scripts/run_xgb_what_manual.py
"""

import sys  # module search path and exit
from collections import Counter  # rows per phase already in the log
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the Works table

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import make_what_folds  # development folds with the label-resolution rule
from tuning import _append, prepare_what_fold_data, read_log, run_trial  # same trial function and log as the earlier phases
from what_target import build_artist_years, build_what_features, build_what_table  # WHAT table and inputs

TARGET = "new_style_year"  # WHAT target
LOG_PATH = ROOT / "results/trials/xgb_what_new_style_year.jsonl"  # shared by the three phases, append-only
EXPECTED_BASE = {"trial": "g13", "max_depth": 5, "reg_lambda": 0.3, "learning_rate": 0.03, "n_estimators": 500}  # base fixed in docs/decisions.md
EXPECTED_P = 0.3067  # positive rate of the labelled rows before 2018, docs/decisions.md "scale_pos_weight rate"

logged = read_log(LOG_PATH)  # rows of the earlier phases
print(f"log rows at start: {len(logged)}, by phase: {dict(Counter(r['phase'] for r in logged))}", flush=True)  # what a rerun will skip

# Base configuration: best non-stage0 trial of phases 1 and 2 by mean PR-AUC (g_best_1000 is eligible, manual rows are not in the pool).
pool = [r for r in logged if r["phase"] in ("random", "grid") and r["trial"] != "stage0"]  # stage0 is excluded by trial id
base_row = max(pool, key=lambda r: r["mean_pr_auc"])  # highest mean PR-AUC over the five folds
base = base_row["params"]  # full parameters of the base
print(f"base trial {base_row['trial']} (mean PR-AUC {base_row['mean_pr_auc']:.5f}), params: {base}", flush=True)
is_g13 = (str(base_row["trial"]) == EXPECTED_BASE["trial"]) and all(base[k] == v for k, v in EXPECTED_BASE.items() if k != "trial")  # compared with the logged design
print(f"base is g13 (max_depth 5, reg_lambda 0.3, learning_rate 0.03, n_estimators 500): {is_g13}", flush=True)
if not is_g13:  # stop before any table build or fit
    sys.exit("the base is not g13, nothing was run")

works = pd.read_parquet(ROOT / "data/processed/works_checkpoint.parquet")  # cleaned Works table
table = build_what_table(build_artist_years(works))  # labelled artist-year rows, history >= 3

# Positive rate of the labelled rows whose year and next active year are before 2018, no row from 2018 onward.
before_2018 = table[(table["year"] < 2018) & (table["next_active_year"] < 2018)]  # labelled rows, both years before the holdout
positives, rows = int(before_2018[TARGET].sum()), len(before_2018)  # counts behind p
p = positives / rows  # positive rate
print(f"p = {p:.6f} ({positives} positives of {rows} rows), scale_pos_weight (1 - p) / p = {(1 - p) / p:.6f}", flush=True)
if round(p, 4) != EXPECTED_P:  # stop before any fit
    sys.exit(f"p differs from {EXPECTED_P}, nothing was fitted")

# Five trials, each changes one thing from the base (the two slow trials change learning_rate and n_estimators together).
trials = {
    "m_depth4": {"max_depth": 4},  # depth outside the grid, between 3 and 5
    "m_depth6": {"max_depth": 6},  # depth outside the grid, between 5 and 7
    "m_lr_0.02": {"learning_rate": 0.02, "n_estimators": 750},  # slower, learning_rate x n_estimators stays at 15
    "m_lr_0.012": {"learning_rate": 0.012, "n_estimators": 1250},  # slower still, learning_rate x n_estimators stays at 15
    "m_spw": {"scale_pos_weight": round((1 - p) / p, 3)},  # negatives per positive, balances the two classes
}
for trial, change in trials.items():  # parameter difference of each trial against the base
    print(f"{trial}: changes versus base: {{{', '.join(f'{k}: {base.get(k, None)} -> {v}' for k, v in change.items())}}}", flush=True)

features = build_what_features(works)  # the 23 columns of build_what_features (vocabulary, novelty, timing and format, genre flags); the 2 table inputs complete the 25 WHAT_FEATURES
folds = make_what_folds(table)  # five development folds, the holdout is never read
fold_data = prepare_what_fold_data(folds, features, TARGET)  # DMatrix per fold, built once for all trials
done = {r["trial"] for r in logged}  # resume: trials already in the log
for trial, change in trials.items():
    if trial in done:  # already finished in an earlier run
        continue
    params = {**base, **change}  # base with the one change
    try:
        result = run_trial(fold_data, params)  # fit on all folds
    except Exception as error:  # keep going if one trial fails
        print(f"{TARGET} {trial} FAILED: {error}", flush=True)
        continue
    _append(LOG_PATH, {"phase": "manual", "target": TARGET, "trial": trial, "params": params, **result})  # logged as soon as it ends
    print(f"{TARGET} {trial}: mean PR-AUC {result['mean_pr_auc']:.5f} in {result['seconds']}s", flush=True)
final = read_log(LOG_PATH)  # log after this run
print(f"log rows at end: {len(final)}, by phase: {dict(Counter(r['phase'] for r in final))}", flush=True)
