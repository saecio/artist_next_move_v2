"""Adoption evaluation of the WHAT XGBoost: stage 0 against the tuned candidate, once, on the development folds.

Both models are refitted with the exact parameters of their trial in results/trials/xgb_what_new_style_year.jsonl and
evaluated through evaluate_folds (1,000 artist draws, 95% interval, seed 1). The adoption rule is the one logged in
docs/decisions.md ("WHAT XGBoost adoption rule"): the candidate replaces stage 0 only if its interval lower bound is above
the stage 0 interval upper bound in at least 4 of the 5 folds. No threshold metric is computed and no holdout row is read.
The output file is created once and never overwritten.

Usage:
    venv/bin/python3 scripts/run_xgb_what_compare.py
"""

import sys  # module search path and exit
import time  # seconds per model
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the Works table and build the report

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import evaluate_folds, make_what_folds  # per-fold report with bootstrap intervals, development folds
from baselines import KILL_GATE_CONFIDENCE, N_BOOTSTRAP  # confidence and draws of the gate and the baseline ladder
from models import make_what_xgboost  # WHAT XGBoost pair, accepts overrides
from tuning import read_log  # trial log reader
from what_target import build_artist_years, build_what_features, build_what_table  # WHAT table and inputs

TARGET = "new_style_year"  # WHAT target
LOG_PATH = ROOT / "results/trials/xgb_what_new_style_year.jsonl"  # trial log, read only here
OUT_PATH = ROOT / "results/trials/xgb_what_compare.csv"  # new file, never overwritten
CANDIDATE_ID = "m_lr_0.012"  # candidate fixed by the selection rule (docs/decisions.md, phase 3 results)
SEED = 1  # bootstrap seed, same as the gate and the five-model comparison
MIN_FOLDS = 4  # folds with separated intervals required by the adoption rule
SEARCHED_KEYS = {"max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "learning_rate", "n_estimators"}  # keys a trial of the log may carry

if OUT_PATH.exists():  # never overwrite
    sys.exit(f"{OUT_PATH.name} already exists, nothing was run")
assert N_BOOTSTRAP == 1000 and KILL_GATE_CONFIDENCE == 0.95, "the bootstrap settings are not the logged ones"  # 1,000 draws and 95%

# Parameters of both models, read from the log by trial id and not typed in.
logged = {str(r["trial"]): r for r in read_log(LOG_PATH)}  # trial id -> logged row
pool = [r for r in logged.values() if r["trial"] != "stage0"]  # every non-stage0 trial, stage0 is excluded by trial id
best = max(pool, key=lambda r: r["mean_pr_auc"])  # candidate by the selection rule
print(f"candidate by the selection rule: {best['trial']} (mean PR-AUC {best['mean_pr_auc']:.6f}), pool of {len(pool)} trials", flush=True)
assert str(best["trial"]) == CANDIDATE_ID, f"the best non-stage0 trial is {best['trial']}, not {CANDIDATE_ID}"  # the log still gives the logged candidate
models = {"stage0": logged["stage0"], "candidate": logged[CANDIDATE_ID]}  # the two rows to evaluate
for name, row in models.items():
    params = row["params"]  # parameters as logged
    assert set(params) == SEARCHED_KEYS, f"{name}: unexpected keys {sorted(set(params) ^ SEARCHED_KEYS)}"  # no key leaked from another trial, no scale_pos_weight
    assert "scale_pos_weight" not in params, f"{name}: scale_pos_weight leaked in"  # explicit, the m_spw trial is not the candidate
    print(f"{name} ({row['trial']}) params: {params}", flush=True)  # printed before any fit
print("differences candidate vs stage0:", {k: (models["stage0"]["params"][k], models["candidate"]["params"][k]) for k in SEARCHED_KEYS if models["stage0"]["params"][k] != models["candidate"]["params"][k]}, flush=True)

# Data: the same table, features and development folds as the gate and the trials.
works = pd.read_parquet(ROOT / "data/processed/works_checkpoint.parquet")  # cleaned Works table
table = build_what_table(build_artist_years(works))  # labelled artist-year rows, history >= 3
features = build_what_features(works)  # the 23 columns of build_what_features; the 2 table inputs complete the 25 WHAT_FEATURES
folds = make_what_folds(table)  # five development folds, the holdout is never read


def checked(fit_fn, params: dict, name: str):
    """Wrap a fit function: after the first fit, print get_params and stop if any logged value is not in the model."""
    state = {"checked": False}  # the check runs once, on the first fold

    def wrapper(train):
        """Fit the model and, on the first fit only, check that its parameters equal the logged ones, exiting if they differ. Input: the train rows. Output: the fitted model."""
        model = fit_fn(train)  # fitted on the fold's train rows
        if not state["checked"]:
            got = model.get_params()  # parameters the model really has
            print(f"{name} get_params after the first fit: {got}", flush=True)
            wrong = {k: (v, got.get(k)) for k, v in params.items() if got.get(k) != v}  # logged value, value in the model
            if wrong:
                sys.exit(f"{name}: the model does not carry the logged parameters {wrong}, nothing was written")
            state["checked"] = True
        return model

    return wrapper


reports = {}  # per-fold report of each model
for name, row in models.items():
    start = time.time()  # seconds of the evaluation
    fit_fn, score_fn = make_what_xgboost(features, target=TARGET, **row["params"])  # logged parameters as overrides
    reports[name] = evaluate_folds(folds, TARGET, checked(fit_fn, row["params"], name), score_fn, n_bootstrap=N_BOOTSTRAP, confidence=KILL_GATE_CONFIDENCE, seed=SEED)
    # Reproduction check: PR-AUC per fold equals the logged trial row at 4 decimals.
    got = reports[name]["pr_auc"].round(4).tolist()  # evaluate_folds
    expected = [round(x, 4) for x in row["pr_auc"]]  # logged by run_trial
    print(f"{name} ({row['trial']}) reproduction, evaluate_folds: {got}", flush=True)
    print(f"{name} ({row['trial']}) reproduction, logged      : {expected}", flush=True)
    print(f"{name} seconds {time.time() - start:.1f}, largest unrounded difference {max(abs(a - b) for a, b in zip(reports[name]['pr_auc'], row['pr_auc'])):.2e}", flush=True)
    if got != expected:  # stop before the output file
        sys.exit(f"{name}: the PR-AUC per fold does not reproduce the logged row, nothing was written")

# Per-fold table and the adoption rule.
base, cand = reports["stage0"], reports["candidate"]  # stage 0 and candidate reports, same folds
result = pd.DataFrame({
    "test_years": base["test_years"], "train_rows": base["train_rows"], "test_rows": base["test_rows"], "positive_rate": base["positive_rate"],
    "stage0_pr_auc": base["pr_auc"], "stage0_lo": base["pr_auc_lo"], "stage0_hi": base["pr_auc_hi"],
    "candidate_pr_auc": cand["pr_auc"], "candidate_lo": cand["pr_auc_lo"], "candidate_hi": cand["pr_auc_hi"],
    "difference_points": (cand["pr_auc"] - base["pr_auc"]) * 100,  # candidate minus stage 0
    "stage0_roc_auc": base["roc_auc"], "candidate_roc_auc": cand["roc_auc"],  # for reference
})
result["non_overlap"] = result["candidate_lo"] > result["stage0_hi"]  # candidate lower bound above the stage 0 upper bound
passing = int(result["non_overlap"].sum())  # folds with separated intervals
verdict = "ADOPT the candidate" if passing >= MIN_FOLDS else "KEEP stage 0 (the candidate is not adopted)"  # logged rule
pd.set_option("display.width", 250)  # readable printed table
print(result.round(4).to_string(), flush=True)
print(f"folds with candidate lower bound above stage 0 upper bound: {passing} of {len(result)} (required {MIN_FOLDS})", flush=True)
print(f"verdict by the logged rule: {verdict}", flush=True)
result.to_csv(OUT_PATH, mode="x", index_label="fold")  # exclusive creation, fails if the file exists
print(f"written {OUT_PATH.name}", flush=True)
