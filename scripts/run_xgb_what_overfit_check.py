"""Train-versus-validation diagnostic of the WHAT XGBoost stage 0 on the five development folds.

Fixed in docs/decisions.md ("WHAT stage 0 train-versus-validation diagnostic"). It is descriptive only: no threshold for
overfitting is set, no decision depends on it, and PR-AUC stays the only selection and adoption metric. Stage 0 is refitted on
each fold with xgb.train and FIXED_PARAMS exactly as the trials, and the same fitted model scores the train rows and the
validation rows. The validation PR-AUC must reproduce the logged stage0 row at 4 decimals before anything is written.
Development folds only, no holdout row is read. The output file is created once and never overwritten.

Usage:
    venv/bin/python3 scripts/run_xgb_what_overfit_check.py
"""

import sys  # module search path and exit
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the Works table and build the report
import xgboost as xgb  # booster training and prediction
from sklearn.metrics import average_precision_score, roc_auc_score  # PR-AUC and ROC-AUC, the same functions as the trials

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import make_what_folds  # development folds with the label-resolution rule
from tuning import FIXED_PARAMS, prepare_what_fold_data, read_log  # same fixed parameters, fold matrices and log reader as the trials
from what_target import WHAT_BASELINE_C_PR_AUC, build_artist_years, build_what_features, build_what_table  # WHAT table, inputs and the logged baseline (c) values

TARGET = "new_style_year"  # WHAT target
LOG_PATH = ROOT / "results/trials/xgb_what_new_style_year.jsonl"  # trial log, read only here
OUT_PATH = ROOT / "results/trials/xgb_what_overfit.csv"  # new file, never overwritten
EXPECTED_VALIDATION_PR_AUC = [0.4607, 0.4751, 0.4946, 0.4947, 0.5002]  # logged WHAT stage 0 per fold, entry "WHAT kill gate: PASSED"
SEARCHED_KEYS = {"max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "learning_rate", "n_estimators"}  # keys of a trial of the log

if OUT_PATH.exists():  # never overwrite
    sys.exit(f"{OUT_PATH.name} already exists, nothing was run")  # refuse to start, the file is created once

# Stage 0 parameters, read from the log by trial id and not typed in.
row = {str(r["trial"]): r for r in read_log(LOG_PATH)}["stage0"]  # the stage0 row of the log
params = row["params"]  # parameters as logged
assert set(params) == SEARCHED_KEYS, f"unexpected keys {sorted(set(params) ^ SEARCHED_KEYS)}"  # exactly the eight searched keys, no scale_pos_weight
print(f"stage 0 params (log row stage0): {params}", flush=True)  # printed before any fit
print(f"logged validation PR-AUC per fold: {[round(x, 4) for x in row['pr_auc']]}", flush=True)  # reference of the reproduction check

# Data: the same table, features and development folds as the gate and the trials.
works = pd.read_parquet(ROOT / "data/processed/works_checkpoint.parquet")  # cleaned Works table
table = build_what_table(build_artist_years(works))  # labelled artist-year rows, history >= 3
features = build_what_features(works)  # the 23 columns of build_what_features; the 2 table inputs complete the 25 WHAT_FEATURES
folds = make_what_folds(table)  # five development folds, the holdout is never read
fold_data = prepare_what_fold_data(folds, features, TARGET)  # DMatrix per fold, built once
assert len(folds) == len(fold_data) == 5, "expected five development folds"  # five folds
assert all(d["dtrain"].num_row() == len(f["train"]) for d, f in zip(fold_data, folds)), "a train matrix does not match its fold"  # same rows as the fold
assert max(int(f["test"]["year"].max()) for f in folds) < 2018, "a validation row reaches the holdout years"  # development years only

# Refit stage 0 on every fold, one fit per fold, and score the train rows and the validation rows with the same booster.
xgb_params = {**FIXED_PARAMS, **{k: v for k, v in params.items() if k != "n_estimators"}}  # as in run_trial, n_estimators is the round count
rows = []  # one dict per fold
for f, d in zip(folds, fold_data):  # one pass per fold, with its matrices
    booster = xgb.train(xgb_params, d["dtrain"], num_boost_round=params["n_estimators"])  # no eval set, no early stopping
    y_train, y_validation = d["dtrain"].get_label(), d["y_test"]  # labels of the rows the model was fitted on and of the validation rows
    p_train, p_validation = booster.predict(d["dtrain"]), booster.predict(d["dtest"])  # probability of target == 1, same booster for both
    train_pr, validation_pr = float(average_precision_score(y_train, p_train)), float(average_precision_score(y_validation, p_validation))  # PR-AUC on train and validation
    train_roc, validation_roc = float(roc_auc_score(y_train, p_train)), float(roc_auc_score(y_validation, p_validation))  # ROC-AUC on train and validation
    rows.append({"fold": f["fold"], "test_years": f["test_years"], "train_rows": len(y_train), "validation_rows": len(y_validation),  # fold identification and sizes
                 "train_positive_rate": float(y_train.mean()), "validation_positive_rate": float(y_validation.mean()),  # PR-AUC depends on the positive rate
                 "train_pr_auc": train_pr, "validation_pr_auc": validation_pr, "gap_pr_auc_points": (train_pr - validation_pr) * 100,  # gap train minus validation, in points
                 "train_roc_auc": train_roc, "validation_roc_auc": validation_roc, "gap_roc_auc_points": (train_roc - validation_roc) * 100})  # same for ROC-AUC
    print(f"fold {f['fold']} done: validation PR-AUC {validation_pr:.4f}", flush=True)  # progress
result = pd.DataFrame(rows)  # one row per fold, full precision

# Reproduction check, before anything is written: validation PR-AUC equals the logged row and the logged gate values at 4 decimals.
got = result["validation_pr_auc"].round(4).tolist()  # this refit
logged = [round(x, 4) for x in row["pr_auc"]]  # log row stage0
print(f"reproduction, this refit  : {got}", flush=True)  # validation PR-AUC of this refit
print(f"reproduction, log row     : {logged}", flush=True)  # validation PR-AUC of the logged stage0 row
print(f"reproduction, gate values : {EXPECTED_VALIDATION_PR_AUC}", flush=True)  # validation PR-AUC of the gate entry
print(f"largest unrounded difference to the log row: {max(abs(a - b) for a, b in zip(result['validation_pr_auc'], row['pr_auc'])):.2e}", flush=True)  # floating point difference to the log row
if got != logged or got != EXPECTED_VALIDATION_PR_AUC:  # stop before the output file, adjust nothing
    sys.exit("the validation PR-AUC does not reproduce the logged values, nothing was written")  # stop before the output file

# Printed table (4 decimals) with the mean gap, and the file (full precision, exclusive creation).
result["baseline_c_pr_auc"] = WHAT_BASELINE_C_PR_AUC  # logged baseline (c) per fold, printed for the reading only
pd.set_option("display.width", 250)  # readable printed table
print(result.drop(columns="test_years").round(4).to_string(index=False), flush=True)  # per-fold table at 4 decimals, the years are in the CSV
print(f"mean gap train minus validation: PR-AUC {result['gap_pr_auc_points'].mean():.2f} points, ROC-AUC {result['gap_roc_auc_points'].mean():.2f} points", flush=True)  # mean gap over the five folds, in points
result.drop(columns="baseline_c_pr_auc").to_csv(OUT_PATH, mode="x", index=False)  # exclusive creation, fails if the file exists, the reference column stays out of the file
print(f"written {OUT_PATH.name}", flush=True)  # confirmation of the single write
