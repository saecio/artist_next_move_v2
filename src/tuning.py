"""
XGBoost tuning helpers: per-fold data built once, and an append-only trial log.

The expensive part of a trial is the model fit, so each fold's features,
labels and DMatrix are built once (`prepare_fold_data`) and reused by every
trial. Each finished trial is appended to a JSON-lines file right away, so
an interrupted search keeps what it finished and resumes where it stopped.
There is no bootstrap CI here, it is computed once for the final chosen
configuration through `evaluate_folds`.

Usage:
    fold_data = prepare_fold_data(folds, "next_1y")
    run_random_search(fold_data, "next_1y", 35, "results/trials/xgb_next_1y.jsonl")
"""

from __future__ import annotations

import itertools  # full product of the grid
import json  # trial log format, one JSON object per line
import time  # seconds per trial
from pathlib import Path  # log file path

import numpy as np  # random sampling of the search space
import pandas as pd  # type of the WHAT features table
import xgboost as xgb  # DMatrix and the training loop
from sklearn.metrics import average_precision_score, roc_auc_score  # per-fold metrics

from models import SEED, WHAT_FEATURES, _what_matrix, build_features  # same feature matrices as every other model

# Values that stay fixed while the other hyperparameters are searched. They
# are the stage 0 values (src/models.py make_xgboost) that are not searched.
FIXED_PARAMS = {
    "objective": "binary:logistic",  # probability of target == 1, same as XGBClassifier
    "tree_method": "hist",  # histogram split search, fast for millions of rows
    "nthread": -1,  # every CPU core
    "seed": SEED,  # reproducible runs
}

# Stage 0 starting point, trial "stage0" of the log, so the tuning pipeline
# can be checked against the evaluate_folds numbers before any search.
STAGE0 = {
    "max_depth": 6, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "reg_alpha": 0.0, "learning_rate": 0.1, "n_estimators": 300,
}


def prepare_fold_data(folds: list, target: str) -> list:
    """Build each fold's DMatrix, labels and sizes once, held in memory for the tuning session."""
    data = []  # one dict per fold
    for f in folds:
        train, test = f["train"], f["test"]  # fold dataframes
        data.append({
            "fold": f["fold"],  # 1-based fold number
            "dtrain": xgb.DMatrix(build_features(train).astype("float32"), label=train[target].to_numpy()),  # NaN stays missing
            "dtest": xgb.DMatrix(build_features(test).astype("float32")),  # test features, no label needed to predict
            "y_test": test[target].to_numpy(),  # resolved test labels
        })
    return data  # list of per-fold dicts


def prepare_what_fold_data(folds: list, features: pd.DataFrame, target: str = "new_style_year") -> list:
    """WHAT sibling of prepare_fold_data: same dict per fold, inputs are the WHAT_FEATURES columns.

    `folds` come from make_what_folds (development years only) and `features` is the output of
    build_what_features. NaN stays missing in the DMatrix, no imputation, as in the WHEN version.
    """
    data = []  # one dict per fold
    for f in folds:
        train, test = f["train"], f["test"]  # fold dataframes
        data.append({
            "fold": f["fold"],  # 1-based fold number
            "dtrain": xgb.DMatrix(_what_matrix(train, features, WHAT_FEATURES).astype("float32"), label=train[target].to_numpy()),  # NaN stays missing
            "dtest": xgb.DMatrix(_what_matrix(test, features, WHAT_FEATURES).astype("float32")),  # test features, no label needed to predict
            "y_test": test[target].to_numpy(),  # resolved test labels
        })
    return data  # list of per-fold dicts


def sample_params(trial_id: int, seed: int = SEED) -> dict:
    """One random combination of the searched hyperparameters, reproducible from the trial id.

    Ranges come from the reference table (max_depth 3-10, subsample 0.5-1.0,
    colsample_bytree 0.3-1.0, reg_alpha and reg_lambda 0.001-9). Two ranges
    are narrowed: learning_rate to 0.03-0.3 (the table's 0.00001-1 is far
    wider than 300 rounds can use) and min_child_weight to 5-500 (the table
    has no range, 50 was the stage 0 value). n_estimators stays at 300 here
    and is tuned in the last phase together with learning_rate.
    """
    rng = np.random.default_rng([seed, trial_id])  # one stream per trial id, so trial i is the same on every run
    log_uniform = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # even spread across orders of magnitude
    return {
        "max_depth": int(rng.integers(3, 11)),  # tree depth, 3 to 10 inclusive
        "min_child_weight": log_uniform(5, 500),  # minimum hessian sum in a leaf
        "subsample": float(rng.uniform(0.5, 1.0)),  # fraction of rows per tree
        "colsample_bytree": float(rng.uniform(0.3, 1.0)),  # fraction of features per tree
        "reg_lambda": log_uniform(0.001, 9),  # L2 penalty on leaf values
        "reg_alpha": log_uniform(0.001, 9),  # L1 penalty on leaf values
        "learning_rate": log_uniform(0.03, 0.3),  # step size of each tree
        "n_estimators": 300,  # boosting rounds, fixed during the random search
    }


def run_trial(fold_data: list, params: dict) -> dict:
    """Fit the params on every fold's train data and score that fold's test data."""
    start = time.time()  # fit time of the whole trial
    xgb_params = {**FIXED_PARAMS, **{k: v for k, v in params.items() if k != "n_estimators"}}  # n_estimators is the round count, not a param
    pr_auc, roc_auc = [], []  # one value per fold
    for d in fold_data:
        booster = xgb.train(xgb_params, d["dtrain"], num_boost_round=params["n_estimators"])  # no eval set, no early stopping
        scores = booster.predict(d["dtest"])  # probability of target == 1
        pr_auc.append(float(average_precision_score(d["y_test"], scores)))  # same PR-AUC as evaluate_folds
        roc_auc.append(float(roc_auc_score(d["y_test"], scores)))  # for reference
    return {
        "pr_auc": pr_auc,  # per fold
        "roc_auc": roc_auc,  # per fold
        "mean_pr_auc": float(np.mean(pr_auc)),  # ranking value, always read next to the per-fold list
        "seconds": round(time.time() - start, 1),  # cost of the trial
    }


def read_log(path) -> list:
    """All finished trials in a log file, empty list if the file does not exist yet."""
    path = Path(path)  # accept str or Path
    if not path.exists():  # first run
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]  # one trial per line


def _append(path, record: dict) -> None:
    """Append one finished trial to the log right away, so an interruption loses nothing."""
    path = Path(path)  # accept str or Path
    path.parent.mkdir(parents=True, exist_ok=True)  # create results/trials on first use
    with path.open("a") as handle:  # append-only
        handle.write(json.dumps(record) + "\n")  # one JSON object per line


def run_random_search(fold_data: list, target: str, n_trials: int, log_path) -> None:
    """Random search: trial "stage0" first, then trials 1..n_trials, skipping those already logged."""
    done = {r["trial"] for r in read_log(log_path)}  # trials finished in an earlier run
    plan = [("stage0", STAGE0)] + [(i, sample_params(i)) for i in range(1, n_trials + 1)]  # stage 0 as the reference row
    for trial, params in plan:
        if trial in done:  # resume: already in the log
            continue
        result = run_trial(fold_data, params)  # fit on all folds
        _append(log_path, {"phase": "random", "target": target, "trial": trial, "params": params, **result})  # logged as soon as it ends
        print(f"{target} trial {trial}: mean PR-AUC {result['mean_pr_auc']:.4f} in {result['seconds']}s", flush=True)  # progress for the background log


# Grid search around the best region of the random search. Everything not
# listed in GRID is fixed at GRID_FIXED.
GRID = {
    "max_depth": [7, 9, 11],  # tree depth
    "reg_lambda": [2, 5, 15],  # L2 penalty on leaf values
    "learning_rate": [0.03, 0.05, 0.08],  # step size of each tree
}
GRID_FIXED = {"subsample": 0.8, "colsample_bytree": 0.55, "min_child_weight": 50, "reg_alpha": 0.01, "n_estimators": 500}


def run_grid_search(fold_data: list, target: str, log_path) -> None:
    """Full product of GRID, same trial function and log as the random search. A failed combo is logged and skipped."""
    done = {r["trial"] for r in read_log(log_path)}  # combos finished in an earlier run
    for i, values in enumerate(itertools.product(*GRID.values()), start=1):
        trial = f"g{i}"  # grid trial id, apart from the random search ids
        if trial in done:  # resume: already in the log
            continue
        params = {**GRID_FIXED, **dict(zip(GRID, values))}  # one combination
        try:
            result = run_trial(fold_data, params)  # fit on all folds
        except Exception as error:  # keep going if one combo fails
            print(f"{target} trial {trial} FAILED: {error}", flush=True)
            continue
        _append(log_path, {"phase": "grid", "target": target, "trial": trial, "params": params, **result})  # logged as soon as it ends
        print(f"{target} trial {trial}: {params['max_depth']}/{params['reg_lambda']}/{params['learning_rate']} mean PR-AUC {result['mean_pr_auc']:.4f} in {result['seconds']}s", flush=True)

    grid_rows = [r for r in read_log(log_path) if r["phase"] == "grid" and r["trial"] != "g_best_1000"]  # finished grid combos
    best = max(grid_rows, key=lambda r: r["mean_pr_auc"])  # best by mean PR-AUC
    if best["params"]["learning_rate"] == 0.03 and "g_best_1000" not in {r["trial"] for r in read_log(log_path)}:  # 500 trees may be too few at this rate
        params = {**best["params"], "n_estimators": 1000}  # same combo, twice the trees
        result = run_trial(fold_data, params)
        _append(log_path, {"phase": "grid", "target": target, "trial": "g_best_1000", "params": params, **result})
        print(f"{target} trial g_best_1000 (from {best['trial']}): mean PR-AUC {result['mean_pr_auc']:.4f}", flush=True)


# WHAT grid (docs/decisions.md, "WHAT XGBoost phase 1 (random search) results and phase 2 (grid) design").
# Everything not listed in WHAT_GRID is fixed at WHAT_GRID_FIXED.
WHAT_GRID = {
    "max_depth": [3, 5, 7],  # tree depth, the phase 1 top 10 span 3 to 7
    "reg_lambda": [0.01, 0.3, 2],  # L2 penalty on leaf values, the phase 1 top 10 span 0.001 to 2
    "learning_rate": [0.03, 0.05, 0.08],  # step size of each tree, same values as the WHEN grid
}
WHAT_GRID_FIXED = {"subsample": 0.811, "colsample_bytree": 0.538, "min_child_weight": 81.4, "reg_alpha": 0.343, "n_estimators": 500}  # median of the 10 best phase 1 trials


def run_what_grid_search(fold_data: list, target: str, log_path) -> None:
    """WHAT sibling of run_grid_search: full product of WHAT_GRID, same trial function and log. A failed combo is logged and skipped."""
    done = {r["trial"] for r in read_log(log_path)}  # combos finished in an earlier run
    for i, values in enumerate(itertools.product(*WHAT_GRID.values()), start=1):
        trial = f"g{i}"  # grid trial id, apart from the random search ids
        if trial in done:  # resume: already in the log
            continue
        params = {**WHAT_GRID_FIXED, **dict(zip(WHAT_GRID, values))}  # one combination
        try:
            result = run_trial(fold_data, params)  # fit on all folds
        except Exception as error:  # keep going if one combo fails
            print(f"{target} trial {trial} FAILED: {error}", flush=True)
            continue
        _append(log_path, {"phase": "grid", "target": target, "trial": trial, "params": params, **result})  # logged as soon as it ends
        print(f"{target} trial {trial}: {params['max_depth']}/{params['reg_lambda']}/{params['learning_rate']} mean PR-AUC {result['mean_pr_auc']:.4f} in {result['seconds']}s", flush=True)

    grid_rows = [r for r in read_log(log_path) if r["phase"] == "grid" and r["trial"] != "g_best_1000"]  # finished grid combos, stage0 is phase random
    best = max(grid_rows, key=lambda r: r["mean_pr_auc"])  # best by mean PR-AUC
    if best["params"]["learning_rate"] == 0.03 and "g_best_1000" not in {r["trial"] for r in read_log(log_path)}:  # 500 trees may be too few at this rate
        params = {**best["params"], "n_estimators": 1000}  # same combo, twice the trees
        result = run_trial(fold_data, params)
        _append(log_path, {"phase": "grid", "target": target, "trial": "g_best_1000", "params": params, **result})
        print(f"{target} trial g_best_1000 (from {best['trial']}): mean PR-AUC {result['mean_pr_auc']:.4f}", flush=True)


def trials_frame(path) -> "pd.DataFrame":
    """Every logged trial as one flat row: phase, trial id, parameters, mean PR-AUC and the five fold PR-AUCs."""
    import pandas as pd  # imported here, the rest of this module does not need it
    rows = [{"phase": r["phase"], "trial": str(r["trial"]), **r["params"], "mean_pr_auc": r["mean_pr_auc"],
             **{f"fold_{i + 1}": v for i, v in enumerate(r["pr_auc"])}, "seconds": r["seconds"]} for r in read_log(path)]  # one dict per trial
    return pd.DataFrame(rows)  # one row per trial
