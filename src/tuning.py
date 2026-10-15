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
from scipy.stats import spearmanr  # rank correlation of the style forecast trials
from style_forecast import DEV_FIRST_TARGET, DEV_LAST_TARGET, TRANCHES, boot_ci, feature_columns, fit_model, predict_bundle, top_hits  # walk-forward pieces of the style forecast

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


# Random Forest of the style forecast: stage 0 and the random search (phase 1).
# Scored on the 25 development target years only, no later year is ever touched.
STAGE0_RF = {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 10, "max_features": "sqrt", "random_state": SEED}  # design settings

def sample_rf_params(trial_id: int, seed: int = SEED) -> dict:
    """One random Random Forest combination, reproducible from the trial id (same stream scheme as sample_params)."""
    rng = np.random.default_rng([seed, trial_id])  # one stream per trial id
    return {
        "n_estimators": 300,  # fixed during the random search
        "max_depth": int(rng.integers(3, 15)),  # tree depth, 3 to 14 inclusive
        "min_samples_leaf": int(round(float(np.exp(rng.uniform(np.log(1), np.log(50)))))),  # log-uniform 1 to 50, rounded
        "max_features": ["sqrt", 0.3, 0.5, 0.8][int(rng.integers(0, 4))],  # four options with equal probability
        "bootstrap": True,  # rows drawn with replacement
        "max_samples": float(rng.uniform(0.3, 1.0)),  # fraction of rows per tree
        "random_state": SEED,  # reproducible forest
    }

def run_rf_trial(panel: pd.DataFrame, params: dict) -> dict:
    """Walk-forward over the 25 development target years: Spearman and hits per year, tranche means and seconds."""
    start = time.time()  # trial clock
    cols = feature_columns(panel)  # same inputs as the design models
    years, spearman, hits10, hits5 = [], [], [], []  # one value per target year
    for target in range(DEV_FIRST_TARGET, DEV_LAST_TARGET + 1):  # one target year at a time
        train = panel[(panel["origin"] <= target - 2) & panel["label"].notna()]  # labels known at the origin
        test = panel[panel["origin"] == target - 1]  # styles eligible at the origin
        pred = predict_bundle(fit_model("random_forest", train, cols, params), test)  # forest with the trial settings
        label = test["label"].to_numpy()  # actual log growth
        years.append(target)  # year of this fold
        spearman.append(float(spearmanr(pred, label)[0]))  # rank correlation over all eligible styles
        hits10.append(int(top_hits(pred, label, 10)))  # overlap of the top 10
        hits5.append(int(top_hits(pred, label, 5)))  # overlap of the top 5
    tranche = [float(np.mean([s for y, s in zip(years, spearman) if a <= y <= b])) for a, b in TRANCHES]  # five tranche means
    return {
        "years": years,  # target years, in order
        "spearman": spearman,  # per year
        "hits10": hits10,  # per year
        "hits5": hits5,  # per year
        "tranche_means": tranche,  # mean Spearman per project tranche
        "mean_spearman": float(np.mean(spearman)),  # selection metric
        "mean_hits10": float(np.mean(hits10)),  # descriptive
        "seconds": round(time.time() - start, 1),  # cost of the trial
    }

def run_rf_random_search(panel: pd.DataFrame, n_trials: int, log_path) -> None:
    """Random search: trial "stage0" first, then trials 1..n_trials, skipping those already logged."""
    done = {r["trial"] for r in read_log(log_path)}  # trials finished in an earlier run
    plan = [("stage0", STAGE0_RF)] + [(i, sample_rf_params(i)) for i in range(1, n_trials + 1)]  # stage 0 as the reference row
    for trial, params in plan:
        if trial in done:  # resume: already in the log
            continue
        result = run_rf_trial(panel, params)  # 25 walk-forward years
        _append(log_path, {"phase": "random", "trial": trial, "params": params, **result})  # logged as soon as it ends
        print(f"rf style trial {trial}: mean Spearman {result['mean_spearman']:.4f}, hits@10 {result['mean_hits10']:.2f} in {result['seconds']}s", flush=True)  # progress for the console log


# Random Forest of the style forecast: phase 2, grid around the best region of the random search.
RF_GRID = {
    "max_depth": [9, 11, 13],  # range of the 10 best phase 1 trials
    "min_samples_leaf": [3, 7, 15],  # leaf sizes around the phase 1 median of the best trials
    "max_features": [0.3, 0.5, 0.8],  # share of features tried at each split
}

def rf_grid_max_samples(log_path) -> float:
    """Median max_samples of the 10 best phase 1 trials by mean Spearman, read from the log (stage 0 has none)."""
    random_rows = [r for r in read_log(log_path) if r["phase"] == "random" and r["trial"] != "stage0"]  # phase 1 trials only
    best = sorted(random_rows, key=lambda r: r["mean_spearman"], reverse=True)[:10]  # ten best by selection metric
    return float(np.median([r["params"]["max_samples"] for r in best]))  # median of their max_samples

def run_rf_grid_search(panel: pd.DataFrame, log_path) -> None:
    """Full product of RF_GRID as trials g1 to g27 (first axis slowest), skipping those already logged."""
    max_samples = rf_grid_max_samples(log_path)  # fixed value, computed before any grid row is appended
    done = {r["trial"] for r in read_log(log_path)}  # trials finished in an earlier run
    combos = itertools.product(RF_GRID["max_depth"], RF_GRID["min_samples_leaf"], RF_GRID["max_features"])  # first axis slowest
    for i, (depth, leaf, features) in enumerate(combos, start=1):
        trial = f"g{i}"  # grid trial id
        if trial in done:  # resume: already in the log
            continue
        params = {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf, "max_features": features, "bootstrap": True, "max_samples": max_samples, "random_state": SEED}  # same keys as phase 1
        result = run_rf_trial(panel, params)  # 25 walk-forward years
        _append(log_path, {"phase": "grid", "trial": trial, "params": params, **result})  # logged as soon as it ends
        print(f"rf style trial {trial}: mean Spearman {result['mean_spearman']:.4f}, hits@10 {result['mean_hits10']:.2f} in {result['seconds']}s", flush=True)  # progress for the console log


# Random Forest of the style forecast: phase 3, manual trials that change one setting of the best configuration so far.
def rf_base_params(log_path) -> tuple:
    """Trial id and parameters of the best non-stage0 trial of every phase by mean Spearman, read from the log."""
    rows = [r for r in read_log(log_path) if r["trial"] != "stage0"]  # every tuned trial, stage 0 is the reference only
    best = max(rows, key=lambda r: r["mean_spearman"])  # selection metric
    return best["trial"], best["params"]  # id and settings of the base configuration

def run_rf_manual_search(panel: pd.DataFrame, log_path) -> None:
    """Seven one-setting changes of the base configuration, skipping those already logged. The slow criterion runs last."""
    _, base = rf_base_params(log_path)  # base configuration, read before any manual row is appended
    no_bootstrap = {k: v for k, v in base.items() if k != "max_samples"}  # max_samples is not allowed without bootstrap
    plan = [
        ("m_depth16", {**base, "max_depth": 16}),  # deeper than the phase 1 range
        ("m_depth_none", {**base, "max_depth": None}),  # unlimited depth
        ("m_leaf30", {**base, "min_samples_leaf": 30}),  # larger leaves than the grid edge
        ("m_trees600", {**base, "n_estimators": 600}),  # check that 300 trees are enough
        ("m_samples1.0", {**base, "max_samples": 1.0}),  # every row drawn per tree
        ("m_nobootstrap", {**no_bootstrap, "bootstrap": False}),  # each tree sees all rows
        ("m_absolute_error", {**base, "criterion": "absolute_error"}),  # robust split criterion, much slower, last
    ]
    done = {r["trial"] for r in read_log(log_path)}  # trials finished in an earlier run
    for trial, params in plan:
        if trial in done:  # resume: already in the log
            continue
        result = run_rf_trial(panel, params)  # 25 walk-forward years
        _append(log_path, {"phase": "manual", "trial": trial, "params": params, **result})  # logged as soon as it ends
        print(f"rf style trial {trial}: mean Spearman {result['mean_spearman']:.4f}, hits@10 {result['mean_hits10']:.2f} in {result['seconds']}s", flush=True)  # progress for the console log


# Random Forest of the style forecast: phase 4, adoption evaluation and train versus validation diagnostic.
def rf_adoption(log_path) -> dict:
    """The three adoption conditions of the candidate (best non-stage0 trial) against stage 0, computed from the log."""
    rows = read_log(log_path)  # all trials
    stage0 = next(r for r in rows if r["trial"] == "stage0")  # reference row
    candidate = max((r for r in rows if r["trial"] != "stage0"), key=lambda r: r["mean_spearman"])  # selection rule
    diff = np.array(candidate["spearman"]) - np.array(stage0["spearman"])  # yearly difference, same 25 years
    mean, low, high = boot_ci(diff)  # paired year bootstrap, 2,000 draws, seed 1, percentiles 2.5 and 97.5
    tranches = int(sum(c >= s for c, s in zip(candidate["tranche_means"], stage0["tranche_means"])))  # tranches where the candidate is at least stage 0
    conditions = {
        "1 lower bound of paired bootstrap difference above 0": (low > 0, (mean, low, high)),  # mean, low, high of the difference
        "2 mean Spearman at least stage 0 in at least 4 of 5 tranches": (tranches >= 4, tranches),  # tranche count
        "3 mean hits@10 not lower than stage 0": (candidate["mean_hits10"] >= stage0["mean_hits10"], (candidate["mean_hits10"], stage0["mean_hits10"])),  # candidate and stage 0
    }
    adopted = candidate["mean_spearman"] > stage0["mean_spearman"] and all(ok for ok, _ in conditions.values())  # stage 0 stays if it has the highest mean
    return {"candidate": candidate, "stage0": stage0, "conditions": conditions, "adopted": adopted, "locked": candidate["params"] if adopted else stage0["params"]}  # everything to print

def rf_overfit_table(panel: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Per development target year: Spearman on the fold's own training rows, on the target year, and the gap."""
    cols = feature_columns(panel)  # same inputs as the design models
    rows = {}  # year -> values
    for target in range(DEV_FIRST_TARGET, DEV_LAST_TARGET + 1):  # one target year at a time
        train = panel[(panel["origin"] <= target - 2) & panel["label"].notna()]  # same training rows as the walk-forward
        test = panel[panel["origin"] == target - 1]  # styles eligible at the origin
        bundle = fit_model("random_forest", train, cols, params)  # fitted once, scored on both sets
        train_sp = float(spearmanr(predict_bundle(bundle, train), train["label"])[0])  # forest on the rows it was fitted on
        val_sp = float(spearmanr(predict_bundle(bundle, test), test["label"])[0])  # forest on the target year
        rows[target] = {"train": train_sp, "validation": val_sp, "gap": train_sp - val_sp}  # keep
    return pd.DataFrame(rows).T  # years as rows

def rf_overfit_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Tranche means and overall mean of the diagnostic table, plus the worst single-year gap."""
    out = {f"{a}-{b}": table.loc[a:b].mean() for a, b in TRANCHES}  # five tranche means
    out["overall"] = table.mean()  # 25-year mean
    out["worst year gap"] = table.loc[table["gap"].idxmax()]  # row of the year with the largest gap
    return pd.DataFrame(out).T  # one row per tranche, overall and worst year
