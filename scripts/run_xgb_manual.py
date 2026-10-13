"""Manual tuning phase for XGBoost: a short, hand-picked list of trials per target, appended to results/trials/.

Usage:
    venv/bin/python3 scripts/run_xgb_manual.py next_1y next_2y
"""

import sys  # command line arguments and module search path
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the snapshots

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import make_folds, split_holdout, year_distribution  # fold generation, same as the earlier phases
from tuning import _append, prepare_fold_data, read_log, run_trial  # same trial function and log as the earlier phases

snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached snapshots
development = split_holdout(snapshots)["development"]  # the holdout stays sealed
low_volume_years = year_distribution(snapshots).query("low_volume").index  # years no test tranche may touch
for target in sys.argv[1:]:  # one session per target, tuned separately
    log_path = ROOT / f"results/trials/xgb_{target}.jsonl"  # same log as the random and grid phases
    grid = [r for r in read_log(log_path) if r["phase"] == "grid" and r["trial"] != "g_best_1000"]  # finished grid combos
    best = max(grid, key=lambda r: r["mean_pr_auc"])["params"]  # winning grid combo, base of the learning rate and scale_pos_weight trials
    by_depth = {d: max((r for r in grid if r["params"]["max_depth"] == d), key=lambda r: r["mean_pr_auc"])["params"] for d in (7, 9)}  # best grid combo at each depth
    rate = development[target].dropna().mean()  # positive rate of the target
    trials = {
        "m_depth7": by_depth[7],  # depth 7 with its best grid learning_rate and reg_lambda
        "m_depth9": by_depth[9],  # depth 9 with its best grid learning_rate and reg_lambda
        "m_lr_0.02": {**best, "learning_rate": 0.02, "n_estimators": 750},  # below the grid range, 750 trees keep learning_rate x trees at the 0.03 x 500 level
        "m_lr_0.12": {**best, "learning_rate": 0.12},  # above the grid range
        "m_spw": {**best, "scale_pos_weight": round((1 - rate) / rate, 3)},  # negatives per positive, balances the two classes
    }
    done = {r["trial"] for r in read_log(log_path)}  # resume: trials already logged
    fold_data = prepare_fold_data(make_folds(development, target=target, low_volume_years=low_volume_years), target)  # features and DMatrix built once
    for trial, params in trials.items():
        if trial in done:
            continue
        try:
            result = run_trial(fold_data, params)  # fit on all folds
        except Exception as error:  # keep going if one trial fails
            print(f"{target} {trial} FAILED: {error}", flush=True)
            continue
        _append(log_path, {"phase": "manual", "target": target, "trial": trial, "params": params, **result})  # logged as soon as it ends
        print(f"{target} {trial}: mean PR-AUC {result['mean_pr_auc']:.4f} in {result['seconds']}s", flush=True)
