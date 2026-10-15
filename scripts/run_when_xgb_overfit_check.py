"""Train vs test PR-AUC per fold for stage 0 and the top tuned candidates of each target.

Usage:
    venv/bin/python3 scripts/run_when_xgb_overfit_check.py next_1y next_2y
"""

import sys  # command line arguments and module search path
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the snapshots
import xgboost as xgb  # training loop
from sklearn.metrics import average_precision_score  # same PR-AUC as every other step

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import make_folds, split_holdout, year_distribution  # same folds as every earlier step
from tuning import FIXED_PARAMS, prepare_fold_data, read_log  # same data prep and fixed parameters as the trials

snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached snapshots
development = split_holdout(snapshots)["development"]  # the holdout stays sealed
low_volume_years = year_distribution(snapshots).query("low_volume").index  # years no test tranche may touch
for target in sys.argv[1:]:
    trials = {str(r["trial"]): r for r in read_log(ROOT / f"results/trials/xgb_{target}.jsonl")}  # every logged trial by id
    best = lambda phase: max((r for r in trials.values() if r["phase"] == phase and r["trial"] != "g_best_1000"), key=lambda r: r["mean_pr_auc"])["trial"]  # best trial of a phase
    final = best("grid")  # the final configuration is the best grid combination
    other_depth = "m_depth9" if trials[final]["params"]["max_depth"] == 7 else "m_depth7"  # the other depth tested by hand
    candidates = {"stage0": "stage0", f"best random ({best('random')})": best("random"), f"final grid ({final})": final, f"other depth ({other_depth})": other_depth}
    fold_data = prepare_fold_data(make_folds(development, target=target, low_volume_years=low_volume_years), target)  # built once
    rows = []
    for name, trial in candidates.items():
        params = trials[str(trial)]["params"]  # logged parameters of the trial
        xgb_params = {**FIXED_PARAMS, **{k: v for k, v in params.items() if k != "n_estimators"}}
        for d in fold_data:
            booster = xgb.train(xgb_params, d["dtrain"], num_boost_round=params["n_estimators"])  # same fit as the trial
            train_pr = average_precision_score(d["dtrain"].get_label(), booster.predict(d["dtrain"]))  # PR-AUC on the rows the model saw
            test_pr = average_precision_score(d["y_test"], booster.predict(d["dtest"]))  # PR-AUC on the held-out tranche
            rows.append({"target": target, "candidate": name, "fold": d["fold"], "train_pr_auc": train_pr, "test_pr_auc": test_pr, "gap": train_pr - test_pr})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / f"results/trials/xgb_overfit_{target}.csv", index=False)
