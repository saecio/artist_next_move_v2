"""Label look-ahead check of development fold 5 for WHEN next_1y, WHEN next_2y and WHAT.

Counts the fold 5 test rows whose label depends on years of the holdout period (2018 onward) and measures
whether dropping them changes the fold 5 PR-AUC of the final configuration. The final configuration is
FINAL_XGB_PARAMS for WHEN and the stage 0 model for WHAT. Each model is refitted once on the fold 5 train rows,
the fit is checked against the logged fold 5 PR-AUC, and the same predictions are re-scored without the affected
rows (no refit). Only development rows are scored, no holdout row is read. The output file is created once.

Usage:
    venv/bin/python3 scripts/run_label_lookahead_check.py
"""

import sys  # module search path and exit
import time  # runtime of each target
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the tables and build the output
from sklearn.metrics import average_precision_score  # PR-AUC, same metric as the tuning trials

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import make_folds, make_what_folds, split_holdout, year_distribution  # development fold builders
from config import CONFIG  # project settings read from config.yaml
from models import FINAL_XGB_PARAMS, make_what_xgboost, make_xgboost  # final WHEN configs and the model pairs
from what_target import build_artist_years, build_what_features, build_what_table  # WHAT table and inputs

HOLDOUT_START = CONFIG["years"]["holdout_start"]  # first holdout year (2018)
HORIZON = {"next_1y": 1, "next_2y": 2}  # years ahead that each WHEN label looks at
TARGETS = ["next_1y", "next_2y", "new_style_year"]  # the three targets, WHAT last
TOLERANCE = 1e-4  # largest accepted difference to the logged fold 5 PR-AUC
OUT_PATH = ROOT / "results/label_lookahead/fold5_label_lookahead.csv"  # new file, never overwritten

if OUT_PATH.exists():  # never overwrite an existing result
    sys.exit(f"{OUT_PATH.name} already exists, nothing was run")  # stop before any work
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # create results/label_lookahead on first use

snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached WHEN snapshots
development = split_holdout(snapshots)["development"]  # development years only, the holdout is never scored
low_volume_years = year_distribution(snapshots).query("low_volume").index  # years no test tranche may touch
rows = []  # one output row per target
pd.set_option("display.width", 250)  # readable printed tables

for target in TARGETS:  # one target at a time
    start = time.time()  # clock of this target
    if target == "new_style_year":  # WHAT target
        works = pd.read_parquet(ROOT / "data/processed/works_checkpoint.parquet")  # cleaned Works table
        table = build_what_table(build_artist_years(works))  # labelled artist-year rows
        features = build_what_features(works)  # WHAT inputs
        fold = make_what_folds(table)[4]  # fold 5, same builder as the tuning
        fit, score = make_what_xgboost(features, target=target)  # stage 0 model, the WHAT final (no overrides)
        logged = pd.read_csv(ROOT / "results/trials/xgb_what_compare.csv")["stage0_pr_auc"].iloc[4]  # logged fold 5 PR-AUC
    else:  # WHEN target
        fold = make_folds(development, target=target, low_volume_years=low_volume_years)[4]  # fold 5, same builder as the tuning
        fit, score = make_xgboost(target, **FINAL_XGB_PARAMS[target])  # final tuned configuration
        logged = pd.read_csv(ROOT / f"results/trials/xgb_final_{target}.csv")["pr_auc"].iloc[4]  # logged fold 5 PR-AUC
    train, test = fold["train"], fold["test"]  # fold 5 train and test rows
    if target == "new_style_year":  # WHAT label is the next new style year of the next active year
        affected = (test["next_active_year"] >= HOLDOUT_START).to_numpy()  # label looks at a year of the holdout
        print(f"{target}: affected = next_active_year >= {HOLDOUT_START}, test_rows_label_beyond_tranche of the fold = {fold['test_rows_label_beyond_tranche']}", flush=True)  # same definition as the fold field
        print(f"{target}: max train next_active_year {train['next_active_year'].max()} < first test year {test['year'].min()}: {train['next_active_year'].max() < test['year'].min()}", flush=True)  # train labels resolved before the test
    else:  # WHEN label window is year + 1 or year + 2
        affected = (test["year"] + HORIZON[target] >= HOLDOUT_START).to_numpy()  # label window reaches a holdout year
        print(f"{target}: affected = year + {HORIZON[target]} >= {HOLDOUT_START}", flush=True)  # definition used
        print(f"{target}: max train year + {HORIZON[target]} = {train['year'].max() + HORIZON[target]} < first test year {test['year'].min()}: {train['year'].max() + HORIZON[target] < test['year'].min()}", flush=True)  # train label windows end before the test
    print(test.assign(affected=affected).groupby("year")["affected"].agg(rows="size", affected="sum").to_string(), flush=True)  # breakdown by origin year
    n_test, n_affected = len(test), int(affected.sum())  # fold 5 test rows and affected rows
    print(f"{target}: fold 5 test rows {n_test}, affected {n_affected} ({n_affected / n_test:.4%})", flush=True)  # count and share
    if n_affected == 0:  # nothing to re-score
        print(f"{target}: no affected row, steps 3 and 4 skipped", flush=True)  # say it explicitly
        rows.append({"target": target, "n_test_fold5": n_test, "n_affected": 0, "share_affected": 0.0})  # count columns only
        continue  # next target
    y = test[target].to_numpy()  # resolved test labels
    scores = score(test, fit(train))  # one refit on the fold 5 train rows, predictions reused below
    full = average_precision_score(y, scores)  # fold 5 PR-AUC on every test row
    print(f"{target}: refit PR-AUC {full:.6f}, logged {logged:.6f}, difference {abs(full - logged):.2e}", flush=True)  # reproduction check
    if abs(full - logged) > TOLERANCE:  # the refit does not match the log
        sys.exit(f"{target}: the fold 5 PR-AUC does not reproduce the logged value (difference {abs(full - logged):.2e}), nothing was written")  # stop and report
    keep = ~affected  # rows whose label does not reach the holdout years
    dropped = average_precision_score(y[keep], scores[keep])  # same predictions, affected rows removed
    rows.append({
        "target": target,  # WHEN next_1y, WHEN next_2y or WHAT
        "n_test_fold5": n_test,  # fold 5 test rows
        "n_affected": n_affected,  # rows whose label reaches the holdout years
        "share_affected": n_affected / n_test,  # share of the fold 5 test rows
        "pr_auc_full": full,  # PR-AUC on every test row
        "pr_auc_dropped": dropped,  # PR-AUC without the affected rows
        "diff": dropped - full,  # dropped minus full
        "pos_rate_full": y.mean(),  # positive rate on every test row
        "pos_rate_dropped": y[keep].mean(),  # positive rate without the affected rows
    })
    print(f"{target}: seconds {time.time() - start:.1f}", flush=True)  # runtime of this target

result = pd.DataFrame(rows)  # one row per target
print(result.to_string(index=False), flush=True)  # CSV content
result.to_csv(OUT_PATH, mode="x", index=False)  # exclusive creation, fails if the file exists
print(f"written {OUT_PATH.name}", flush=True)  # done
