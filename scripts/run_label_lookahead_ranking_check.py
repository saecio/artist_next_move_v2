"""Does the config choice change when the fold 5 rows with label look-ahead are dropped?

For WHEN next_1y and next_2y the two configs are stage 0 and FINAL_XGB_PARAMS. For WHAT they are stage 0 and the
candidate m_lr_0.012 of the adoption step. Both are refitted on the fold 5 train rows, checked against the logged
fold 5 PR-AUC, and scored on the fold 5 test rows with and without the rows whose label reaches 2018 or later
(WHEN: year + horizon >= 2018, WHAT: next_active_year >= 2018). The difference B minus stage 0 gets a paired
artist-clustered bootstrap interval in both versions (same artist draws for both configs). Folds 1 to 4 use the logged
values, so the 5-fold mean difference is shown in both versions. Only development rows are scored.

Usage:
    venv/bin/python3 scripts/run_label_lookahead_ranking_check.py
"""

import sys  # module search path and exit
import time  # runtime of each target
from pathlib import Path  # project root

import numpy as np  # scores and the bootstrap quantiles
import pandas as pd  # dataframe library used to read the tables and build the output

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import _artist_bootstrap_pr_auc, _sorted_inputs, _weighted_average_precision, make_folds, make_what_folds, split_holdout, year_distribution  # folds, PR-AUC and the fast clustered bootstrap
from config import CONFIG  # project settings read from config.yaml
from models import FINAL_XGB_PARAMS, make_what_xgboost, make_xgboost  # final WHEN configs and the model pairs
from tuning import read_log  # trial log reader
from what_target import build_artist_years, build_what_features, build_what_table  # WHAT table and inputs

HOLDOUT_START = CONFIG["years"]["holdout_start"]  # first holdout year (2018)
SEED = CONFIG["project"]["SEED"]  # seed of the bootstrap draws
N_BOOT = CONFIG["bootstrap"]["draws"]  # bootstrap draws (1,000)
ALPHA = (1 - CONFIG["bootstrap"]["confidence"]) / 2  # tail of each side of the interval
HORIZON = {"next_1y": 1, "next_2y": 2}  # years ahead that each WHEN label looks at
TARGETS = ["next_1y", "next_2y", "new_style_year"]  # the three targets, WHAT last
CANDIDATE_ID = "m_lr_0.012"  # WHAT candidate of the adoption step (docs/decisions.md)
TOLERANCE = 1e-4  # largest accepted difference to the logged fold 5 PR-AUC
OUT_PATH = ROOT / "results/label_lookahead/fold5_ranking_check.csv"  # new file, never overwritten

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
        table = build_artist_years(works)  # artist-year rows
        table = build_what_table(table)  # labelled rows
        features = build_what_features(works)  # WHAT inputs
        fold = make_what_folds(table)[4]  # fold 5, same builder as the tuning
        compare = pd.read_csv(ROOT / "results/trials/xgb_what_compare.csv")  # logged stage 0 and candidate per fold
        logged = {"stage0": compare["stage0_pr_auc"].tolist(), "B": compare["candidate_pr_auc"].tolist()}  # logged PR-AUC, five folds each
        candidate = {str(r["trial"]): r for r in read_log(ROOT / "results/trials/xgb_what_new_style_year.jsonl")}[CANDIDATE_ID]  # trial row of the candidate
        config_b = CANDIDATE_ID  # name of config B in the output
        pairs = {"stage0": make_what_xgboost(features, target=target), "B": make_what_xgboost(features, target=target, **candidate["params"])}  # stage 0 and logged candidate parameters
        affected = (fold["test"]["next_active_year"] >= HOLDOUT_START).to_numpy()  # label looks at a year of the holdout
    else:  # WHEN target
        fold = make_folds(development, target=target, low_volume_years=low_volume_years)[4]  # fold 5, same builder as the tuning
        stage0_row = {str(r["trial"]): r for r in read_log(ROOT / f"results/trials/xgb_{target}.jsonl")}["stage0"]  # logged stage 0 trial
        final = pd.read_csv(ROOT / f"results/trials/xgb_final_{target}.csv")  # logged final configuration per fold
        logged = {"stage0": stage0_row["pr_auc"], "B": final["pr_auc"].tolist()}  # logged PR-AUC, five folds each
        config_b = "FINAL_XGB_PARAMS"  # name of config B in the output
        pairs = {"stage0": make_xgboost(target), "B": make_xgboost(target, **FINAL_XGB_PARAMS[target])}  # default and locked parameters
        affected = (fold["test"]["year"] + HORIZON[target] >= HOLDOUT_START).to_numpy()  # label window reaches a holdout year
    train, test = fold["train"], fold["test"]  # fold 5 train and test rows
    y, artists = test[target].to_numpy(), test["artist_id"].to_numpy()  # labels and artists of the test rows
    scores = {name: np.asarray(score(test, fit(train))) for name, (fit, score) in pairs.items()}  # one refit per config, predictions reused
    for name in scores:  # reproduction check of each config on all rows
        got = _weighted_average_precision(*_sorted_inputs(y, scores[name]))  # fold 5 PR-AUC, same function as evaluate_folds
        print(f"{target} {name}: refit {got:.6f}, logged {logged[name][4]:.6f}, difference {abs(got - logged[name][4]):.2e}", flush=True)  # reproduction result
        if abs(got - logged[name][4]) > TOLERANCE:  # the refit does not match the log
            sys.exit(f"{target} {name}: the fold 5 PR-AUC does not reproduce the logged value (difference {abs(got - logged[name][4]):.2e}), nothing was written")  # stop and report
    row = {"target": target, "config_b": config_b}  # start of the output row
    for version, keep in (("full", np.ones(len(y), dtype=bool)), ("dropped", ~affected)):  # all rows, then rows without look-ahead
        pr = {name: _weighted_average_precision(*_sorted_inputs(y[keep], scores[name][keep])) for name in scores}  # fold 5 PR-AUC per config
        boot = {name: _artist_bootstrap_pr_auc(y[keep], scores[name][keep], artists[keep], N_BOOT, SEED) for name in scores}  # same seed, so the artist draws are shared
        diff_boot = boot["B"] - boot["stage0"]  # paired difference per draw
        row.update({
            f"pr_auc_stage0_{version}": pr["stage0"],  # stage 0 on these rows
            f"pr_auc_b_{version}": pr["B"],  # config B on these rows
            f"diff_{version}": pr["B"] - pr["stage0"],  # B minus stage 0
            f"ci_{version}_low": np.nanquantile(diff_boot, ALPHA),  # lower bound of the paired interval
            f"ci_{version}_high": np.nanquantile(diff_boot, 1 - ALPHA),  # upper bound of the paired interval
        })
        mean_stage0 = np.mean(logged["stage0"][:4] + [pr["stage0"]])  # 5-fold mean, folds 1 to 4 logged
        mean_b = np.mean(logged["B"][:4] + [pr["B"]])  # same for config B
        row[f"mean5_diff_{version}"] = mean_b - mean_stage0  # 5-fold mean difference, B minus stage 0
        print(f"{target} {version}: rows {keep.sum()}, positive rate {y[keep].mean():.4f}", flush=True)  # rows scored in this version
    rows.append(row)  # keep the target
    print(f"{target}: seconds {time.time() - start:.1f}", flush=True)  # runtime of this target

result = pd.DataFrame(rows)  # one row per target
print(result.T.to_string(), flush=True)  # CSV content, one column per target
result.to_csv(OUT_PATH, mode="x", index=False)  # exclusive creation, fails if the file exists
print(f"written {OUT_PATH.name}", flush=True)  # done
