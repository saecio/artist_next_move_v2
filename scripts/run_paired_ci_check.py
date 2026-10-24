"""Paired artist-clustered bootstrap of the thin PR-AUC separations, descriptive only.

Re-reads three comparisons that were judged by overlapping or barely separated intervals and gives the interval of the
PR-AUC difference itself, on the same artist draws (backtesting.paired_pr_auc_difference, 1,000 draws, 95%, seed 1):
(1) WHAT baseline ladder, (c) history and volume minus (b) history, folds 1 to 5;
(2) WHAT Logistic Regression minus Naive Bayes, folds 1 to 5;
(3) WHEN next_1y, Random Forest, HistGradientBoosting and untuned XGBoost minus Logistic Regression, fold 5.
Models are untuned, as in the comparisons they re-read. No rule, model, feature or setting changes and no holdout row is read.
The output file is created once and never overwritten.

Usage:
    venv/bin/python3 scripts/run_paired_ci_check.py
"""

import sys  # module search path and exit
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the tables and build the report

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import make_folds, make_what_folds, paired_pr_auc_difference, split_holdout, year_distribution  # folds and paired interval
from baselines import KILL_GATE_CONFIDENCE, N_BOOTSTRAP, add_what_buckets, fit_what_lookup_baseline, score_what_lookup_baseline  # settings and the lookup baselines
from models import make_hist_gradient_boosting, make_logistic_regression, make_naive_bayes, make_random_forest, make_what_logistic_regression, make_what_naive_bayes, make_xgboost  # untuned models
from what_target import build_artist_years, build_what_features, build_what_table  # WHAT table and inputs

OUT_PATH = ROOT / "results/paired_ci/paired_differences.csv"  # new file, never overwritten
if OUT_PATH.exists():  # never overwrite
    sys.exit(f"{OUT_PATH.name} already exists, nothing was run")
assert N_BOOTSTRAP == 1000 and KILL_GATE_CONFIDENCE == 0.95, "the bootstrap settings are not the logged ones"  # 1,000 draws and 95%

rows = []  # one dict per comparison and fold


def compare(case: str, target: str, fold: dict, a_name: str, b_name: str, scores_a, scores_b) -> None:
    """Add the paired difference a minus b on the test rows of one fold. Inputs: case label, target, the fold, the two model names and their test scores. Output: none, the row is stored and printed."""
    test = fold["test"]  # test rows of this fold
    result = paired_pr_auc_difference(test[target].to_numpy(), scores_a, scores_b, test["artist_id"].to_numpy())  # same draws for both
    rows.append({"case": case, "fold": fold["fold"], "a": a_name, "b": b_name, "diff_points": result["diff"] * 100, "lo_points": result["lo"] * 100, "hi_points": result["hi"] * 100, "share_draws_a_above_b": result["share_positive"]})  # in PR-AUC points
    print(rows[-1], flush=True)


# WHAT: same table, features and folds as the gate and the five-model comparison.
target = "new_style_year"  # WHAT target
works = pd.read_parquet(ROOT / "data/processed/works_checkpoint.parquet")  # cleaned Works table
table = build_what_table(build_artist_years(works))  # labelled artist-year rows, history >= 3
features = build_what_features(works)  # the 23 feature columns
what_folds = make_what_folds(table)  # five development folds, the holdout is never read
for fold in what_folds:  # bucket columns for the lookup baselines
    fold["train"] = add_what_buckets(fold["train"])  # train rows with buckets
    fold["test"] = add_what_buckets(fold["test"])  # test rows with buckets
for fold in what_folds:
    train, test = fold["train"], fold["test"]  # fold rows
    fit_b, fit_c = fit_what_lookup_baseline(train, ["history_bucket"]), fit_what_lookup_baseline(train, ["history_bucket", "volume_bucket"])  # the two lookups
    compare("what_ladder", target, fold, "c_history_volume", "b_history", score_what_lookup_baseline(test, fit_c).to_numpy(), score_what_lookup_baseline(test, fit_b).to_numpy())
    (lr_fit, lr_score), (nb_fit, nb_score) = make_what_logistic_regression(features), make_what_naive_bayes(features)  # untuned pairs
    compare("what_lr_vs_nb", target, fold, "logistic_regression", "naive_bayes", lr_score(test, lr_fit(train)), nb_score(test, nb_fit(train)))

# WHEN next_1y, fold 5: same folds as the tuning and the comparison.
target = "next_1y"  # WHEN target
snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached snapshots
development = split_holdout(snapshots)["development"]  # the holdout stays sealed
low_volume_years = year_distribution(snapshots).query("low_volume").index  # years no test tranche may touch
fold = make_folds(development, target=target, low_volume_years=low_volume_years)[4]  # fold 5
train, test = fold["train"], fold["test"]  # fold rows
lr_fit, lr_score = make_logistic_regression(target)  # reference of the comparison
lr_scores = lr_score(test, lr_fit(train))  # Logistic Regression scores, fitted once
for name, maker in {"random_forest": make_random_forest, "hist_gradient_boosting": make_hist_gradient_boosting, "xgboost_stage0": make_xgboost}.items():
    fit_fn, score_fn = maker(target)  # untuned pair
    compare("when_next_1y_trees_vs_lr", target, fold, name, "logistic_regression", score_fn(test, fit_fn(train)), lr_scores)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # new folder
pd.DataFrame(rows).to_csv(OUT_PATH, index=False)  # new file
print("written", OUT_PATH.relative_to(ROOT))
