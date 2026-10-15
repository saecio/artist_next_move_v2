"""One-time final evaluation of the WHAT XGBoost stage 0 on the sealed holdout (2018-2023), with baseline (c) as the reference.

Stage 0 is fitted once on the training rows (year < 2018 and next_active_year < 2018) with the parameters of the log row stage0,
and scored once on the holdout rows (2018 <= year <= 2023 and next_active_year <= 2023), as fixed in the entries "WHAT holdout label rule",
"WHAT XGBoost adoption evaluation: stage 0 kept" and "WHAT threshold-metric report" of docs/decisions.md. Nothing is tuned or chosen after
seeing the results. Everything is computed in memory first; the files are written with exclusive creation into results/holdout_what_tmp/ and the
folder is renamed to results/holdout_what/ only after every file is written, so nothing is left under results/holdout_what/ if the script fails.
The script refuses to start if either folder already has files, so the holdout is evaluated once.

Cutoff rule of every threshold report: a row is flagged as "new style" when score >= threshold.

Usage (stdout is the .out file with the printed tables):
    venv/bin/python3 scripts/run_xgb_what_holdout.py > results/xgb_what_holdout_<timestamp>.out
"""

import json  # result files
import os  # atomic rename of the output folder
import sys  # module search path and exit
import warnings  # silence the sklearn warning of the internal report of evaluate_holdout
from pathlib import Path  # project root

import numpy as np  # scores and labels
import pandas as pd  # dataframe library used to read the Works table and build the tables
from sklearn.exceptions import UndefinedMetricWarning  # raised when a class has no predicted row
from sklearn.metrics import precision_recall_fscore_support  # per-class precision, recall, F1 and support

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import evaluate_holdout  # the one-time final evaluation, PR-AUC and ROC-AUC with artist-clustered intervals
from baselines import KILL_GATE_CONFIDENCE, N_BOOTSTRAP, add_what_buckets, fit_what_lookup_baseline, score_what_lookup_baseline  # baseline (c) and the bootstrap settings
from models import make_what_xgboost  # WHAT XGBoost pair, accepts overrides
from tuning import read_log  # trial log reader
from what_target import build_artist_years, build_what_features, build_what_table  # WHAT table and inputs

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)  # baseline (c) may flag no row at 0.5, handled below with nan

TARGET = "new_style_year"  # WHAT target
LOG_PATH = ROOT / "results/trials/xgb_what_new_style_year.jsonl"  # trial log, read only here
OUT_DIR = ROOT / "results/holdout_what"  # final folder, appears only through the rename at the end
TMP_DIR = ROOT / "results/holdout_what_tmp"  # files are written here first
HOLDOUT_START, HOLDOUT_END = 2018, 2023  # holdout years of the entry "WHAT holdout label rule"
EXPECTED_TRAIN_ROWS = 687_628  # training rows of the holdout label rule
EXPECTED_LABELLED_HOLDOUT = 190_846  # labelled rows with 2018 <= year <= 2023
EXPECTED_HOLDOUT_ROWS = 143_228  # holdout rows with next_active_year <= 2023
EXPECTED_DROPPED = 47_618  # labelled holdout rows dropped by the label rule (next_active_year after 2023)
EXPECTED_P = 0.3067  # training positive rate, entry "WHAT XGBoost scale_pos_weight rate"
THRESHOLDS = [0.5, 0.3067]  # fixed in the entry "WHAT threshold-metric report", the second is the training positive rate
TOP_SHARE = 0.10  # share of holdout rows with the highest score for the ranking figure
SEARCHED_KEYS = {"max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "learning_rate", "n_estimators"}  # keys of a trial of the log
CLASS_NAMES = {0: "no new style", 1: "new style"}  # class labels of the tables

for folder in (OUT_DIR, TMP_DIR):  # the holdout is a one-time number
    if folder.exists() and any(folder.iterdir()):
        sys.exit(f"{folder.relative_to(ROOT)} already has files, the sealed holdout is evaluated once")
assert N_BOOTSTRAP == 1000 and KILL_GATE_CONFIDENCE == 0.95, "the bootstrap settings are not the logged ones"  # 1,000 draws and 95%
print("cutoff rule of every threshold report: a row is flagged as new style when score >= threshold (the expression of evaluate_holdout)", flush=True)

# Stage 0 parameters, read from the log by trial id and not typed in.
params = {r["trial"]: r for r in read_log(LOG_PATH)}["stage0"]["params"]  # parameters of the stage0 row
assert set(params) == SEARCHED_KEYS, f"unexpected keys {sorted(set(params) ^ SEARCHED_KEYS)}"  # exactly the eight searched keys, no scale_pos_weight
print(f"stage 0 params (log row stage0): {params}", flush=True)  # printed before any fit

# Data: the same table and features as the gate, then the two row sets of the holdout label rule.
works = pd.read_parquet(ROOT / "data/processed/works_checkpoint.parquet")  # cleaned Works table
table = build_what_table(build_artist_years(works))  # labelled artist-year rows, history >= 3, all years
features = build_what_features(works)  # the 23 columns of build_what_features; the 2 table inputs complete the 25 WHAT_FEATURES
train = table[(table["year"] < HOLDOUT_START) & (table["next_active_year"] < HOLDOUT_START)]  # label resolved before the holdout starts
labelled_holdout = table[(table["year"] >= HOLDOUT_START) & (table["year"] <= HOLDOUT_END)]  # every labelled row of the holdout years
holdout = labelled_holdout[labelled_holdout["next_active_year"] <= HOLDOUT_END]  # labels resolved within the loaded years
dropped = len(labelled_holdout) - len(holdout)  # rows dropped by the label rule
print(f"train rows {len(train)}, labelled holdout rows {len(labelled_holdout)}, holdout rows {len(holdout)}, dropped by the label rule {dropped}", flush=True)
assert len(train) == EXPECTED_TRAIN_ROWS, f"train rows {len(train)}"  # logged count
assert len(labelled_holdout) == EXPECTED_LABELLED_HOLDOUT, f"labelled holdout rows {len(labelled_holdout)}"  # logged count
assert len(holdout) == EXPECTED_HOLDOUT_ROWS, f"holdout rows {len(holdout)}"  # logged count
assert dropped == EXPECTED_DROPPED, f"dropped rows {dropped}"  # logged count
assert train["year"].max() < HOLDOUT_START and train["next_active_year"].max() < HOLDOUT_START, "a train row reaches 2018 or later"  # no train year >= 2018
assert holdout["year"].min() >= HOLDOUT_START and holdout["next_active_year"].max() <= HOLDOUT_END, "a holdout row is out of range"  # no holdout year < 2018
p = float(train[TARGET].mean())  # training positive rate
print(f"p = {p:.6f} ({int(train[TARGET].sum())} positives of {len(train)} training rows), thresholds {THRESHOLDS}", flush=True)
assert round(p, 4) == EXPECTED_P, f"p differs from {EXPECTED_P}"  # before any fit


def checked(fit_fn, logged_params: dict):
    """Wrap a fit function: after the single fit, print get_params and stop if any logged value is not in the model."""
    def wrapper(rows):
        model = fit_fn(rows)  # fitted on the training rows only
        got = model.get_params()  # parameters the model really has
        print(f"stage 0 get_params after the fit: {got}", flush=True)
        wrong = {k: (v, got.get(k)) for k, v in logged_params.items() if got.get(k) != v}  # logged value, value in the model
        if wrong:
            sys.exit(f"the model does not carry the logged parameters {wrong}, nothing was written")
        return model

    return wrapper


# Fit once and score once, stage 0 and baseline (c), with PR-AUC and ROC-AUC and their 95% artist-clustered intervals.
fit_fn, score_fn = make_what_xgboost(features, target=TARGET, **params)  # stage 0 pair with the logged parameters
results = {"stage 0": evaluate_holdout(train, holdout, TARGET, checked(fit_fn, params), score_fn, n_bootstrap=N_BOOTSTRAP, confidence=KILL_GATE_CONFIDENCE, seed=1, return_scores=True)}
train_b, holdout_b = add_what_buckets(train), add_what_buckets(holdout)  # history and volume buckets, columns known at the cutoff
baseline_fit = lambda rows: fit_what_lookup_baseline(rows, ["history_bucket", "volume_bucket"])  # as in run_what_gate
results["baseline (c)"] = evaluate_holdout(train_b, holdout_b, TARGET, baseline_fit, score_what_lookup_baseline, n_bootstrap=N_BOOTSTRAP, confidence=KILL_GATE_CONFIDENCE, seed=1, return_scores=True)
for name, result in results.items():
    assert len(result["scores"]) == len(holdout) == EXPECTED_HOLDOUT_ROWS, f"{name}: scores do not cover the holdout rows"  # one score per holdout row
    assert result["dropped_unresolved"] == 0, f"{name}: unresolved rows in the holdout"  # every holdout row has a label


def threshold_report(y: np.ndarray, scores: np.ndarray, threshold: float) -> tuple:
    """Accuracy summary and per-class precision, recall, F1 and support at one threshold. Undefined values stay nan."""
    flagged = (scores >= threshold).astype(int)  # cutoff rule: score >= threshold, the expression of evaluate_holdout
    precision, recall, f1, support = precision_recall_fscore_support(y, flagged, labels=[0, 1], zero_division=np.nan)  # nan where a class has no predicted or no true row
    f1 = np.where(np.isnan(precision) | np.isnan(recall), np.nan, f1)  # F1 is undefined when precision or recall is
    summary = {"threshold": threshold, "flagged_share": float(flagged.mean()), "accuracy": float((flagged == y).mean()),
               "majority_class_accuracy": float((y == 0).mean()), "holdout_positive_rate": float(y.mean())}  # references: always no new style, no-skill precision
    by_class = [{"threshold": threshold, "class": CLASS_NAMES[c], "precision": precision[c], "recall": recall[c], "f1": f1[c], "support": int(support[c])} for c in (0, 1)]
    return summary, by_class


summary_rows, class_rows, top_rows = [], [], []  # one entry per model and threshold, per class, and per model
for name, result in results.items():
    y, scores = result["labels"].astype(int), result["scores"]  # resolved labels and scores, same row order
    for threshold in THRESHOLDS:
        summary, by_class = threshold_report(y, scores, threshold)  # report at this threshold
        summary_rows.append({"model": name, **summary})
        class_rows += [{"model": name, **row} for row in by_class]
    # Cutoff check: the report at 0.5 built here equals the internal report of evaluate_holdout (same expression, score >= threshold).
    mine, internal = threshold_report(y, scores, 0.5), result["classification_report"]
    assert np.isclose(mine[0]["accuracy"], internal["accuracy"]) and np.isclose(mine[1][1]["recall"], internal["new work"]["recall"]), f"{name}: the cutoff does not match evaluate_holdout"
    # Ranking figure without a threshold: precision among the rows with a score at or above the 90th percentile of the holdout scores.
    cut = float(np.quantile(scores, 1 - TOP_SHARE))  # 90th percentile of the holdout scores
    top = scores >= cut  # selected rows, ties included
    top_rows.append({"model": name, "score_cutoff": cut, "selected_share": float(top.mean()), "selected_rows": int(top.sum()), "precision_top": float(y[top].mean()),
                     "holdout_positive_rate": float(y.mean()), "lift": float(y[top].mean() / y.mean())})
print("cutoff check passed: the report at 0.5 equals the internal report of evaluate_holdout (accuracy and recall of the new style class)", flush=True)

# Tables, printed as the .out file.
pd.set_option("display.width", 250)  # readable printed tables
pd.set_option("display.float_format", lambda v: f"{v:.4f}")  # four decimals
headline = pd.DataFrame([{"model": n, "rows": r["test_rows"], "artists": r["test_artists"], "positive_rate": r["positive_rate"], "pr_auc": r["pr_auc"], "pr_auc_lo": r["pr_auc_lo"], "pr_auc_hi": r["pr_auc_hi"],
                          "roc_auc": r["roc_auc"], "roc_auc_lo": r["roc_auc_lo"], "roc_auc_hi": r["roc_auc_hi"]} for n, r in results.items()])
print("\nHOLDOUT, pooled, PR-AUC and ROC-AUC with 95% artist-clustered intervals (1,000 draws, seed 1)\n" + headline.to_string(index=False), flush=True)
by_year = pd.DataFrame([{"model": n, "year": y, **v} for n, r in results.items() for y, v in r["by_year"].items()])
print("\nBY YEAR (descriptive)\n" + by_year.to_string(index=False), flush=True)
summary_table = pd.DataFrame(summary_rows)
print("\nTHRESHOLD REPORT, accuracy with its references (majority class = always no new style; holdout positive rate = precision of a score without skill)\n" + summary_table.to_string(index=False), flush=True)
class_table = pd.DataFrame(class_rows)  # undefined precision, recall or F1 stay nan
print("\nTHRESHOLD REPORT, per class (undefined = the class has no predicted row)\n" + class_table.to_string(index=False, na_rep="undefined"), flush=True)
top_table = pd.DataFrame(top_rows)
print("\nTOP 10% BY SCORE (rows with a score at or above the 90th percentile; the selected share shows the effect of ties)\n" + top_table.to_string(index=False), flush=True)

# One row per holdout row with the scores of both models, same row order (asserted).
assert np.array_equal(results["stage 0"]["labels"], results["baseline (c)"]["labels"]), "the two models do not share the row order"  # same labels in the same order
assert np.array_equal(results["stage 0"]["labels"], holdout[TARGET].to_numpy()), "the scored rows are not the holdout rows in order"  # rows are the holdout rows, in order
row_scores = pd.DataFrame({"artist_id": holdout["artist_id"].to_numpy(), "year": holdout["year"].to_numpy(), "label": results["stage 0"]["labels"].astype(int),
                           "stage0_score": results["stage 0"]["scores"], "baseline_c_score": results["baseline (c)"]["scores"]})  # one row per holdout row
assert len(row_scores) == EXPECTED_HOLDOUT_ROWS, "row_scores does not have one row per holdout row"  # 143,228 rows

# Files: written with exclusive creation into the temporary folder, which is renamed to the final folder only after every file is written.
# Scores and labels are not stored in the JSON files; the internal report of evaluate_holdout is replaced by the threshold tables.
TMP_DIR.mkdir(parents=True, exist_ok=False)  # fails if the folder exists, even empty
for name, result in results.items():
    stored = {k: v for k, v in result.items() if k not in ("scores", "labels", "classification_report", "threshold")}  # plain JSON values only
    with open(TMP_DIR / f"{'stage0' if name == 'stage 0' else 'baseline_c'}_{TARGET}.json", "x") as handle:  # exclusive creation
        json.dump(stored, handle, indent=1)
for file_name, frame in {"headline.csv": headline, "by_year.csv": by_year, "threshold_summary.csv": summary_table, "threshold_by_class.csv": class_table, "top10_lift.csv": top_table, "row_scores.csv": row_scores}.items():
    frame.to_csv(TMP_DIR / file_name, mode="x", index=False)  # exclusive creation
print("\nfiles written to the temporary folder:", sorted(path.name for path in TMP_DIR.iterdir()), flush=True)
os.rename(TMP_DIR, OUT_DIR)  # atomic on the same file system, the final folder appears complete or not at all
print("renamed to results/holdout_what; files:", sorted(path.name for path in OUT_DIR.iterdir()), flush=True)
