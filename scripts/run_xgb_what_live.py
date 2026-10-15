"""Live curiosity demo for WHAT: stage 0 score of the last active year of a fixed list of well-known artists.

Stage 0 (parameters of the log row stage0) is fitted once on the holdout training rows (year < 2018 and next_active_year < 2018),
as in scripts/run_xgb_what_holdout.py, and scores the last active year row of each artist. Those rows are built directly from the
artist-year table, because the WHAT table drops the rows whose label is unknown. No label exists for them. The saved holdout score
of the artist's last holdout year is read from results/holdout_what/row_scores.csv as a cross-check; no holdout score is recomputed.
This is an illustration: it changes no model, metric or decision. The output file is created once and never overwritten.

Usage:
    venv/bin/python3 scripts/run_xgb_what_live.py
"""

import sys  # module search path and exit
from pathlib import Path  # project root

import numpy as np  # score comparison
import pandas as pd  # dataframe library used to read the Works table and build the report

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from models import make_what_xgboost  # WHAT XGBoost pair, accepts overrides
from tuning import read_log  # trial log reader
from what_demo import LIVE_ARTISTS, last_active_rows, last_holdout_scores  # fixed list and readers of the demo
from what_target import build_artist_years, build_what_features, build_what_table  # WHAT table and inputs

TARGET = "new_style_year"  # WHAT target
LOG_PATH = ROOT / "results/trials/xgb_what_new_style_year.jsonl"  # trial log, read only here
HOLDOUT_DIR = ROOT / "results/holdout_what"  # saved holdout files, read only here
OUT_PATH = ROOT / "results/demo/what_live_scores.csv"  # new file, never overwritten
EXPECTED_TRAIN_ROWS = 687_628  # training rows of the holdout label rule
EXPECTED_P = 0.3067  # training positive rate, entry "WHAT XGBoost scale_pos_weight rate"
SEARCHED_KEYS = {"max_depth", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "learning_rate", "n_estimators"}  # keys of a trial of the log

if OUT_PATH.exists():  # never overwrite
    sys.exit(f"{OUT_PATH.name} already exists, nothing was run")  # refuse to start, the file is created once

# Stage 0 parameters, read from the log by trial id and not typed in.
params = {str(r["trial"]): r for r in read_log(LOG_PATH)}["stage0"]["params"]  # parameters of the stage0 row
assert set(params) == SEARCHED_KEYS, f"unexpected keys {sorted(set(params) ^ SEARCHED_KEYS)}"  # exactly the eight searched keys
print(f"stage 0 params (log row stage0): {params}", flush=True)  # printed before any fit

# Data: the artist-year table (with unlabelled rows), the WHAT table and the features.
works = pd.read_parquet(ROOT / "data/processed/works_checkpoint.parquet")  # cleaned Works table
dated = works[works["year"].notna()]  # dated Works only
per_year = dated[(dated["year"] >= 2018) & (dated["year"] <= 2026)].groupby("year").size()  # Works per year, loading delay and a partial 2026
print("dated Works per year (2018-2026; recent years are still being loaded and 2026 is partial):", {int(y): int(n) for y, n in per_year.items()}, flush=True)  # loading delay shows in the last years
artist_years = build_artist_years(works)  # one row per artist-year, label missing where no next active year
table = build_what_table(artist_years)  # labelled rows with history >= 3
features = build_what_features(works)  # the 23 columns of build_what_features; the 2 table inputs complete the 25 WHAT_FEATURES
train = table[(table["year"] < 2018) & (table["next_active_year"] < 2018)]  # label resolved before the holdout starts
assert len(train) == EXPECTED_TRAIN_ROWS, f"train rows {len(train)}"  # logged count
p = float(train[TARGET].mean())  # training positive rate
assert round(p, 4) == EXPECTED_P, f"p differs from {EXPECTED_P}"  # logged value
print(f"train rows {len(train)}, p = {p:.6f}", flush=True)  # sizes and rate of the training rows

# The last active year row of each fixed artist, built directly from the artist-year table.
live, missing = last_active_rows(artist_years, list(LIVE_ARTISTS))  # rows with history_count >= 3 and the artists without such a row
print(f"artists with a last-active-year row of history_count >= 3: {len(live)} of {len(LIVE_ARTISTS)}; without: {[(LIVE_ARTISTS[a], why) for a, why in missing]}", flush=True)  # the rows kept and the artists left out, with the reason
assert len(live) > 0, "no live row could be built"  # stop and report if the rows cannot be built
live["artist_id"] = live["artist_id"].astype("int64")  # integer ids
live["year"] = live["year"].astype("int64")  # integer years

# Fit once on the training rows, score the live rows with the same fitted model.
fit_fn, score_fn = make_what_xgboost(features, target=TARGET, **params)  # stage 0 pair with the logged parameters
model = fit_fn(train)  # fitted once, on training rows only
got = model.get_params()  # parameters the model really has
assert all(got.get(k) == v for k, v in params.items()), "the model does not carry the logged parameters"  # logged values are in the model
live["score"] = score_fn(live, model)  # probability that the next Work introduces a new style, relative score and not a forecast

# Report columns, the cross-check with the saved holdout scores and the training rate.
names = works[works["artist_id"].isin(live["artist_id"])].drop_duplicates("artist_id").set_index("artist_id")["artist"]  # one name per artist id
lookup = features.set_index(["artist_id", "year"])  # features by artist-year
keys = pd.MultiIndex.from_frame(live[["artist_id", "year"]])  # artist-years of the live rows
report = pd.DataFrame({  # table written to the CSV
    "artist_id": live["artist_id"], "artist": live["artist_id"].map(names), "last_active_year": live["year"],  # who and which row
    "history_count": live["history_count"].astype(int), "n_works_year": live["n_works_year"].astype(int),  # table inputs of the row
    "n_styles_seen": lookup.reindex(keys)["n_styles_seen"].to_numpy(), "years_since_new_style": lookup.reindex(keys)["years_since_new_style"].to_numpy(),  # two readable inputs
    "score": live["score"],  # stage 0 score of the row
})
report = report.merge(last_holdout_scores(HOLDOUT_DIR, list(live["artist_id"])), on="artist_id", how="left")  # saved holdout cross-check, empty if no holdout row
report["training_rate_p"] = p  # reference point saved with the table
report["last_holdout_year"] = report["last_holdout_year"].astype("Int64")  # integer year, empty when missing
same = report[report["last_active_year"] == report["last_holdout_year"]]  # rows that are also a holdout row, scored by the same fit
print(f"rows that are also a saved holdout row: {len(same)}; largest absolute difference between the live score and the saved score: "  # consistency with the saved holdout scores
      f"{float(np.abs(same['score'] - same['last_holdout_score']).max()) if len(same) else float('nan'):.2e}", flush=True)  # same model, same row
pd.set_option("display.width", 250)  # readable printed table
print(report.sort_values("score", ascending=False).round(4).to_string(index=False), flush=True)  # printed table, highest score first
report.to_csv(OUT_PATH, mode="x", index=False)  # exclusive creation, fails if the file exists
print(f"written {OUT_PATH.name}", flush=True)  # confirmation of the single write
