"""Test of the live data layer: live values against the stored ones for all demo artists. Prints only, writes no files.

Usage:
    venv/bin/python3 scripts/run_app_live_test.py
"""

import sys  # module path
import time  # timings
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

import pandas as pd  # tables

from app_live import artist_timeline, load_models, load_tables, score_what, score_when, style_live  # layer under test
from live_style_assets import FIRST, LAST, PROJECTION, TOLERANCE, VALUE_COLUMNS  # style years and limits
from when_demo import TARGETS, demo_artist_ids  # demo artist ids

ids = demo_artist_ids()  # the 40 demo artists


def report(name: str, merged: pd.DataFrame, live_col: str, stored_col: str) -> None:  # print one comparison
    """Print matched rows, unmatched rows and the maximum absolute difference of one comparison."""
    matched = merged[merged["_merge"] == "both"]  # rows on both sides
    diff = (matched[live_col] - matched[stored_col]).abs().max()  # largest difference
    print(f"{name}: matched {len(matched)}, unmatched {len(merged) - len(matched)}, max abs diff {diff:.3g}, within {TOLERANCE}: {diff <= TOLERANCE}")  # print the result


def timed(name: str, fn, *args):  # time the first and the repeated call
    """Run fn twice and print the first and the repeated call time."""
    t0 = time.time()  # first call
    out = fn(*args)  # result
    t1 = time.time()  # between calls
    fn(*args)  # repeated call
    print(f"time {name}: first {t1 - t0:.3f} s, repeated {time.time() - t1:.3f} s")  # print the result
    return out  # first result


for loader in (load_models, load_tables):  # the two cached loaders
    t0 = time.time()  # first call reads the files
    loader()  # load
    t1 = time.time()  # between calls
    loader()  # cached call
    print(f"time {loader.__name__}: first {t1 - t0:.3f} s, cached {time.time() - t1:.3f} s")  # print the result
tables = load_tables()  # stored tables, cached

# WHEN against the stored scores
when = pd.concat([score_when(i) for i in ids], ignore_index=True)  # live scores, 40 artists
long = when.melt(id_vars=["artist_id", "snapshot_year", "work_id"], value_vars=TARGETS, var_name="target", value_name="live")  # one row per target
stored = tables["when_stored"]  # stored scores of both targets
both = long.merge(stored, on=["artist_id", "snapshot_year", "work_id", "target"], how="outer", indicator=True)  # live vs stored
report("score_when", both, "live", "score")  # run and show the comparison

# WHAT against the stored scores
what = pd.concat([score_what(i) for i in ids], ignore_index=True)  # live scores, 40 artists
both = what.merge(tables["what_stored"][["artist_id", "year", "stage0_score"]], on=["artist_id", "year"], how="outer", indicator=True, suffixes=("_live", "_stored"))  # live vs stored, stored rows are the 160 holdout rows
report("score_what, stored holdout rows", both[both["_merge"] != "left_only"], "stage0_score_live", "stage0_score_stored")  # live rows without a stored row are the new ones
new = what[~what.set_index(["artist_id", "year"]).index.isin(tables["what_stored"].set_index(["artist_id", "year"]).index)]  # rows beyond the stored holdout
print(f"score_what, new rows: {len(new)}, years {sorted(new['year'].unique().tolist())}, scores in (0, 1): {bool(new['stage0_score'].between(0, 1).all())}, flagged labels {int(new['label_after_holdout'].fillna(False).sum())}, labels present {int(new['label'].notna().sum())}")  # new rows summary
print(f"score_what, total rows {len(what)}, duplicated keys {int(what.duplicated(['artist_id', 'year']).sum())}")  # 299 rows expected, no duplicates

# style for every year
live_style = pd.concat([style_live(y) for y in list(range(FIRST, LAST + 1)) + [PROJECTION]], ignore_index=True)  # all target years
for method in ["random_forest", "repeat_last"]:  # both methods
    for col in VALUE_COLUMNS:  # compared columns
        a = live_style[live_style["method"] == method]  # live rows of the method
        b = tables["predictions"][tables["predictions"]["method"] == method]  # stored rows of the method
        both = a.merge(b, on=["style", "target_year"], how="outer", indicator=True, suffixes=("_live", "_stored"))  # live vs stored
        report(f"style_live {method} {col}", both, f"{col}_live", f"{col}_stored")  # run and show the comparison

# timings, first and cached call
timed("score_when", score_when, ids[0])  # time this function
timed("score_what", score_what, ids[0])  # time this function
timed("artist_timeline", artist_timeline, ids[0])  # time this function
timed("style_live", style_live, FIRST)  # time this function

# empty states
pd.set_option("display.width", 250)  # readable tables
pd.set_option("display.max_columns", 40)  # all columns
show = ["year", "title", "is_rerecording", "next_1y", "next_2y", "label_next_1y", "is_projection_next_1y", "stage0_score", "what_label", "baseline_c_score", "artist_train_rows"]  # printed columns
zero = artist_timeline(7592689)  # artist with 0 train rows in WHEN
print("\n7592689 timeline (artist_train_rows", zero["artist_train_rows"].iloc[0], ")")  # print the result
print(zero[show].to_string(index=False))  # print the result
for i in ids:  # first artist with a Work in a WHAT year but no WHAT row
    t = artist_timeline(i)  # timeline
    gap = t[t["year"].between(2018, 2022) & t["work_id"].notna() & t["stage0_score"].isna()]  # Work without WHAT
    if len(gap):  # found
        print(f"\n{i} has Works in {gap['year'].tolist()} but no WHAT row")  # print the result
        print(t[show].to_string(index=False))  # print the result
        print("\nempty cells per column:")  # print the result
        print(t.isna().sum().to_string())  # print the result
        break  # stop after the first artist

# WHAT coverage after the holdout
for i in (1124645, 7592689, 2929902):  # artist with all years, artist with gaps, artist without any eligible year
    t = artist_timeline(i)  # timeline
    w = t[t["year"] >= 2023][["year", "title", "stage0_score", "what_label", "what_label_after_holdout"]]  # WHAT columns from 2023
    print(f"\n{i} WHAT from 2023 (rows with a score: {int(w['stage0_score'].notna().sum())})")  # artist header
    print(w.assign(title=w["title"].str[:30]).to_string(index=False))  # table
