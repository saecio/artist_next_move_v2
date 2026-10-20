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
stored = tables["when_stored"][tables["when_stored"]["artist_id"].isin(ids)]  # stored scores of both targets, pool A only
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

# pool B against its stored files
from app_data import load_demo_config  # demo settings
cfgp = load_demo_config()  # settings
ids_b = [int(a["discogs_id"]) for a in cfgp["artists_b"]]  # the 21 pool B ids
pbp = cfgp["pool_b"]["paths"]  # pool B file paths
print("\n== pool B")  # section title
when_b = pd.concat([score_when(i) for i in ids_b], ignore_index=True)  # live WHEN scores, 21 artists
long_b = when_b.melt(id_vars=["artist_id", "snapshot_year", "work_id"], value_vars=TARGETS, var_name="target", value_name="live")  # one row per target
file_b = pd.read_csv(ROOT / pbp["when_scores"])  # stored pool B WHEN scores
both = long_b.merge(file_b, on=["artist_id", "snapshot_year", "work_id", "target"], how="outer", indicator=True)  # live vs file
report("score_when pool B vs when_demo_scores_b.csv", both, "live", "score")  # run and show the comparison
sb = tables["when_stored"][tables["when_stored"]["artist_id"].isin(ids_b)].reset_index(drop=True)  # pool B part of the stored table
same_file = bool(sb.fillna(-1).equals(file_b.fillna(-1)))  # labels, flags and artist_train_rows come from the file unchanged
print(f"pool B stored rows {len(file_b)}, stored table equals the file (labels, flags, artist_train_rows): {same_file}")  # row count of the file
assert same_file  # the app reads the file as it is
what_b = pd.concat([score_what(i) for i in ids_b], ignore_index=True)  # live WHAT scores, 21 artists
feat_b = pd.read_csv(ROOT / pbp["what_features"])  # pool B WHAT feature rows
lab_b = pd.read_csv(ROOT / pbp["what_labels"])  # pool B WHAT labels
keys_b = set(map(tuple, feat_b[["artist_id", "year"]].to_numpy()))  # rows with a WHAT score by file
got = set(map(tuple, what_b[["artist_id", "year"]].to_numpy()))  # rows with a WHAT score by function
m = what_b.merge(lab_b, on=["artist_id", "year"], suffixes=("", "_file"))  # labels from the file
same = bool(((m["label"].fillna(-1) == m["label_file"].fillna(-1)) & (m["label_after_holdout"] == m["label_after_holdout_file"])).all())  # label and flag equal
print(f"score_what pool B: rows {len(what_b)} (file {len(feat_b)}), key sets equal {keys_b == got}, labels and flags equal {same}, baseline_c_score all empty {bool(what_b['baseline_c_score'].isna().all())}, scores in (0, 1) {bool(what_b['stage0_score'].between(0, 1).all())}")  # WHAT check
assert len(file_b) == both["_merge"].eq("both").sum() and keys_b == got and same and what_b["baseline_c_score"].isna().all()  # all pool B checks
# pool A timeline unchanged by the concatenation: a few fixed facts of the earlier runs
t1 = artist_timeline(1124645).set_index("year")  # Taylor Swift timeline
t2 = artist_timeline(7592689).set_index("year")  # Olivia Rodrigo timeline
print(f"pool A unchanged: 1124645 2020 WHEN {t1.loc[2020, 'next_1y']:.3f} WHAT {t1.loc[2020, 'stage0_score']:.3f}; 7592689 2023 WHEN {t2.loc[2023, 'next_1y']:.3f} WHAT {t2.loc[2023, 'stage0_score']:.3f}")  # earlier values 0.977, 0.445, 0.937, 0.453
assert (round(t1.loc[2020, "next_1y"], 3), round(t1.loc[2020, "stage0_score"], 3), round(t2.loc[2023, "next_1y"], 3), round(t2.loc[2023, "stage0_score"], 3)) == (0.977, 0.445, 0.937, 0.453)  # same values as the earlier runs

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

print("\n== release titles of the reality cells (results/demo/release_pick.csv through load_tables)")  # section title
picks = load_tables()["release_pick"]  # cached pick table
works = load_tables()["works"]  # Works of both pools
assert list(picks.columns) == ["artist_id", "year", "work_id", "title", "is_album_like", "picked_by"], "pick file columns"  # file layout
assert not picks.duplicated(["artist_id", "year"]).any() and picks["title"].notna().all(), "one title per artist-year"  # keys and titles
assert set(picks["picked_by"]) == {"album", "fallback"} and picks[picks["picked_by"] == "album"]["is_album_like"].all() and not picks[picks["picked_by"] == "fallback"]["is_album_like"].any(), "album picks are album-like, fallback picks are not"  # rule B flags
assert picks["work_id"].isin(works.loc[~works["is_rerecording"], "work_id"]).all(), "every pick is a non-re-recording Work of the demo files"  # membership
need = works[(works["year"] >= 2018) & ~works["is_rerecording"]][["artist_id", "year"]].drop_duplicates()  # artist-years that need a title
assert len(need.merge(picks, on=["artist_id", "year"])) == len(need), "every artist-year with a Work from 2018 on has a pick"  # coverage, 2024 and later included
print(f"picks {len(picks)}: {picks['picked_by'].value_counts().to_dict()}, years {int(picks['year'].min())} to {int(picks['year'].max())}, all {len(need)} artist-years with a Work covered")  # result

from app_live import when_rows  # WHEN rows of an artist-year, with the counting flags

n_c, n_s, n_n = 0, 0, 0  # counted rows, asterisk rows, rows without a reality
for aid in ids:  # every demo artist
    for year in range(2018, 2027):  # every year with a possible row
        w = when_rows(aid, year)  # rows of the artist-year
        if w is None:  # no row
            continue  # nothing to check
        for h in (w["next_1y"], w["next_2y"]):  # both horizons
            assert not (h["counted"] and h["star"]), "a row is never counted and starred"  # exclusive flags
            assert (h["label"] is None) == (not h["counted"] and not h["star"]), "no label means no flag, a label means counted or starred"  # flags follow the label
            n_c, n_s, n_n = n_c + h["counted"], n_s + h["star"], n_n + (h["label"] is None)  # tally
        assert w["next_1y"]["counted"] == (w["next_1y"]["label"] is not None and year + 1 <= 2025), f"{aid} {year}: next-year rule by reality year"  # reality year year + 1
        assert w["next_1y"]["star"] == (w["next_1y"]["label"] is not None and year + 1 >= 2026), f"{aid} {year}: next-year asterisk"  # reality in 2026
        if year == 2026:  # forecast year
            assert w["kind"] == "work" and w["next_1y"]["label"] is None and w["next_2y"]["label"] is None, f"{aid}: 2026 is a Work row without reality"  # no year-end row, no label
print(f"counting rule by reality year: {n_c} counted horizons, {n_s} with asterisk, {n_n} without reality, 2026 rows have no reality")  # result
