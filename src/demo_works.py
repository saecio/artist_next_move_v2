"""Works file of the demo artists: dated Works for the timeline cards, with re-recording flag and representative flag.

Design fixed in docs/decisions.md (entries "Demo: works file of the demo artists", "Demo: re-recordings" and
"Demo: timeline rule per artist and year, and baseline label").
"""

import json  # list columns stored as JSON text

import pandas as pd  # tables

from config import CONFIG, ROOT  # project settings and root
from when_demo import demo_artist_ids  # the 40 demo artist ids from app/demo_config.yaml

WORKS_PATH = ROOT / CONFIG["paths"]["works"]  # cleaned Works table
SNAPSHOTS_PATH = ROOT / CONFIG["paths"]["snapshots"]  # cached snapshots
FEATURES_PATH = ROOT / CONFIG["paths"]["when_live_features"]  # WHEN feature rows of the demo artists
OUT_PATH = ROOT / CONFIG["paths"]["works_demo"]  # output file, never overwritten
FIRST_SHOWN = CONFIG["years"]["demo_works_first_year"]  # first year in the file (display choice)
HOLDOUT_START = CONFIG["years"]["holdout_start"]  # from this year on every dated Work is kept
MARKER = "taylor's version"  # re-recording marker of the entry "Demo: re-recordings"
COLUMNS = ["work_id", "artist_id", "title", "year", "genres", "styles", "is_rerecording", "works_so_far_this_year", "works_this_year", "is_representative"]  # output order


def is_rerecording(titles: pd.Series) -> pd.Series:  # flag re-recording titles
    """True when the title holds the marker, matching the straight and the typographic apostrophe."""
    clean = titles.fillna("").str.replace("’", "'", regex=False).str.lower()  # typographic apostrophe to straight, lower case
    return clean.str.contains(MARKER, regex=False)  # plain substring match


def build_full(ids: list) -> pd.DataFrame:  # build the full table of dated Works
    """Dated Works of the demo artists from FIRST_SHOWN on, with the position columns, before the year cut of 2010 to 2017."""
    keep = [("artist_id", "in", [float(i) for i in ids])]  # artist filter pushed to the parquet reader
    works = pd.read_parquet(WORKS_PATH, columns=["work_id", "artist_id", "title", "year", "genres", "styles"], filters=keep)  # selected columns only
    snaps = pd.read_parquet(SNAPSHOTS_PATH, columns=["work_id", "works_so_far_this_year", "works_this_year"], filters=keep)  # position columns
    table = works.dropna(subset=["year"]).merge(snaps, on="work_id", how="inner", validate="one_to_one")  # dated Works only, one snapshot each
    table["artist_id"] = table["artist_id"].astype("int64")  # whole-number ids
    table["year"] = table["year"].astype("int64")  # whole-number years
    table = table[table["year"] >= FIRST_SHOWN].copy()  # nothing before the first shown year
    table["is_rerecording"] = is_rerecording(table["title"])  # re-recording flag
    table["is_representative"] = table["works_so_far_this_year"] == table["works_this_year"]  # last Work of the year in the snapshot order
    for col in ["genres", "styles"]:  # list columns to JSON text
        table[col] = table[col].map(lambda v: json.dumps(list(v) if v is not None else [], ensure_ascii=False))  # stable text form
    return table  # hand back the finished table


def cut_years(full: pd.DataFrame) -> pd.DataFrame:  # apply the year cut
    """Every Work from the holdout start on; only the representative Work for the earlier years."""
    keep = (full["year"] >= HOLDOUT_START) | full["is_representative"]  # rule of the entry
    return full[keep].sort_values(["artist_id", "year", "work_id"])[COLUMNS].reset_index(drop=True)  # fixed order and columns


def run_controls(full: pd.DataFrame, out: pd.DataFrame, ids: list, last_year: int) -> bool:  # run and print the four controls
    """Print the four controls and return True when all pass."""
    feats = pd.read_csv(FEATURES_PATH, usecols=["work_id", "works_so_far_this_year"])  # live WHEN features
    ok = {}  # control name -> passed
    missing = set(feats["work_id"]) - set(out["work_id"])  # control 1
    ok["1 when_features work_ids in file"] = not missing  # record whether this control passed
    print(f"control 1: {len(feats)} work_ids in when_features.csv, {len(missing)} missing from the file")  # show the value on screen
    reps = full.groupby(["artist_id", "year"])["is_representative"].sum()  # representatives per artist-year, all years from 2010
    bad = reps[reps != 1]  # control 2
    ok["2 one representative per artist-year"] = bad.empty  # record whether this control passed
    print(f"control 2: {len(reps)} artist-years, {len(bad)} without exactly one representative")  # show the value on screen
    merged = feats.merge(out[["work_id", "works_so_far_this_year"]], on="work_id", suffixes=("_f", "_o"))  # control 3a
    diff_f = int((merged["works_so_far_this_year_f"] != merged["works_so_far_this_year_o"]).sum())  # count rows where the two values differ
    snaps = pd.read_parquet(SNAPSHOTS_PATH, columns=["work_id", "works_this_year"], filters=[("work_id", "in", out["work_id"].tolist())])  # fresh read for the check
    merged2 = snaps.merge(out[["work_id", "works_this_year"]], on="work_id", suffixes=("_s", "_o"))  # join the snapshot values to the file on work_id
    diff_s = int((merged2["works_this_year_s"] != merged2["works_this_year_o"]).sum())  # count rows where the two values differ
    ok["3 position columns match"] = diff_f == 0 and diff_s == 0 and len(merged) == len(feats) and len(merged2) == len(out)  # record whether this control passed
    print(f"control 3: works_so_far_this_year vs when_features {len(merged)} shared, {diff_f} differ; works_this_year vs snapshots {len(merged2)} shared, {diff_s} differ")  # show the value on screen
    empty = int((out["title"].fillna("").str.strip() == "").sum())  # control 4
    out_of_range = int((~out["year"].between(FIRST_SHOWN, last_year)).sum())  # count years outside the allowed range
    absent = sorted(set(ids) - set(out["artist_id"]))  # demo artists that have no row in the file
    ok["4 titles, years"] = empty == 0 and out_of_range == 0  # record whether this control passed
    print(f"control 4: empty titles {empty}, years outside {FIRST_SHOWN} to {last_year} {out_of_range}, distinct artists {out['artist_id'].nunique()} of {len(ids)}, missing {absent}")  # show the value on screen
    for name, passed in ok.items():  # summary
        print(("PASS " if passed else "FAIL ") + name)  # show the value on screen
    return all(ok.values())  # True only when every control passed


def run() -> None:  # build, check and write the file
    """Build, check and write the file; refuses to run when it exists and writes nothing when a control fails."""
    if OUT_PATH.exists():  # never overwrite
        raise SystemExit(f"{OUT_PATH.relative_to(ROOT)} already exists, nothing was run")  # stop the run with a message
    ids = demo_artist_ids()  # demo artist ids
    full = build_full(ids)  # all dated Works from 2010
    last_year = int(full["year"].max())  # last year of the data
    print("last year of the data:", last_year)  # show the value on screen
    out = cut_years(full)  # year cut
    if not run_controls(full, out, ids, last_year):  # stop before writing
        raise SystemExit("a control failed, nothing was written")  # stop the run with a message
    with open(OUT_PATH, "x", encoding="utf-8", newline="") as f:  # exclusive creation
        out.to_csv(f, index=False)  # write the file
    per_artist = out.groupby("artist_id").size()  # rows per artist
    rr = out[out["is_rerecording"]]  # re-recording rows
    print(f"path {OUT_PATH.relative_to(ROOT)}, {OUT_PATH.stat().st_size / 1024:.1f} KB")  # show the value on screen
    print(f"rows {len(out)}: {FIRST_SHOWN} to {HOLDOUT_START - 1} {int((out['year'] < HOLDOUT_START).sum())}, {HOLDOUT_START} on {int((out['year'] >= HOLDOUT_START).sum())}")  # show the value on screen
    print(f"rows per artist: min {per_artist.min()}, median {per_artist.median()}, max {per_artist.max()}")  # show the value on screen
    print(f"is_rerecording rows {len(rr)}, artists {sorted(rr['artist_id'].unique().tolist())}")  # show the value on screen
