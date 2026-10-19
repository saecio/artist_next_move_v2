"""Extended WHAT live assets: features, stored labels and control for every eligible demo artist-year from the holdout start."""

import json  # column list and control rows
import time  # runtime of the step

import numpy as np  # NaN-aware comparisons
import pandas as pd  # tables
from xgboost import XGBClassifier  # the existing WHAT model, loaded and not refitted

from app_data import load_demo_config  # demo settings and file names
from config import CONFIG, ROOT  # project settings and root
from models import WHAT_FEATURES, _what_matrix  # model input columns and matrix builder
from what_target import build_artist_years, build_what_features  # same builders as the first live assets step
from when_demo import HOLDOUT_START, demo_artist_ids  # first holdout year and demo artist ids

LIVE = load_demo_config()["live"]  # live file names
FOLDER = ROOT / LIVE["folder"]  # live assets folder, already exists
OUT_FILES = {k: FOLDER / LIVE[k] for k in ("what_features", "what_labels", "what_control")}  # new files, never overwritten
TOLERANCE = 0.00005  # largest accepted score difference (4 decimals)
FEATURE_TOLERANCE = 1e-9  # largest accepted feature difference
HOLDOUT_END = CONFIG["years"]["holdout_end"]  # last holdout year
MIN_PREVIOUS = CONFIG["what"]["min_previous_works"]  # earlier dated Works required, same as the WHAT table
CUTS = [CONFIG["years"]["development_last"], 2022, 2023, 2024, 2025]  # years used by the point-in-time control


def refuse_if_written() -> None:  # stop when a new file exists
    """Stop when any of the new files exists."""
    for path in OUT_FILES.values():  # each new file
        if path.exists():  # never overwrite
            raise SystemExit(f"{path.relative_to(ROOT)} already exists, nothing was run")  # stop early


def compare(a: pd.DataFrame, b: pd.DataFrame, cols: list) -> tuple:  # compare two tables on the shared keys
    """Rows on both sides, rows on one side only, and the largest absolute difference over cols (NaN equal to NaN)."""
    a, b = a.assign(artist_id=a["artist_id"].astype("int64")), b.assign(artist_id=b["artist_id"].astype("int64"))  # whole-number ids
    m = a[["artist_id", "year"] + cols].merge(b[["artist_id", "year"] + cols], on=["artist_id", "year"], how="outer", suffixes=("_a", "_b"), indicator=True)  # same keys on both sides
    both = m[m["_merge"] == "both"]  # shared rows
    x, y = both[[c + "_a" for c in cols]].astype(float).to_numpy(), both[[c + "_b" for c in cols]].astype(float).to_numpy()  # float arrays
    worst = float("inf") if (np.isnan(x) != np.isnan(y)).any() else (float(np.nanmax(np.abs(x - y))) if len(both) else 0.0)  # inf when NaN cells differ
    return len(both), int((m["_merge"] != "both").sum()), worst  # counts and difference


def build_demo(works: pd.DataFrame, ids: list) -> tuple:  # build the demo-only tables
    """Artist-year table and features built from the Works of the demo artists only."""
    mine = works[works["artist_id"].isin([float(i) for i in ids])]  # demo artists' Works
    return mine, build_artist_years(mine), build_what_features(mine)  # Works, artist-years, features


def eligible(years: pd.DataFrame) -> pd.DataFrame:  # rows to score
    """Artist-years from the holdout start with enough earlier Works; the label is not required."""
    keep = (years["year"] >= HOLDOUT_START) & (years["history_count"] >= MIN_PREVIOUS)  # scoring rule
    out = years.loc[keep, ["artist_id", "year", "history_count", "n_works_year", "next_active_year", "new_style_year"]].copy()  # rows and label columns
    out["artist_id"] = out["artist_id"].astype("int64")  # whole-number ids
    return out.reset_index(drop=True)  # clean index


def label_table(rows: pd.DataFrame) -> pd.DataFrame:  # labels and after-holdout flags
    """Stored label and the after-holdout flag: the label exists but the row is not a holdout row."""
    out = rows[["artist_id", "year", "new_style_year", "next_active_year"]].rename(columns={"new_style_year": "label"})  # label columns
    out["label_after_holdout"] = out["label"].notna() & ((out["year"] > HOLDOUT_END) | (out["next_active_year"] > HOLDOUT_END))  # outside the holdout rule
    return out  # label table


def controls(works: pd.DataFrame, ids: list, rows: pd.DataFrame, feats: pd.DataFrame, labels: pd.DataFrame, live: pd.DataFrame, columns: list, matrix: pd.DataFrame, full_feats: pd.DataFrame) -> pd.DataFrame:  # run every control
    """Run every control and return the table; the caller prints it and stops when one fails."""
    out = []  # control rows
    cols = [c for c in feats.columns if c not in ("artist_id", "year")]  # model columns
    old = pd.read_csv(FOLDER / LIVE["what_features_control"])  # first feature file, the 160 holdout rows
    n, un, d = compare(old, feats, cols)  # old rows against the new file
    out.append({"control": "1a old rows in new file, features equal", "rows": n, "unmatched": len(old) - n, "max_abs_diff": d, "limit": FEATURE_TOLERANCE})  # feature control
    stored = pd.read_csv(ROOT / LIVE["what_scores"])  # stored WHAT scores
    stored["artist_id"] = stored["artist_id"].astype("int64")  # whole-number ids
    s = live.merge(stored, on=["artist_id", "year"], how="inner")  # live against stored scores
    out.append({"control": "1b live score vs stored stage0_score", "rows": len(s), "unmatched": len(old) - len(s), "max_abs_diff": float((s["score"] - s["stage0_score"]).abs().max()), "limit": TOLERANCE})  # score control
    lab = labels.merge(stored, on=["artist_id", "year"], how="inner", suffixes=("", "_stored"))  # stored labels of the holdout rows
    out.append({"control": "6 holdout labels equal stored, none flagged", "rows": len(lab), "unmatched": len(old) - len(lab), "max_abs_diff": float((lab["label"] - lab["label_stored"]).abs().max()) + float(lab["label_after_holdout"].sum()), "limit": 0.0})  # label control
    out.append({"control": "2 model columns equal columns.json", "rows": len(columns), "unmatched": int(columns != list(matrix.columns)), "max_abs_diff": 0.0, "limit": 0.0})  # column control
    for cut in CUTS:  # point-in-time: features of year <= cut must not change when later Works are removed
        cutw = works[works["year"].isna() | (works["year"] <= cut)]  # Works known at the end of the cut year
        g = build_what_features(cutw).merge(build_artist_years(cutw)[["artist_id", "year", "history_count", "n_works_year"]], on=["artist_id", "year"])  # features and table inputs from that data only
        g = g[g["year"] >= HOLDOUT_START]  # scored years
        full = feats[(feats["year"] <= cut)]  # same rows from the full build
        n, un, d = compare(full, g, cols)  # cut build against full build
        out.append({"control": f"3 point-in-time, data cut at {cut}", "rows": n, "unmatched": len(full) - n, "max_abs_diff": d, "limit": FEATURE_TOLERANCE})  # one row per cut
    dated = works.dropna(subset=["year"]).assign(artist_id=lambda w: w["artist_id"].astype("int64"))  # dated Works
    counts = dated.groupby(["artist_id", "year"]).size().rename("n").reset_index()  # Works per artist-year
    counts["history"] = counts.groupby("artist_id")["n"].cumsum() - counts["n"]  # dated Works strictly before the year
    want = counts[(counts["year"] >= HOLDOUT_START) & (counts["history"] >= MIN_PREVIOUS)]  # independent eligibility
    k1, k2 = set(map(tuple, want[["artist_id", "year"]].to_numpy())), set(map(tuple, rows[["artist_id", "year"]].to_numpy()))  # key sets
    out.append({"control": "4 row keys equal independent count", "rows": len(k2), "unmatched": len(k1 ^ k2), "max_abs_diff": 0.0, "limit": 0.0})  # row-count control
    ff = full_feats[full_feats["artist_id"].isin([float(i) for i in ids]) & (full_feats["year"] >= HOLDOUT_START)]  # demo rows of the full-works build
    n, un, d = compare(ff, feats, [c for c in cols if c in ff.columns])  # full build against demo-only build, feature columns only
    out.append({"control": "5 full-works build equals demo-only build", "rows": n, "unmatched": len(feats) - n, "max_abs_diff": d, "limit": FEATURE_TOLERANCE})  # cross-check control
    table = pd.DataFrame(out)  # control table
    table["passed"] = (table["max_abs_diff"] <= table["limit"]) & (table["unmatched"] == 0)  # pass rule
    return table  # all controls


def run() -> None:  # build, control and write
    """Build, control, then write the three new files with exclusive creation; nothing is written if a control fails."""
    start = time.time()  # clock of the step
    refuse_if_written()  # stop early
    ids = demo_artist_ids()  # the 40 demo artists
    works = pd.read_parquet(ROOT / CONFIG["paths"]["works"])  # cleaned Works table
    full_feats = build_what_features(works)  # full-works build for the cross-check
    mine, years, feats_all = build_demo(works, ids)  # demo-only builds
    rows = eligible(years)  # artist-years to score
    matrix = _what_matrix(rows[["artist_id", "year", "history_count", "n_works_year"]], feats_all.assign(artist_id=feats_all["artist_id"].astype("int64")), WHAT_FEATURES)  # model input columns
    columns = json.loads((FOLDER / LIVE["columns"]).read_text())["what_stage0"]  # ordered columns of the saved model
    model = XGBClassifier()  # empty model
    model.load_model(FOLDER / LIVE["what_model"])  # saved final model, no refit
    assert columns == list(matrix.columns), "model columns differ from the feature rows"  # same names and order
    live = rows[["artist_id", "year"]].assign(score=model.predict_proba(matrix[columns])[:, 1])  # live scores
    feats = pd.concat([rows[["artist_id", "year"]], matrix], axis=1)  # feature file rows
    labels = label_table(rows)  # stored labels and flags
    table = controls(mine, ids, rows, feats, labels, live, columns, matrix, full_feats)  # all controls
    print(table.to_string(index=False), flush=True)  # control output
    if not table["passed"].all():  # any failure stops the step
        raise SystemExit("a control failed, nothing was written")  # write nothing
    print("\nrows per year (eligible, labelled, flagged):")  # section title
    print(labels.groupby("year").agg(eligible=("year", "size"), labelled=("label", lambda s: int(s.notna().sum())), flagged=("label_after_holdout", "sum")).to_string())  # counts per year
    late = rows[rows["year"] >= HOLDOUT_END].groupby("artist_id")["year"].apply(lambda s: sorted(s.tolist()))  # eligible years from 2023 per artist
    print(f"\neligible years {HOLDOUT_END} to the last data year, per artist ({len(late)} of {len(ids)} artists):")  # section title
    for a in ids:  # every demo artist
        print(f"  {a}: {late.get(a, [])}")  # years or empty
    with open(OUT_FILES["what_features"], "x", encoding="utf-8", newline="") as handle:  # exclusive creation
        feats.to_csv(handle, index=False)  # feature rows
    with open(OUT_FILES["what_labels"], "x", encoding="utf-8", newline="") as handle:  # exclusive creation
        labels.to_csv(handle, index=False)  # labels and flags
    with open(OUT_FILES["what_control"], "x", encoding="utf-8", newline="") as handle:  # exclusive creation
        table.to_csv(handle, index=False)  # control table
    print("\nfiles:")  # section title
    for key, path in OUT_FILES.items():  # sizes from disk
        n = len(pd.read_csv(path))  # rows written
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KB  {n} rows")  # name, size, rows
    print(f"runtime {time.time() - start:.0f} s")  # runtime
