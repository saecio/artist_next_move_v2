"""Extended WHAT live assets: features, stored labels and control for every eligible demo artist-year from the holdout start."""

import hashlib  # SHA-256 of the existing files
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


def refuse_if_written(files: dict = OUT_FILES) -> None:  # stop when a new file exists
    """Stop when any of the new files exists."""
    for path in files.values():  # each new file
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


def run(ids: list | None = None, files: dict | None = None) -> None:  # build, control and write; ids and files select pool B
    """Build, control, then write the three new files with exclusive creation; nothing is written if a control fails."""
    start = time.time()  # clock of the step
    if ids is not None:  # another pool: own builders, controls and outputs
        run_pool(ids, files)  # pool B step
        return  # the pool A code below is not used
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


def independent_labels(works: pd.DataFrame, ids: list) -> pd.DataFrame:  # new_style_year from the raw Works, without the project builders
    """new_style_year per artist-year recomputed with plain sets: styles of undated Works count as seen, the label needs a next active year with style tags."""
    rows = []  # artist, year, label
    for i in ids:  # one artist at a time
        w = works[works["artist_id"] == float(i)]  # this artist's Works
        seen = set(s for lst in w[w["year"].isna()]["styles"] for s in (lst if lst is not None else []))  # styles of undated Works are already seen
        dated = w[w["year"].notna()]  # dated Works
        by_year = {int(y): set(s for lst in g["styles"] for s in (lst if lst is not None else [])) for y, g in dated.groupby("year")}  # styles per year
        ys = sorted(by_year)  # active years
        for k, y in enumerate(ys):  # each active year
            seen |= by_year[y]  # styles known at the end of the year
            nxt = by_year[ys[k + 1]] if k + 1 < len(ys) else set()  # styles of the next active year
            rows.append({"artist_id": i, "year": y, "label": float(len(nxt - seen) > 0) if nxt else np.nan})  # label, empty without next styles
    return pd.DataFrame(rows)  # independent labels


def assemble(works_all: pd.DataFrame, ids: list, model, columns: list) -> tuple:  # rows, features, labels and live scores of some ids
    """Eligible rows, feature file rows, label table and live scores for some artist ids, with the same builders as the pool A step."""
    mine, years, feats_all = build_demo(works_all, ids)  # artist tables
    rows = eligible(years)  # artist-years to score
    matrix = _what_matrix(rows[["artist_id", "year", "history_count", "n_works_year"]], feats_all.assign(artist_id=feats_all["artist_id"].astype("int64")), WHAT_FEATURES)  # model input columns
    assert columns == list(matrix.columns), "model columns differ from the feature rows"  # same names and order
    live = rows[["artist_id", "year"]].assign(score=model.predict_proba(matrix[columns])[:, 1])  # live scores
    return mine, rows, pd.concat([rows[["artist_id", "year"]], matrix], axis=1), label_table(rows), live, matrix  # all pieces


def hashes_of(folders: list, skip: set) -> dict:  # SHA-256 of existing files
    """SHA-256 of every file under the folders except the names in skip and the files in skip paths."""
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for f in folders for p in sorted(f.rglob("*")) if p.is_file() and p.name not in (".DS_Store", "live_scores.csv") and p.resolve() not in skip}  # path -> hash


def run_pool(ids: list, files: dict) -> None:  # WHAT assets of another pool
    """Score the artists of another pool with the saved model, control against pool A and independent counts, then write with exclusive creation."""
    start = time.time()  # clock of the step
    refuse_if_written(files)  # stop early
    skip = {p.resolve() for p in files.values()}  # files this step creates
    folders = [ROOT / "results/demo", FOLDER, ROOT / "results/holdout_what"]  # folders whose files must not change
    before = hashes_of(folders, skip)  # hashes before the run
    works = pd.read_parquet(ROOT / CONFIG["paths"]["works"])  # all Works
    columns = json.loads((FOLDER / LIVE["columns"]).read_text())["what_stage0"]  # ordered columns of the saved model
    model = XGBClassifier()  # empty model
    model.load_model(FOLDER / LIVE["what_model"])  # saved final model, no refit
    ids_a = demo_artist_ids()  # pool A ids
    full_feats = build_what_features(works)  # full-works build
    mine, rows, feats, labels, live, matrix = assemble(works, ids, model, columns)  # pool B pieces
    _, rows_a, feats_a, labels_a, live_a, _ = assemble(works, ids_a, model, columns)  # same code on pool A
    cols = [c for c in feats.columns if c not in ("artist_id", "year")]  # model columns
    out = []  # control rows
    def add(name, n, unmatched, diff, limit):  # one control row
        out.append({"control": name, "rows": n, "unmatched": unmatched, "max_abs_diff": diff, "limit": limit, "passed": bool(diff <= limit and unmatched == 0)})  # pass rule
    old_f = pd.read_csv(FOLDER / LIVE["what_features"])  # existing pool A features
    n, un, d = compare(feats_a, old_f, cols)  # features against the file
    add("1a pool A features equal what_features_all_years.csv", n, un + abs(len(old_f) - n), d, FEATURE_TOLERANCE)  # feature control
    old_l = pd.read_csv(FOLDER / LIVE["what_labels"])  # existing pool A labels
    la = labels_a.assign(label_after_holdout=labels_a["label_after_holdout"].astype(float))  # flag as number
    ol = old_l.assign(label_after_holdout=old_l["label_after_holdout"].astype(float))  # flag as number
    n, un, d = compare(la, ol, ["label", "next_active_year", "label_after_holdout"])  # labels, flags and next_active_year
    add("1b pool A labels, flags and next_active_year exactly equal what_labels.csv", n, un + abs(len(old_l) - n), d, 0.0)  # exact control
    stored = pd.read_csv(ROOT / LIVE["what_scores"])  # holdout scores
    stored["artist_id"] = stored["artist_id"].astype("int64")  # whole-number ids
    s = live_a.merge(stored, on=["artist_id", "year"])  # the 160 holdout rows
    add("1c pool A scores equal stage0_score of row_scores.csv", len(s), len(stored[stored["artist_id"].isin(ids_a)]) - len(s), float((s["score"] - s["stage0_score"]).abs().max()), TOLERANCE)  # score control
    for cut in CUTS:  # point-in-time on the pool B Works
        cutw = mine[mine["year"].isna() | (mine["year"] <= cut)]  # Works known at the end of the cut year
        g = build_what_features(cutw).merge(build_artist_years(cutw)[["artist_id", "year", "history_count", "n_works_year"]], on=["artist_id", "year"])  # features and table inputs from that data only
        g = g[g["year"] >= HOLDOUT_START]  # scored years
        full = feats[feats["year"] <= cut]  # same rows from the full build
        n, un, d = compare(full, g, cols)  # cut against full
        add(f"2 point-in-time, data cut at {cut}", n, len(full) - n, d, FEATURE_TOLERANCE)  # one row per cut, extra rows of the other side are not eligible rows
    ff = full_feats[full_feats["artist_id"].isin([float(i) for i in ids]) & (full_feats["year"] >= HOLDOUT_START)]  # pool B rows of the full-works build
    n, un, d = compare(ff, feats, [c for c in cols if c in ff.columns])  # full build against pool-only build
    add("3 full-works build equals the pool-only build", n, len(feats) - n, d, FEATURE_TOLERANCE)  # cross-check, the full build also holds non-eligible artist-years
    dated = works.dropna(subset=["year", "artist_id"])  # dated Works with an artist
    dated = dated[dated["artist_id"].isin([float(i) for i in ids])].assign(artist_id=lambda w: w["artist_id"].astype("int64"))  # pool B only, whole-number ids
    counts = dated.groupby(["artist_id", "year"]).size().rename("n").reset_index()  # Works per artist-year
    counts["history"] = counts.groupby("artist_id")["n"].cumsum() - counts["n"]  # dated Works strictly before the year
    want = counts[(counts["year"] >= HOLDOUT_START) & (counts["history"] >= MIN_PREVIOUS)]  # independent eligibility
    k1, k2 = set(map(tuple, want[["artist_id", "year"]].to_numpy())), set(map(tuple, rows[["artist_id", "year"]].to_numpy()))  # key sets
    add("4 row keys equal the independent count", len(k2), len(k1 ^ k2), 0.0, 0.0)  # row-count control
    add("5a model columns equal columns.json", len(columns), int(columns != list(matrix.columns)), 0.0, 0.0)  # column control
    add("5b scores in 0 to 1", len(live), int(((live["score"] < 0) | (live["score"] > 1)).sum()), 0.0, 0.0)  # range control
    add("5c no duplicated (artist_id, year)", len(live), int(live.duplicated(["artist_id", "year"]).sum()), 0.0, 0.0)  # duplicate control
    ind = independent_labels(mine, ids)  # independent labels
    held = labels[labels["label"].notna() & ~labels["label_after_holdout"]]  # rows with a holdout-rule label
    m = held.merge(ind, on=["artist_id", "year"], how="left", suffixes=("", "_ind"))  # same keys
    add("6 holdout-rule labels equal the independent recomputation", len(held), int(m["label_ind"].isna().sum()), float((m["label"] - m["label_ind"]).abs().max()) if len(m) else 0.0, 0.0)  # label control
    table = pd.DataFrame(out)  # control table
    pd.set_option("display.width", 250, "display.max_colwidth", 90)  # readable table
    print(table.to_string(index=False), flush=True)  # control output
    if not table["passed"].all():  # any failure stops the step
        raise SystemExit("a control failed, nothing was written")  # write nothing
    names = {int(a["discogs_id"]): a["name"] for a in load_demo_config()["artists_b"]}  # display names
    per = labels.assign(year=labels["year"].astype(int)).groupby("artist_id").apply(lambda g: pd.Series({"eligible_years": sorted(g["year"].tolist()), "labelled": int(g["label"].notna().sum()), "flagged": int(g["label_after_holdout"].sum())}), include_groups=False)  # per artist
    print("\nper artist (eligible years from 2018, rows with a label, flagged after the holdout):")  # section title
    for i in ids:  # list order
        r = per.loc[i] if i in per.index else None  # row or none
        print(f"  {names[i]} ({i}): {r['eligible_years'] if r is not None else []}, labelled {r['labelled'] if r is not None else 0}, flagged {r['flagged'] if r is not None else 0}")  # one line per artist
    print("\nrows per year (eligible, labelled, flagged):")  # section title
    print(labels.groupby("year").agg(eligible=("year", "size"), labelled=("label", lambda s: int(s.notna().sum())), flagged=("label_after_holdout", "sum")).to_string())  # counts per year
    for key, frame in (("what_features", feats), ("what_labels", labels), ("what_control", table)):  # the three new files
        with open(files[key], "x", encoding="utf-8", newline="") as handle:  # exclusive creation
            frame.to_csv(handle, index=False)  # write
    after = hashes_of(folders, skip)  # hashes after the run
    print(f"\ncontrol 1d existing files unchanged: {before == after} ({len(before)} files hashed, live_scores.csv not read)")  # hash control
    if before != after:  # cannot happen unless a file changed
        raise SystemExit("an existing file changed")  # report
    print("\nfiles:")  # section title
    for key, path in files.items():  # sizes from disk
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KB  {len(pd.read_csv(path))} rows")  # name, size, rows
    print(f"runtime {time.time() - start:.0f} s")  # runtime
