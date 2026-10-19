"""WHEN assets of the pool B artists: scores from the saved final models (no refit), feature rows, stored labels and flags, controls."""

import hashlib  # SHA-256 of the existing files
import json  # ordered column lists
import time  # runtime of the step

import numpy as np  # NaN-aware comparisons
import pandas as pd  # tables
from xgboost import XGBClassifier  # the saved WHEN models, loaded and not refitted

from app_data import load_demo_config  # demo settings
from backtesting import holdout_gap, split_holdout  # same training rows as the holdout run
from config import CONFIG, ROOT  # project settings and root
from models import build_features  # same model input builder as the first live assets step
from snapshots import build_snapshots  # same snapshot builder as the whole project
from when_demo import HOLDOUT_END, HOLDOUT_START, TARGETS, demo_artist_ids, demo_scores  # rule code of the pool A file, reused as it is

CFG = load_demo_config()  # demo settings
LIVE = CFG["live"]  # live section
POOL_B = CFG["pool_b"]  # pool B section
FOLDER = ROOT / LIVE["folder"]  # live assets folder
DEMO = ROOT / "results/demo"  # demo results folder
OUT = {k: ROOT / POOL_B["paths"][k] for k in ("when_features", "when_scores")}  # new files, never overwritten
OUT["control"] = FOLDER / "control_when_b.csv"  # control table, next to the other live files
TOLERANCE = 0.00005  # largest accepted score difference (4 decimals)
FEATURE_TOLERANCE = 1e-9  # largest accepted feature difference
CUTS = [CONFIG["years"]["development_last"], 2022, 2023, 2024, 2025]  # point-in-time cuts, the same years as the WHAT step
NEVER_READ = "live_scores.csv"  # file that is never read, not even for its hash
KEYS = ["artist_id", "snapshot_year", "work_id"]  # row identity of the feature file


def file_hashes() -> dict:  # SHA-256 of every existing file
    """SHA-256 of every existing file under results/demo, except the one that is never read and the new files."""
    new = {p.resolve() for p in OUT.values()}  # files this step creates
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEMO.rglob("*")) if p.is_file() and p.name not in (NEVER_READ, ".DS_Store") and p.resolve() not in new}  # path -> hash


def refuse_if_written() -> None:  # stop early
    """Stop when any new file exists."""
    for path in OUT.values():  # each new file
        if path.exists():  # never overwrite
            raise SystemExit(f"{path.relative_to(ROOT)} already exists, nothing was run")  # stop early


def load_models() -> tuple:  # saved models and columns
    """The two saved WHEN models and their ordered columns, read from results/demo/live."""
    columns = json.loads((FOLDER / LIVE["columns"]).read_text())  # ordered columns per model
    models = {}  # target -> model
    for target in TARGETS:  # one model per target
        models[target] = XGBClassifier()  # empty model
        models[target].load_model(FOLDER / LIVE["when_models"][target])  # native JSON file, no refit
    return models, {t: columns[f"when_{t}"] for t in TARGETS}  # models and columns


def rows_of(snapshots: pd.DataFrame, ids: list) -> pd.DataFrame:  # snapshot rows of some artists
    """Snapshot rows of the given ids from the holdout start on, with whole-number ids."""
    rows = snapshots[(snapshots["year"] >= HOLDOUT_START) & snapshots["artist_id"].isin([float(i) for i in ids])].copy()  # same filter as the pool A step
    rows["artist_id"] = rows["artist_id"].astype("int64")  # whole-number ids
    return rows  # rows in snapshot order


def build_assets(rows: pd.DataFrame, models: dict, columns: dict, train_counts: pd.Series) -> tuple:  # features and scores
    """Feature file rows and score rows in the layout of the pool A files; labels and flags come from demo_scores."""
    matrix = build_features(rows)  # model input columns
    keys = pd.DataFrame({"artist_id": rows["artist_id"], "snapshot_year": rows["year"], "work_id": rows["work_id"]})  # row identity
    features = pd.concat([keys, matrix], axis=1)  # feature file
    parts = []  # score rows per target
    for target in TARGETS:  # both targets
        score_fn = lambda r, m, t=target: m.predict_proba(build_features(r)[columns[t]])[:, 1]  # score with the saved model and its columns
        parts.append(demo_scores(score_fn, models[target], target, rows, train_counts))  # same rule code as the pool A file
    return features, pd.concat(parts, ignore_index=True), matrix  # features, scores, matrix


def compare(a: pd.DataFrame, b: pd.DataFrame, keys: list, cols: list) -> tuple:  # aligned comparison
    """Rows on both sides, rows on one side only, and the largest absolute difference over cols (NaN equal to NaN)."""
    m = a[keys + cols].merge(b[keys + cols], on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)  # same keys on both sides
    both = m[m["_merge"] == "both"]  # shared rows
    x, y = both[[c + "_a" for c in cols]].astype(float).to_numpy(), both[[c + "_b" for c in cols]].astype(float).to_numpy()  # float arrays
    worst = float("inf") if (np.isnan(x) != np.isnan(y)).any() else (float(np.nanmax(np.abs(x - y))) if len(both) and not np.isnan(x).all() else 0.0)  # inf when NaN cells differ
    return len(both), int((m["_merge"] != "both").sum()), worst  # counts and difference


def controls(ids: list, ids_a: list, snapshots: pd.DataFrame, works_b: pd.DataFrame, models: dict, columns: dict, train_counts: pd.Series, out: tuple, hashes: dict) -> pd.DataFrame:  # all controls
    """Run every control and return the table; the caller prints it and stops when one fails."""
    feats, scores, matrix = out  # pool B outputs
    rows = []  # control rows
    def add(name, n, unmatched, diff, limit):  # one control row
        rows.append({"control": name, "rows": n, "unmatched": unmatched, "max_abs_diff": diff, "limit": limit, "passed": bool(diff <= limit and unmatched == 0)})  # pass rule
    rows_a = rows_of(snapshots, ids_a)  # pool A rows
    feats_a, scores_a, _ = build_assets(rows_a, models, columns, train_counts)  # same code applied to pool A
    old_feats, old_scores = pd.read_csv(FOLDER / LIVE["when_features"]), pd.read_csv(DEMO / "when_demo_scores.csv")  # existing pool A files
    fcols = [c for c in old_feats.columns if c not in KEYS]  # feature columns
    n, un, d = compare(feats_a, old_feats, KEYS, fcols)  # features against when_features.csv
    add("1a pool A features equal when_features.csv", n, un + abs(len(old_feats) - n), d, FEATURE_TOLERANCE)  # feature control
    skeys = ["artist_id", "snapshot_year", "work_id", "target"]  # score row identity
    n, un, d = compare(scores_a, old_scores, skeys, ["score"])  # scores against when_demo_scores.csv
    add("1b pool A scores equal when_demo_scores.csv", n, un + abs(len(old_scores) - n), d, TOLERANCE)  # score control
    exact = ["label", "is_projection", "label_after_holdout", "artist_train_rows"]  # columns that must be exactly equal
    m = scores_a.merge(old_scores, on=skeys, suffixes=("_new", "_old"))  # shared rows
    wrong = sum(int((m[c + "_new"].astype("float64").fillna(-1) != m[c + "_old"].astype("float64").fillna(-1)).sum()) for c in exact)  # differing cells
    add("1c pool A labels, flags and artist_train_rows exactly equal", len(m), 0, float(wrong), 0.0)  # exact control
    add("1d pool A column order equals when_demo_scores.csv", len(scores_a.columns), int(list(scores_a.columns) != list(old_scores.columns)), 0.0, 0.0)  # schema control
    for cut in CUTS:  # point-in-time: features of year <= cut must not change when later Works are removed
        cutw = works_b[works_b["year"].isna() | (works_b["year"] <= cut)]  # Works known at the end of the cut year
        snap = build_snapshots(cutw)  # snapshots from that data only
        snap = snap[(snap["year"] >= HOLDOUT_START) & (snap["year"] <= cut)].copy()  # scored years up to the cut
        snap["artist_id"] = snap["artist_id"].astype("int64")  # whole-number ids
        cut_feats = pd.concat([pd.DataFrame({"artist_id": snap["artist_id"], "snapshot_year": snap["year"], "work_id": snap["work_id"]}), build_features(snap)], axis=1)  # features of the cut build
        full = feats[feats["snapshot_year"] <= cut]  # same rows from the full build
        n, un, d = compare(full, cut_feats, KEYS, fcols)  # cut build against full build
        add(f"2 point-in-time, data cut at {cut}", n, un + abs(len(full) - n), d, FEATURE_TOLERANCE)  # one row per cut
    pool_only = build_snapshots(works_b)  # snapshots from the pool B Works only
    pool_only = rows_of(pool_only, ids)  # pool B rows
    full_works = pd.read_parquet(ROOT / CONFIG["paths"]["works"])  # all Works, not only pool B
    full_build = rows_of(build_snapshots(full_works), ids)  # snapshots from all Works, pool B rows
    for name, other in (("3a full-works rebuild", full_build), ("3b cached snapshots.parquet", rows_of(snapshots, ids))):  # two full builds
        a = pd.concat([pd.DataFrame({"artist_id": pool_only["artist_id"], "snapshot_year": pool_only["year"], "work_id": pool_only["work_id"]}), build_features(pool_only), pool_only[TARGETS]], axis=1)  # pool-only features and labels
        b = pd.concat([pd.DataFrame({"artist_id": other["artist_id"], "snapshot_year": other["year"], "work_id": other["work_id"]}), build_features(other), other[TARGETS]], axis=1)  # other build features and labels
        n, un, d = compare(a, b, KEYS, fcols + TARGETS)  # features and labels
        add(f"{name} equals the pool-only build (features and labels)", n, un + abs(len(a) - n), d, FEATURE_TOLERANCE)  # full-build control
    works_csv = pd.read_csv(ROOT / POOL_B["paths"]["works"])  # pool B Works file of phase P2
    want = set(works_csv.loc[works_csv["year"] >= HOLDOUT_START, "work_id"])  # one snapshot per dated Work from the holdout start
    got = set(scores["work_id"])  # work ids of the new scores
    add("4 row keys equal the Works file (one row per Work from 2018, two targets)", len(scores), len(want ^ got) + abs(len(scores) - 2 * len(want)), 0.0, 0.0)  # key control
    add("5a model columns equal columns.json", sum(len(v) for v in columns.values()), int(any(list(matrix.columns) != columns[t] for t in TARGETS)), 0.0, 0.0)  # column control
    add("5b scores in 0 to 1", len(scores), int(((scores["score"] < 0) | (scores["score"] > 1)).sum()), 0.0, 0.0)  # range control
    add("5c no duplicated key (work_id, target)", len(scores), int(scores.duplicated(["work_id", "target"]).sum()), 0.0, 0.0)  # duplicate control
    last = scores[scores["snapshot_year"] == CONFIG["years"]["partial"]]  # partial year rows
    add("6 partial-year rows are projections without label", len(last), int((~last["is_projection"]).sum() + last["label"].notna().sum()), 0.0, 0.0)  # projection control
    table = pd.DataFrame(rows)  # control table
    return table  # all controls


def run() -> None:  # whole step
    """Score, control, then write the three new files with exclusive creation; nothing is written if a control fails."""
    start = time.time()  # clock of the step
    refuse_if_written()  # stop early
    hashes = file_hashes()  # hashes before the run
    ids, ids_a = [int(a["discogs_id"]) for a in CFG["artists_b"]], demo_artist_ids()  # pool B and pool A ids
    snapshots = pd.read_parquet(ROOT / CONFIG["paths"]["snapshots"])  # cached snapshots of all Works
    final_train, _ = holdout_gap(split_holdout(snapshots)["development"])  # training rows of the holdout run
    train_counts = final_train.assign(artist_id=final_train["artist_id"].astype("int64")).groupby("artist_id").size()  # training rows per artist
    models, columns = load_models()  # saved models, no refit
    out = build_assets(rows_of(snapshots, ids), models, columns, train_counts)  # pool B assets
    works_b = pd.read_parquet(ROOT / CONFIG["paths"]["works"], filters=[("artist_id", "in", [float(i) for i in ids])])  # Works of the pool B artists
    table = controls(ids, ids_a, snapshots, works_b, models, columns, train_counts, out, hashes)  # all controls
    pd.set_option("display.width", 250, "display.max_colwidth", 90)  # readable table
    print(table.to_string(index=False), flush=True)  # control output
    if not table["passed"].all():  # any failure stops the step
        raise SystemExit("a control failed, nothing was written")  # write nothing
    feats, scores, _ = out  # pool B assets
    with open(OUT["when_features"], "x", encoding="utf-8", newline="") as f:  # exclusive creation
        feats.to_csv(f, index=False)  # feature rows
    with open(OUT["when_scores"], "x", encoding="utf-8", newline="") as f:  # exclusive creation
        scores.to_csv(f, index=False)  # scores, labels and flags
    after = file_hashes()  # hashes after the run
    table = pd.concat([table, pd.DataFrame([{"control": f"1e existing files in results/demo unchanged ({len(hashes)} files)", "rows": len(hashes), "unmatched": int(hashes != after), "max_abs_diff": 0.0, "limit": 0.0, "passed": hashes == after}])], ignore_index=True)  # hash control
    with open(OUT["control"], "x", encoding="utf-8", newline="") as f:  # exclusive creation
        table.to_csv(f, index=False)  # control table
    print(f"\ncontrol 1e existing files unchanged: {hashes == after} ({len(hashes)} files hashed, {NEVER_READ} not read)")  # hash control
    if hashes != after:  # cannot happen unless a file changed
        raise SystemExit("an existing file changed")  # report
    g = scores.groupby(["target", "snapshot_year"]).agg(rows=("work_id", "size"), labelled=("is_projection", lambda s: int((~s).sum())), projection=("is_projection", "sum"), label_after_holdout=("label_after_holdout", "sum"))  # counts per year
    print("\ncounts per target and snapshot year:\n" + g.to_string())  # counts
    print("\nfiles:")  # section title
    for k, p in OUT.items():  # sizes from disk
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.1f} KB  {len(pd.read_csv(p))} rows")  # name, size, rows
    print(f"runtime {time.time() - start:.0f} s")  # runtime
