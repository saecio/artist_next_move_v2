"""WHEN assets for artist-years without a dated Work: year-end snapshots scored with the saved models (no refit, no evaluation)."""

import hashlib  # SHA-256 of the existing files
import time  # runtime of the step

import numpy as np  # NaN-aware arithmetic and comparisons
import pandas as pd  # tables

from app_data import load_demo_config  # demo settings
from backtesting import holdout_gap, split_holdout  # same training rows as the holdout run
from config import CONFIG, ROOT  # project settings and root
from demo_game import verdict  # same Yes / No rule as tab 6
from live_pool_b import load_models  # saved models and columns, no refit
from models import GENRES, NUMERIC_FEATURES, build_features  # same model input builder as the other WHEN steps
from snapshots import build_snapshots  # same snapshot builder as the whole project
from when_demo import HOLDOUT_END, HOLDOUT_START, TARGETS, demo_scores  # same label and flag rules as the pool A file

CFG = load_demo_config()  # demo settings
LAST_COMPLETE = CONFIG["years"]["last_complete"]  # last fully observed year, last year with a row
THRESHOLD = CFG["game"]["threshold"]  # Yes at or above this score, as in tab 6
DEMO = ROOT / "results/demo"  # demo results folder
LIVE = ROOT / CFG["live"]["folder"]  # live assets folder
OUT = {
    "features": LIVE / "when_features_year_end.csv",  # feature rows
    "scores": DEMO / "when_demo_scores_year_end.csv",  # long score file
    "route": DEMO / "route_table_year_end.csv",  # extended route table
    "control": LIVE / "control_when_year_end.csv",  # control table
}  # new files, never overwritten
NEVER_READ = "live_scores.csv"  # file that is never read, not even for its hash
CHECKED = ["history_count", "years_since_last", "years_active_so_far", "historical_rate"]  # history features of check (c)
CUTS = [2022, 2023, 2024, 2025]  # point-in-time cuts, the same years as the pool B step
TOL = 1e-9  # largest accepted feature difference
KEYS = ["artist_id", "year"]  # row identity of the year-end rows


def refuse_if_written() -> None:  # stop early
    """Stop when any new file exists."""
    for path in OUT.values():  # each new file
        if path.exists():  # never overwrite
            raise SystemExit(f"{path.relative_to(ROOT)} already exists, nothing was run")  # stop early


def file_hashes() -> dict:  # SHA-256 of every existing file
    """SHA-256 of every file under results/demo (live folder included), except the unread file and the new files."""
    new = {p.resolve() for p in OUT.values()}  # files this step creates
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEMO.rglob("*")) if p.is_file() and p.name not in (NEVER_READ, ".DS_Store") and p.resolve() not in new}  # path -> hash


def pools() -> dict:  # pool key of every demo artist
    """Artist id -> pool key, from the lists named in pool_b.pool_lists."""
    return {int(a["discogs_id"]): key for key, lst in CFG["pool_b"]["pool_lists"].items() for a in CFG[lst]}  # id -> pool


def dated_works(ids: list) -> pd.DataFrame:  # Works of the demo artists with a year
    """Dated Works of the given artists from the cleaned Works table (undated Works never get a snapshot)."""
    works = pd.read_parquet(ROOT / CONFIG["paths"]["works"], filters=[("artist_id", "in", [float(i) for i in ids])])  # demo artists only
    return works[works["year"].notna()].reset_index(drop=True)  # dated Works


def year_end_rows(snaps: pd.DataFrame, years: list, last_complete: int = LAST_COMPLETE) -> pd.DataFrame:  # one row per artist-year without a Work
    """Year-end rows for every artist-year in `years` with no Work in that year and at least one earlier Work.

    `snaps` is the output of build_snapshots (sorted, one row per dated Work). Each feature is the value the
    snapshot of the first Work of that year would have (history_count = Works before the year, years_since_last,
    years_active_so_far, historical_rate), with the same-year counts and the Work type set to 0. Genres are those
    of the last Work before the year in the snapshot sort order. The targets use the rule of build_snapshots.
    """
    rows = []  # one dict per year-end row
    for artist_id, g in snaps.groupby("artist_id", sort=False):  # each artist, snapshots already sorted
        active = set(g["year"].astype(int))  # years with at least one Work
        for year in years:  # candidate years
            prior = g[g["year"] < year]  # only Works dated before the year (point in time)
            if year in active or prior.empty:  # a Work exists that year, or nothing came before
                continue  # no year-end row
            first, last = int(prior["year"].iloc[0]), int(prior["year"].iloc[-1])  # first and last Work years up to the cut
            elapsed = year - first  # years since the first Work, always >= 1 here
            later = sorted(y for y in active if y > year)  # active years after the cut, used only for the targets
            observed = last_complete - year  # fully observed years that follow
            target = {}  # next_1y and next_2y
            for name, horizon in (("next_1y", 1), ("next_2y", 2)):  # both horizons
                if later:  # a next active year exists
                    target[name] = float(later[0] - year <= horizon)  # measured gap
                else:  # none observed: a confident 0 only when enough complete years followed
                    target[name] = 0.0 if observed >= horizon else np.nan  # else unknown
            rows.append({
                "artist_id": int(artist_id), "year": year, "work_id": np.nan,  # identity, no Work
                "works_so_far_this_year": 0, "albums_so_far_this_year": 0, "singles_so_far_this_year": 0,  # no Work in the year
                "is_album_like": 0, "is_single_like": 0,  # both false means other
                "history_count": len(prior), "years_since_last": year - last, "years_active_so_far": elapsed,  # point-in-time history
                "historical_rate": len(prior) / elapsed,  # elapsed > 0 by construction
                "genres": prior["genres"].iloc[-1],  # genres of the last Work before the cut
                **target,  # next_1y and next_2y
            })
    return pd.DataFrame(rows)  # empty frame if nothing qualifies


def features_of(ye: pd.DataFrame) -> pd.DataFrame:  # feature file layout
    """Identity columns plus the model input columns of the year-end rows."""
    keys = pd.DataFrame({"artist_id": ye["artist_id"], "snapshot_year": ye["year"], "work_id": ye["work_id"]})  # row identity
    return pd.concat([keys, build_features(ye)], axis=1)  # feature file


def score_rows(ye: pd.DataFrame, models: dict, columns: dict, train_counts: pd.Series) -> pd.DataFrame:  # long score file
    """Scores in the layout of when_demo_scores.csv plus is_year_end; labels and flags come from demo_scores."""
    parts = []  # one block per target
    for target in TARGETS:  # both horizons
        score_fn = lambda r, m, t=target: m.predict_proba(build_features(r)[columns[t]])[:, 1]  # saved model and its columns
        parts.append(demo_scores(score_fn, models[target], target, ye, train_counts))  # same rule code as the pool A file
    out = pd.concat(parts, ignore_index=True)  # both targets
    out["is_year_end"] = True  # every row of this file is a year-end row
    return out  # long format


def max_diff(a: pd.DataFrame, b: pd.DataFrame, cols: list) -> float:  # largest absolute difference, NaN equal to NaN
    """Largest absolute difference over `cols` of two aligned frames; inf when the NaN positions differ."""
    x, y = a[cols].to_numpy(float), b[cols].to_numpy(float)  # arrays
    if (np.isnan(x) != np.isnan(y)).any():  # different missing positions
        return float("inf")  # not equal
    return float(np.nanmax(np.abs(x - y))) if x.size else 0.0  # largest difference


def control_row(name: str, passed: bool, detail: str) -> dict:  # one control line
    """One row of the control table."""
    return {"control": name, "passed": bool(passed), "detail": detail}  # table row


def point_in_time(works: pd.DataFrame, full: pd.DataFrame) -> list:  # control (b)
    """Rebuild the rows with the Works after each cut removed and compare features with the full build up to the cut."""
    rows = []  # control rows
    for cut in CUTS:  # each cut
        part = build_snapshots(works[works["year"] <= cut], last_complete_year=cut)  # no Work after the cut exists
        small = year_end_rows(part, list(range(HOLDOUT_START, cut + 1)), cut)  # rows rebuilt from the past only
        big = full[full["year"] <= cut]  # same rows of the full build
        keys_equal = small[KEYS].reset_index(drop=True).equals(big[KEYS].reset_index(drop=True))  # same artist-years
        diff = max_diff(features_of(small), features_of(big), [c for c in features_of(big).columns if c not in ("artist_id", "snapshot_year", "work_id")]) if keys_equal else float("inf")  # features
        rows.append(control_row(f"point in time, cut {cut}", keys_equal and diff <= TOL, f"{len(small)} rows, max diff {diff:.3g}"))  # result
    return rows  # control rows


def reproduction(works: pd.DataFrame, snaps: pd.DataFrame, years: list) -> tuple:  # control (c)
    """Year-end rows built for artist-years that have Works (that year's Works removed) against the first-Work snapshot."""
    parts = []  # rebuilt rows per year
    for year in years:  # each year
        cut = build_snapshots(works[works["year"] != year])  # that year's Works removed from the table
        parts.append(year_end_rows(cut, [year]))  # year-end row of every artist without a Work that year
    rebuilt = pd.concat(parts, ignore_index=True)  # all rebuilt rows
    first = snaps[(snaps["works_so_far_this_year"] == 1) & snaps["year"].isin(years)].assign(artist_id=lambda d: d["artist_id"].astype("int64"))  # first Work of each artist-year
    merged = rebuilt.merge(first, on=["artist_id", "year"], suffixes=("", "_snap"))  # artist-years present on both sides
    diffs = {c: float(np.nanmax(np.abs(merged[c].to_numpy(float) - merged[f"{c}_snap"].to_numpy(float)))) if len(merged) else 0.0 for c in CHECKED + ["next_1y", "next_2y"]}  # per column
    nan_ok = all(((merged[c].isna()) == (merged[f"{c}_snap"].isna())).all() for c in CHECKED + ["next_1y", "next_2y"])  # same missing positions
    same_genre = float(np.mean([set(a) == set(b) for a, b in zip(merged["genres"], merged["genres_snap"])])) if len(merged) else 1.0  # informative only
    return merged, diffs, nan_ok, same_genre  # for the control table


def yn(label) -> object:  # reality text
    """'yes' or 'no' for a stored label, None when empty."""
    return None if pd.isna(label) else ("yes" if label == 1 else "no")  # label text


def horizon_columns(scores: pd.DataFrame, horizon: str) -> pd.DataFrame:  # score, verdict, reality, counted, right of one horizon
    """Columns of one horizon from rows holding score, label, is_projection and label_after_holdout."""
    s = scores.reset_index(drop=True)  # clean index
    out = pd.DataFrame({f"score_{horizon}": s["score"].round(4), f"verdict_{horizon}": [verdict(x, THRESHOLD) for x in s["score"]]})  # score and model answer
    out[f"reality_{horizon}"] = [yn(x) for x in s["label"]]  # stored label as text, shown even when not counted
    out[f"counted_{horizon}"] = (s["label"].notna() & ~s["is_projection"].astype(bool) & ~s["label_after_holdout"].astype(bool)).to_numpy()  # same rule as is_counted of src/demo_game.py
    out[f"right_{horizon}"] = [(v == r) if (r is not None and c) else None for v, r, c in zip(out[f"verdict_{horizon}"], out[f"reality_{horizon}"], out[f"counted_{horizon}"])]  # right only when counted
    return out  # horizon block


def route_table(ye_scores: pd.DataFrame, ye: pd.DataFrame) -> tuple:  # extended route table
    """Existing representative rows plus the year-end rows, with 1-year and 2-year columns and the WHAT columns."""
    old = pd.read_csv(DEMO / "route_table.csv")  # existing 430 rows
    stored = pd.concat([pd.read_csv(DEMO / "when_demo_scores.csv"), pd.read_csv(DEMO / "when_demo_scores_b.csv")], ignore_index=True)  # stored scores of both pools
    rep = pd.concat([pd.read_csv(DEMO / "works_demo.csv"), pd.read_csv(DEMO / "works_demo_b.csv")], ignore_index=True)  # Works of both pools
    rep = rep[rep["is_representative"]][["artist_id", "year", "work_id"]]  # representative Work of each artist-year
    base = old[["artist", "artist_id", "pool", "year"]].merge(rep, on=["artist_id", "year"], how="left")  # Work behind each existing row
    blocks = []  # horizon blocks of the existing rows
    for target, horizon in (("next_1y", "1y"), ("next_2y", "2y")):  # both horizons
        sel = stored[stored["target"] == target][["work_id", "score", "label", "is_projection", "label_after_holdout"]]  # stored rows of the horizon
        blocks.append(horizon_columns(base[["work_id"]].merge(sel, on="work_id", how="left"), horizon))  # existing rows
    existing = pd.concat([base[["artist", "artist_id", "pool", "year"]], blocks[0], blocks[1]], axis=1)  # existing rows, new layout
    existing.insert(4, "is_year_end", False)  # representative rows
    what_cols = [c for c in old.columns if c.startswith("what_")]  # WHAT columns of route_table.csv
    existing = pd.concat([existing, old[what_cols]], axis=1)  # WHAT columns carried over
    names = {int(a["discogs_id"]): a["name"] for lst in CFG["pool_b"]["pool_lists"].values() for a in CFG[lst]}  # id -> name
    pool = pools()  # id -> pool
    new = pd.DataFrame({"artist": ye["artist_id"].map(names), "artist_id": ye["artist_id"], "pool": ye["artist_id"].map(pool), "year": ye["year"], "is_year_end": True})  # identity of the new rows
    nblocks = []  # horizon blocks of the new rows
    for target, horizon in (("next_1y", "1y"), ("next_2y", "2y")):  # both horizons
        sel = ye_scores[ye_scores["target"] == target].set_index(["artist_id", "snapshot_year"]).loc[list(zip(ye["artist_id"], ye["year"]))].reset_index()  # rows in ye order
        nblocks.append(horizon_columns(sel, horizon))  # new rows
    new = pd.concat([new.reset_index(drop=True), nblocks[0], nblocks[1]], axis=1)  # new rows, same layout
    for c in what_cols:  # WHAT columns do not exist for the new rows
        new[c] = False if c == "what_available" else None  # no WHAT score here
    table = pd.concat([existing, new], ignore_index=True).sort_values(["artist", "year", "is_year_end"]).reset_index(drop=True)  # one table
    return table, old, existing  # table, original, recomputed existing rows


def rate(s: pd.Series) -> str:  # right of known
    """Text 'k right of n with a counted reality'."""
    k = s.dropna()  # rows with a counted reality
    return f"{int(k.sum())} right of {len(k)} with a counted reality"  # text


def summary(table: pd.DataFrame) -> None:  # route table report
    """Print the 1-year No report; no route is chosen."""
    pd.set_option("display.width", 250, "display.max_rows", 2000)  # readable output
    n1 = table[table["verdict_1y"] == "no"]  # 1-year No verdicts
    n12 = n1[n1["verdict_2y"] == "yes"]  # of those, 2-year Yes
    n11 = n1[n1["verdict_2y"] == "no"]  # of those, 2-year No
    print(f"rows {len(table)} (representative {int((~table['is_year_end']).sum())}, year-end {int(table['is_year_end'].sum())})")  # size
    print(f"1-year No verdicts: {len(n1)} ({rate(n1['right_1y'])}; year-end {int(n1['is_year_end'].sum())})")  # 1-year No
    print(f"  of them 2-year Yes: {len(n12)}; 2-year verdict right on those: {rate(n12['right_2y'])}")  # 2-year Yes
    print(f"  of them 2-year No: {len(n11)}; 2-year verdict right on those: {rate(n11['right_2y'])}")  # 2-year No
    cols = ["artist", "pool", "year", "is_year_end", "score_1y", "verdict_1y", "reality_1y", "counted_1y", "score_2y", "verdict_2y", "reality_2y", "counted_2y"]  # printed columns
    print("\n1-year No and 2-year Yes:")  # title
    print(n12[cols].to_string(index=False))  # rows
    print("\n1-year No and 2-year No:")  # title
    print(n11[cols].to_string(index=False))  # rows


def run() -> None:  # whole step
    """Build, control, and only when every control passes write the four new files with exclusive creation."""
    start = time.time()  # clock of the step
    refuse_if_written()  # stop early
    before = file_hashes()  # hashes before the run
    pool = pools()  # id -> pool
    ids = sorted(pool)  # the 61 demo ids
    works = dated_works(ids)  # dated Works of the demo artists
    snaps = build_snapshots(works)  # snapshots of the demo artists, sort order and genres
    years = list(range(HOLDOUT_START, LAST_COMPLETE + 1))  # candidate years
    ye = year_end_rows(snaps, years)  # year-end rows
    cached = pd.read_parquet(ROOT / CONFIG["paths"]["snapshots"])  # cached snapshots of all Works
    final_train, _ = holdout_gap(split_holdout(cached)["development"])  # training rows of the holdout run
    train_counts = final_train.assign(artist_id=final_train["artist_id"].astype("int64")).groupby("artist_id").size()  # training rows per artist
    models, columns = load_models()  # saved models, no refit
    features = features_of(ye)  # feature rows
    scores = score_rows(ye, models, columns, train_counts)  # long score file
    table, old, existing = route_table(scores, ye)  # extended route table
    rows = []  # control table
    rows.append(control_row("existing files unchanged (before the write)", file_hashes() == before, f"{len(before)} files hashed, {NEVER_READ} neither read nor hashed"))  # (a)
    rows += point_in_time(works, ye)  # (b)
    merged, diffs, nan_ok, same_genre = reproduction(works, snaps, years)  # (c)
    rows.append(control_row("reproduction vs first-Work snapshot", nan_ok and max(diffs.values()) <= TOL and len(merged) > 0, f"{len(merged)} artist-years, max diff per column {({k: float(f'{v:.3g}') for k, v in diffs.items()})}, genre set equal in {same_genre:.1%}"))  # (c)
    rows.append(control_row("model columns equal columns.json", all(list(build_features(ye)[columns[t]].columns) == columns[t] for t in TARGETS) and set(columns[TARGETS[0]]) == set(NUMERIC_FEATURES) | {f"genre_{g}" for g in GENRES}, f"{len(columns[TARGETS[0]])} columns per model"))  # (d)
    matrix = build_features(ye)[columns[TARGETS[0]]].to_numpy(float)  # model inputs
    rows.append(control_row("no missing or infinite model input", bool(np.isfinite(matrix).all()), f"{matrix.shape[0]} rows x {matrix.shape[1]} columns"))  # (d)
    rows.append(control_row("scores between 0 and 1", bool(scores["score"].between(0, 1).all()), f"min {scores['score'].min():.4f}, max {scores['score'].max():.4f}"))  # (d)
    rows.append(control_row("no duplicated (artist, year)", not ye.duplicated(KEYS).any() and not scores.duplicated(["artist_id", "snapshot_year", "target"]).any(), f"{len(ye)} rows"))  # (e)
    rows.append(control_row("scope: 2018 to last complete year, after the first Work, 61 artists", bool(ye["year"].between(HOLDOUT_START, LAST_COMPLETE).all()) and set(ye["artist_id"]) <= set(ids) and len(ids) == 61, f"{ye['artist_id'].nunique()} artists with a row, years {ye['year'].min()} to {ye['year'].max()}"))  # scope
    same_old = existing[["pool", "year", "verdict_1y", "reality_1y"]].reset_index(drop=True).equals(old[["pool", "year", "when_verdict", "when_reality"]].rename(columns={"when_verdict": "verdict_1y", "when_reality": "reality_1y"}).reset_index(drop=True))  # recomputed vs stored route table
    rows.append(control_row("existing route rows reproduced from the stored score files", bool(same_old), f"{len(old)} rows, 1-year verdict and reality equal route_table.csv"))  # route table
    control = pd.DataFrame(rows)  # control table
    pd.set_option("display.width", 250, "display.max_colwidth", 140)  # readable table
    print(control.to_string(index=False), flush=True)  # control output
    if not control["passed"].all():  # any failure stops the step
        raise SystemExit("a control failed, nothing was written")  # write nothing
    figures(ye, scores, table, pool)  # point 5(f) figures
    for key, frame in (("features", features), ("scores", scores), ("route", table)):  # data files
        with open(OUT[key], "x", encoding="utf-8", newline="") as f:  # exclusive creation
            frame.to_csv(f, index=False)  # write
    after_ok = file_hashes() == before  # hashes after the write
    control = pd.concat([control, pd.DataFrame([control_row("existing files unchanged (after the write)", after_ok, f"{len(before)} files hashed")])], ignore_index=True)  # final line
    with open(OUT["control"], "x", encoding="utf-8", newline="") as f:  # exclusive creation
        control.to_csv(f, index=False)  # write
    for path in OUT.values():  # written files
        print("written", path.relative_to(ROOT), f"{path.stat().st_size / 1024:.1f} KB")  # name and size
    print("existing files unchanged after the write:", after_ok)  # (a) after
    summary(table)  # point 6 report
    print(f"runtime {time.time() - start:.1f} s")  # seconds


def figures(ye: pd.DataFrame, scores: pd.DataFrame, table: pd.DataFrame, pool: dict) -> None:  # point 5(f)
    """Print rows per year and pool, Yes shares, No/Yes crossings and the share of counted realities."""
    print("\nrows per year:", ye.groupby("year").size().to_dict())  # per year
    print("rows per pool:", ye["artist_id"].map(pool).value_counts().to_dict())  # per pool
    for target in TARGETS:  # both horizons
        s = scores[scores["target"] == target]  # score rows
        print(f"{target}: Yes share at {THRESHOLD}: {(s['score'] >= THRESHOLD).mean():.1%} ({int((s['score'] >= THRESHOLD).sum())} of {len(s)})")  # Yes share
    wide = scores.pivot(index=["artist_id", "snapshot_year"], columns="target", values="score")  # both scores per row
    print("next_1y No and next_2y Yes:", int(((wide["next_1y"] < THRESHOLD) & (wide["next_2y"] >= THRESHOLD)).sum()))  # crossing
    new = table[table["is_year_end"]]  # year-end rows of the table
    print(f"counted reality: 1-year {new['counted_1y'].mean():.1%}, 2-year {new['counted_2y'].mean():.1%}")  # share counted
    print(f"any label: 1-year {new['reality_1y'].notna().mean():.1%}, 2-year {new['reality_2y'].notna().mean():.1%}")  # share with a label
