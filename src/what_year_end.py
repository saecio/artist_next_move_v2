"""WHAT assets for artist-years without a dated Work: year-end rows built from the last active year and scored with the saved model (no refit, no evaluation)."""

import json  # column list
import time  # runtime of the step

import numpy as np  # NaN-aware arithmetic
import pandas as pd  # tables
from xgboost import XGBClassifier  # the saved WHAT model, loaded and not refitted

from app_data import load_demo_config  # demo settings and file names
from config import CONFIG, ROOT  # project settings and root
from live_what_extended import FEATURE_TOLERANCE, TOLERANCE, assemble, compare, hashes_of, independent_labels, label_table  # same rules and helpers as the extended WHAT step
from models import WHAT_FEATURES, _what_matrix  # model input columns and matrix builder
from what_target import build_artist_years, build_what_features  # existing builders, not edited
from when_demo import HOLDOUT_START, demo_artist_ids  # first year shown and pool A ids
from when_year_end import pools  # id -> pool of the 61 demo artists

LIVE = load_demo_config()["live"]  # live file names
FOLDER = ROOT / LIVE["folder"]  # live assets folder
OUT = {
    "features": FOLDER / LIVE["what_features_year_end"],  # model input rows
    "labels": FOLDER / LIVE["what_labels_year_end"],  # labels and flags
    "control": FOLDER / LIVE["what_control_year_end"],  # control table
    "scores": ROOT / LIVE["what_scores_year_end"],  # scores read by the app
}  # new files, never overwritten
LAST_COMPLETE = CONFIG["years"]["last_complete"]  # last year with a row, same range as the WHEN year-end rows
MIN_PREVIOUS = CONFIG["what"]["min_previous_works"]  # earlier dated Works required
SHARES = ["albums_share_year", "singles_share_year"]  # shares set to 0 for a year without Works
SAME_YEAR = ["n_works_year", "n_styles_year", "n_new_styles_year"] + SHARES  # same-year columns set to 0
CUTS = [2022, 2024]  # point-in-time cuts


def refuse_if_written() -> None:  # stop early
    """Stop when any new file exists."""
    for path in OUT.values():  # each new file
        if path.exists():  # never overwrite
            raise SystemExit(f"{path.relative_to(ROOT)} already exists, nothing was run")  # stop early


def gap_rows(ay: pd.DataFrame, years: list) -> pd.DataFrame:  # artist-years without a Work
    """One row per artist-year in `years` with no dated Work and an earlier one: last active year L and the dated Works before the year."""
    rows = []  # one dict per gap row
    for aid, g in ay.groupby("artist_id"):  # each artist
        g = g.sort_values("year")  # active years ascending
        active = set(g["year"].astype(int))  # years with a Work
        for year in years:  # candidate years
            before = g[g["year"] < year]  # active years before the year
            if year in active or before.empty:  # a Work exists, or nothing came before
                continue  # no row
            last = before.iloc[-1]  # the L row
            rows.append({"artist_id": int(aid), "year": year, "last_year": int(last["year"]), "history_count": int(last["history_count"] + last["n_works_year"]), "l_history": int(last["history_count"]), "l_works": int(last["n_works_year"])})  # Works before the year = before L plus L
    return pd.DataFrame(rows)  # empty frame if nothing qualifies


def build(mine: pd.DataFrame, years: list) -> tuple:  # gap rows and model matrix from some Works
    """(scored gap rows, model matrix) built from the given Works only; rows need at least MIN_PREVIOUS earlier dated Works."""
    ay = build_artist_years(mine)  # artist-years
    ay = ay.assign(artist_id=ay["artist_id"].astype("int64"))  # whole-number ids
    feats = build_what_features(mine)  # features of every active year
    feats = feats.assign(artist_id=feats["artist_id"].astype("int64"))  # whole-number ids
    gaps = gap_rows(ay, years)  # all gap rows
    gaps = gaps[gaps["history_count"] >= MIN_PREVIOUS].reset_index(drop=True)  # enough earlier Works
    base = pd.DataFrame({"artist_id": gaps["artist_id"], "year": gaps["last_year"], "history_count": gaps["l_history"], "n_works_year": gaps["l_works"]})  # the L rows
    m = _what_matrix(base, feats, WHAT_FEATURES).reset_index(drop=True)  # model columns of the L rows
    gap = (gaps["year"] - gaps["last_year"]).to_numpy(float)  # years since L
    m["history_count"] = gaps["history_count"].to_numpy(float)  # Works before the year
    for c in SAME_YEAR + [c for c in m.columns if c.startswith("genre_")]:  # same-year values and genres
        m[c] = 0.0  # a year without Works
    m["years_since_prev_active_year"] = gap  # distance to L
    m["years_active_so_far"] = m["years_active_so_far"] + gap  # years since the first active year
    m["years_since_new_style"] = m["years_since_new_style"] + gap  # NaN stays NaN
    return gaps, m  # rows and matrix


def lookup(ay: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:  # label rows of the L years
    """Gap rows with the label and next active year of their L row, in the layout of label_table."""
    l = ay.rename(columns={"year": "last_year"})[["artist_id", "last_year", "new_style_year", "next_active_year"]]  # L rows
    return gaps[["artist_id", "year", "last_year"]].merge(l, on=["artist_id", "last_year"], how="left")  # label of L next to the gap row


def sensitivity(m: pd.DataFrame, model, columns: list, cutoff: float) -> None:  # NaN encodings, printed once
    """Scores with NaN instead of 0 for the two shares, then also for n_styles_year and n_new_styles_year: correlation, Yes-share and median absolute difference."""
    base = model.predict_proba(m[columns])[:, 1]  # scores of the chosen encoding
    a = m.copy()  # copy
    a[SHARES] = np.nan  # shares missing
    b = a.copy()  # copy
    b[["n_styles_year", "n_new_styles_year"]] = np.nan  # style counts missing too
    for name, x in (("NaN shares", a), ("NaN shares and style counts", b)):  # both variants
        s = model.predict_proba(x[columns])[:, 1]  # scores
        print(f"sensitivity {name}: correlation {np.corrcoef(base, s)[0, 1]:.3f}, Yes-share {(s >= cutoff).mean():.1%} (zeros: {(base >= cutoff).mean():.1%}), median abs diff {np.median(np.abs(base - s)):.2f}")  # figures


def run() -> None:  # whole step
    """Build, control, and only when every control passes write the four new files with exclusive creation."""
    start = time.time()  # clock of the step
    refuse_if_written()  # stop early
    skip = {p.resolve() for p in OUT.values()}  # files this step creates
    folders = [ROOT / "results/demo", FOLDER, ROOT / "results/holdout_what"]  # folders whose files must not change
    before = hashes_of(folders, skip)  # hashes before the run
    ids = sorted(pools())  # the 61 demo ids
    works = pd.read_parquet(ROOT / CONFIG["paths"]["works"])  # all Works
    mine = works[works["artist_id"].isin([float(i) for i in ids])]  # demo artists' Works
    years = list(range(HOLDOUT_START, LAST_COMPLETE + 1))  # candidate years
    columns = json.loads((FOLDER / LIVE["columns"]).read_text())["what_stage0"]  # ordered columns of the saved model
    model = XGBClassifier()  # empty model
    model.load_model(FOLDER / LIVE["what_model"])  # saved final model, no refit
    gaps, m = build(mine, years)  # scored gap rows and matrix
    ay = build_artist_years(mine)  # artist-years for the labels
    ay = ay.assign(artist_id=ay["artist_id"].astype("int64"))  # whole-number ids
    lab = lookup(ay, gaps)  # L labels
    labels = label_table(lab)  # label, next active year, after-holdout flag
    labels.insert(2, "last_active_year", lab["last_year"])  # L year next to the key
    score = model.predict_proba(m[columns])[:, 1]  # saved model scores
    feats = pd.concat([gaps[["artist_id", "year"]], m], axis=1)  # feature file rows
    scores = pd.concat([gaps[["artist_id", "year", "last_year", "history_count"]].rename(columns={"last_year": "last_active_year"}), pd.DataFrame({"stage0_score": score}), labels[["label", "next_active_year", "label_after_holdout"]].rename(columns={"label": "what_label", "label_after_holdout": "what_label_after_holdout"})], axis=1)  # score file rows
    cutoff = float(pd.read_csv(ROOT / LIVE["what_top10"]).set_index("model").loc[LIVE["what_top10_model"], "score_cutoff"])  # top 10% cutoff of the holdout
    rows = []  # control table
    def add(name, ok, detail):  # one control line
        rows.append({"control": name, "passed": bool(ok), "detail": detail})  # table row
    cols = list(m.columns)  # model columns
    for cut in CUTS:  # point-in-time
        cw = mine[mine["year"].isna() | (mine["year"] <= cut)]  # Works known at the end of the cut year
        g2, m2 = build(cw, [y for y in years if y <= cut])  # rebuild from the past only
        small = pd.concat([g2[["artist_id", "year"]], m2], axis=1)  # cut rows
        big = feats[feats["year"] <= cut]  # same rows of the full build
        n, un, d = compare(big, small, cols)  # features
        add(f"(a) point in time, cut {cut}", un == 0 and d <= FEATURE_TOLERANCE, f"{n} rows, {un} unmatched, max diff {d:.3g}")  # result
    ind = independent_labels(mine, ids).rename(columns={"year": "last_year", "label": "label_ind"})  # labels from plain sets
    chk = lab.merge(ind, on=["artist_id", "last_year"], how="left")  # independent label of the L row
    same = (chk["new_style_year"].isna() & chk["label_ind"].isna()) | (chk["new_style_year"] == chk["label_ind"])  # equal, NaN equal to NaN
    add("(b) gap-row label equals the label of its L row", same.all() and np.array_equal(labels["label"].to_numpy(float), lab["new_style_year"].to_numpy(float), equal_nan=True), f"{len(chk)} rows, {int((~same).sum())} differ")  # result
    nw = pd.read_csv(ROOT / LIVE["what_new_styles"]).assign(has_new=lambda t: t["new_styles"].map(lambda s: float(len(json.loads(s)) > 0)))  # new styles file, as a 0 or 1
    st = lab[lab["new_style_year"].notna()].merge(nw, left_on=["artist_id", "last_year"], right_on=["artist_id", "year"], how="inner")  # labelled rows present in the file
    add("(b) new_styles file agrees with the label (label == new styles not empty)", (st["new_style_year"] == st["has_new"]).all(), f"{len(st)} labelled L rows compared, {int((st['new_style_year'] != st['has_new']).sum())} differ")  # result
    add("(c) model columns equal columns.json", columns == cols, f"{len(columns)} columns")  # columns
    add("(c) no infinite value", bool(np.isfinite(m.to_numpy(float)[~np.isnan(m.to_numpy(float))]).all()), f"{m.shape[0]} rows x {m.shape[1]} columns, NaN cells {int(m.isna().sum().sum())}")  # infinities
    add("(c) scores between 0 and 1", bool(((score >= 0) & (score <= 1)).all()), f"min {score.min():.4f}, max {score.max():.4f}")  # range
    add("(c) no duplicated (artist_id, year)", not scores.duplicated(["artist_id", "year"]).any() and not feats.duplicated(["artist_id", "year"]).any(), f"{len(scores)} rows")  # duplicates
    ids_a = demo_artist_ids()  # pool A ids
    _, _, _, _, live, _ = assemble(works, ids_a, model, columns)  # normal-year scores recomputed in this run
    stored = pd.read_csv(ROOT / LIVE["what_scores"])  # stored holdout scores
    stored = stored.assign(artist_id=stored["artist_id"].astype("int64"))  # whole-number ids
    s = live.merge(stored, on=["artist_id", "year"])  # same artist-years
    d = float((s["score"] - s["stage0_score"]).abs().max())  # largest difference
    add("(d) normal-year scores reproduce stored stage0_score", len(s) > 0 and d <= TOLERANCE, f"{len(s)} rows, max diff {d:.3g}")  # result
    add("(e) existing files unchanged (before the write)", hashes_of(folders, skip) == before, f"{len(before)} files hashed, live_scores.csv neither read nor hashed")  # hashes
    control = pd.DataFrame(rows)  # control table
    pd.set_option("display.width", 250, "display.max_colwidth", 140)  # readable table
    print(control.to_string(index=False), flush=True)  # control output
    if not control["passed"].all():  # any failure stops the step
        raise SystemExit("a control failed, nothing was written")  # write nothing
    figures(gaps, labels, scores, cutoff)  # counts and distribution
    sensitivity(m, model, columns, cutoff)  # NaN encodings
    for key, frame in (("features", feats), ("labels", labels), ("scores", scores)):  # data files
        with open(OUT[key], "x", encoding="utf-8", newline="") as f:  # exclusive creation
            frame.to_csv(f, index=False)  # write
    after_ok = hashes_of(folders, skip) == before  # hashes after the write
    control = pd.concat([control, pd.DataFrame([{"control": "(e) existing files unchanged (after the write)", "passed": after_ok, "detail": f"{len(before)} files hashed"}])], ignore_index=True)  # final line
    with open(OUT["control"], "x", encoding="utf-8", newline="") as f:  # exclusive creation
        control.to_csv(f, index=False)  # write
    print("existing files unchanged after the write:", after_ok)  # (e) after
    for path in OUT.values():  # written files
        print("written", path.relative_to(ROOT), f"{path.stat().st_size / 1024:.1f} KB")  # name and size
    print(f"runtime {time.time() - start:.1f} s")  # seconds


def figures(gaps: pd.DataFrame, labels: pd.DataFrame, scores: pd.DataFrame, cutoff: float) -> None:  # counts only
    """Row counts, score distribution and Yes-share per year; no accuracy figure of any kind."""
    print(f"\nscored rows {len(scores)} (at least {MIN_PREVIOUS} earlier Works), unlabeled {int(labels['label'].isna().sum())}, after the holdout {int(labels['label_after_holdout'].sum())}")  # counts
    print("rows per year:", scores.groupby("year").size().to_dict())  # per year
    print("score distribution:", scores["stage0_score"].describe().round(4).to_dict())  # distribution
    print(f"Yes-share at the cutoff {cutoff:.4f}: {(scores['stage0_score'] >= cutoff).mean():.1%}")  # overall
    print("Yes-share per year:", (scores["stage0_score"] >= cutoff).groupby(scores["year"]).mean().round(3).to_dict())  # per year
