"""Duplicate sensitivity analysis on the development partition: collapse same-artist same-title works, refit, compare."""

import re  # title normalization
import time  # seconds per fit

import numpy as np  # arrays and random draws
import pandas as pd  # dataframes
from sklearn.metrics import average_precision_score, roc_auc_score  # point metrics

from backtesting import HOLDOUT_START_YEAR, make_folds, make_what_folds, split_holdout, year_distribution  # fold builders and the split
from baselines import N_BOOTSTRAP, SEED  # bootstrap draws and the project seed
from models import FINAL_XGB_PARAMS, make_what_xgboost, make_xgboost  # model pairs, stage 0 plus overrides
from snapshots import build_snapshots  # WHEN snapshots
from tuning import read_log  # trial log reader
from what_target import build_artist_years, build_what_features, build_what_table  # WHAT table and inputs

WHAT_TARGET = "new_style_year"  # WHAT target column
TARGETS = ["next_1y", "next_2y", WHAT_TARGET]  # the three targets


def normalize_title(title):
    """Lowercase, strip punctuation, collapse whitespace (copied verbatim from notebooks/01_cleaning_eda.ipynb)."""
    text = str(title).strip().lower()  # plain lowercase text
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation becomes a space
    text = re.sub(r"\s+", " ", text).strip()  # repeated whitespace collapses
    return text  # normalized title


def collapse_works(works: pd.DataFrame) -> tuple:
    """Definition A: keep the earliest work of each (artist_id, normalized title) cluster, any year.

    A missing year sorts last and ties go to the smaller work_id. Rows without artist_id are never
    clustered. Returns (collapsed works in the original row order, dict of cluster counts).
    """
    assert works["work_id"].is_unique, "work_id is not unique"  # the tie-break needs a unique id
    title = works["title"].map(normalize_title)  # normalized title of every row
    has_artist = works["artist_id"].notna()  # null artist_id rows are never clustered
    frame = pd.DataFrame({
        "work_id": works["work_id"].to_numpy(),  # tie-break and row identity
        "artist_id": works["artist_id"].to_numpy(),  # cluster key part 1
        "title": title.to_numpy(),  # cluster key part 2
        "year": works["year"].to_numpy(),  # earliest year wins
    })[has_artist.to_numpy()]  # only rows that can be clustered
    frame = frame.sort_values(["artist_id", "title", "year", "work_id"], na_position="last")  # earliest first, missing year last
    size = frame.groupby(["artist_id", "title"])["work_id"].transform("size")  # cluster size of every row
    keep = ~frame.duplicated(["artist_id", "title"], keep="first")  # first row of each cluster stays
    late = frame.assign(late=frame["year"] >= HOLDOUT_START_YEAR).groupby(["artist_id", "title"])["late"].transform("any")  # cluster with a member dated 2018 or later
    drop_ids = frame.loc[~keep, "work_id"]  # works removed by the collapse
    collapsed = works[~works["work_id"].isin(drop_ids)]  # every other row stays, original order
    info = {
        "works_before": len(works),  # works in the original table
        "works_after": len(collapsed),  # works after the collapse
        "works_removed": len(drop_ids),  # works removed
        "clusters_size_2_plus": int(((size > 1) & keep).sum()),  # clusters with at least two members
        "clusters_with_work_2018_plus": int(((size > 1) & keep & late).sum()),  # clusters whose members include a work dated 2018 or later
        "works_null_title": int(works["title"].isna().sum()),  # titles that normalize to the string "nan"
        "works_empty_normalized_title": int((title == "").sum()),  # titles that normalize to an empty string
        "works_null_artist": int((~has_artist).sum()),  # never clustered
        "works_missing_year": int(works["year"].isna().sum()),  # sort last inside a cluster
    }
    return collapsed, info  # collapsed table and counts


def build_version(works: pd.DataFrame, low_volume_years=None) -> dict:
    """Rebuild the WHEN snapshots and the WHAT table in memory, then keep development rows only."""
    snapshots = build_snapshots(works)  # all works, as the original tables were built
    if low_volume_years is None:  # the original version defines the low-volume years
        low_volume_years = year_distribution(snapshots).query("low_volume").index  # years no test tranche may touch
    development = split_holdout(snapshots)["development"]  # rows with year < 2018, the rest is dropped here
    table = build_what_table(build_artist_years(works))  # labelled artist-year rows, all years
    table = table[table["year"] < HOLDOUT_START_YEAR]  # drop 2018 onward at once
    features = build_what_features(works)  # WHAT inputs, all years
    features = features[features["year"] < HOLDOUT_START_YEAR]  # drop 2018 onward at once
    assert development["year"].max() < HOLDOUT_START_YEAR and table["year"].max() < HOLDOUT_START_YEAR, "a row from 2018 or later is left"  # no holdout row
    return {"dev": development, "table": table, "features": features, "low": low_volume_years}  # dev tables of one data version


def get_folds(version: dict, target: str) -> list:
    """Development folds of one target, from the existing fold builders."""
    if target == WHAT_TARGET:  # WHAT folds use the label-resolution rule
        return make_what_folds(version["table"])  # five folds, no fixed gap
    return make_folds(version["dev"], target=target, low_volume_years=version["low"])  # WHEN folds, label gap of 2 years


def configs(target: str) -> dict:
    """Two configurations per target: stage 0 and final (WHEN) or tuned candidate (WHAT)."""
    if target == WHAT_TARGET:  # WHAT compares stage 0 with the rejected candidate
        logged = {str(r["trial"]): r for r in read_log("results/trials/xgb_what_new_style_year.jsonl")}  # trial id to logged row
        return {"stage0": {}, "final": logged["m_lr_0.012"]["params"]}  # candidate params read from the log
    return {"stage0": {}, "final": FINAL_XGB_PARAMS[target]}  # WHEN final params from src/models.py


def fit_fold(version: dict, folds: list, target: str, params: dict, k: int) -> tuple:
    """Fit one fold (index k) and score its test rows. Returns (rows with score, fit seconds)."""
    fold = folds[k]  # train and test frames
    if target == WHAT_TARGET:  # WHAT model pair
        fit_fn, score_fn = make_what_xgboost(version["features"], target=target, **params)  # same builder as the tuning
    else:  # WHEN model pair
        fit_fn, score_fn = make_xgboost(target, **params)  # same builder as the tuning
    start = time.time()  # clock of the fit and the scoring
    test = fold["test"]  # test rows of the fold
    scores = np.asarray(score_fn(test, fit_fn(fold["train"])))  # fit on train only, score test
    seconds = time.time() - start  # fit plus scoring time
    key = test["artist_id"].to_numpy().astype("int64") * 10000 + test["year"].to_numpy() if target == WHAT_TARGET else test["work_id"].to_numpy()  # row identity: artist-year for WHAT, work for WHEN
    rows = pd.DataFrame({
        "fold": fold["fold"],  # 1-based fold number
        "key": key,  # row identity shared by both versions
        "artist_id": test["artist_id"].to_numpy(),  # cluster of the bootstrap
        "y": test[target].to_numpy().astype(float),  # resolved label
        "score": scores,  # model score
    })
    return rows, seconds  # scores and time


def point_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    """PR-AUC, ROC-AUC, positive-rate baseline and gain (PR-AUC minus baseline)."""
    pr = float(average_precision_score(y, score))  # PR-AUC
    base = float(y.mean())  # no-skill PR-AUC
    return {"pr_auc": pr, "roc_auc": float(roc_auc_score(y, score)), "baseline": base, "gain": pr - base}  # one row of metrics


def _sorted_set(y: np.ndarray, score: np.ndarray, code: np.ndarray) -> tuple:
    """Sort rows by descending score once; return labels, artist codes and the last index of each tie block."""
    order = np.argsort(-score, kind="stable")  # best score first
    s = score[order]  # sorted scores
    ends = np.append(np.flatnonzero(s[1:] != s[:-1]), len(s) - 1)  # last row of every tie block
    return y[order], code[order], ends  # sorted inputs


def _weighted_metrics(y: np.ndarray, w: np.ndarray, ends: np.ndarray) -> tuple:
    """Weighted PR-AUC (average precision), ROC-AUC and baseline of rows sorted by score."""
    tp = np.cumsum(w * y)[ends]  # weighted positives up to each threshold
    seen = np.cumsum(w)[ends]  # weighted rows up to each threshold
    pos, neg = tp[-1], seen[-1] - tp[-1]  # weighted positives and negatives
    if pos == 0 or neg == 0:  # one class only, undefined
        return np.nan, np.nan, np.nan  # all undefined
    recall = tp / pos  # recall at each threshold
    precision = tp / np.maximum(seen, 1e-12)  # precision at each threshold
    pr = np.sum(np.diff(recall, prepend=0.0) * precision)  # average precision
    fpr = (seen - tp) / neg  # false positive rate at each threshold
    tpr_prev = np.concatenate(([0.0], recall[:-1]))  # recall at the previous threshold
    roc = np.sum(np.diff(fpr, prepend=0.0) * (recall + tpr_prev) / 2)  # trapezoid area
    return pr, roc, pos / seen[-1]  # PR-AUC, ROC-AUC, baseline


def bootstrap_sets(sets: dict, n_artists: int, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Artist-clustered bootstrap of several row sets with the same artist draws.

    `sets` maps a name to (y, score, artist code). Every draw resamples artists with replacement once
    and gives each row the weight of its artist, so all sets see the same resampled artists.
    Returns name -> array (n_bootstrap, 3) of PR-AUC, ROC-AUC and baseline.
    """
    prepared = {name: _sorted_set(*s) for name, s in sets.items()}  # sort every set once
    out = {name: np.empty((n_bootstrap, 3)) for name in sets}  # draws of every set
    rng = np.random.default_rng(seed)  # fixed seed, same draws for every group of equal size
    for i in range(n_bootstrap):  # one resample per iteration
        counts = np.bincount(rng.integers(0, n_artists, n_artists), minlength=n_artists).astype(float)  # times each artist is drawn
        for name, (y, code, ends) in prepared.items():  # same counts for every set
            out[name][i] = _weighted_metrics(y, counts[code], ends)  # metrics under these weights
    return out  # draws per set


def interval(values: np.ndarray, level: float = 0.95) -> tuple:
    """Percentile interval of bootstrap draws, NaN draws ignored."""
    alpha = (1 - level) / 2  # tail on each side
    return float(np.nanquantile(values, alpha)), float(np.nanquantile(values, 1 - alpha))  # (low, high)


GROUPS = {**{f"fold{k}": [k] for k in range(1, 6)}, "pooled": [1, 2, 3, 4, 5], "pooled_no_fold5": [1, 2, 3, 4]}  # folds, all rows pooled, and pooled without the look-ahead fold 5


def analyze(orig: pd.DataFrame, coll: pd.DataFrame, target: str, config: str, universe: pd.Index) -> list:
    """Metrics, original-data intervals and paired differences of one target and configuration, one dict per group.

    `orig` and `coll` hold the scored test rows of each version (fold, key, artist_id, y, score).
    `universe` is the artist_id index shared by both versions, so both see the same bootstrap artists.
    """
    out = []  # one dict per group
    for group, fold_list in GROUPS.items():  # folds and pooled groups
        o = orig[orig["fold"].isin(fold_list)]  # original rows of the group
        c = coll[coll["fold"].isin(fold_list)]  # collapsed rows of the group
        oc = o[np.isin(o["key"].to_numpy(), c["key"].to_numpy())]  # original rows also present in the collapsed version
        cc = c[np.isin(c["key"].to_numpy(), o["key"].to_numpy())]  # collapsed rows also present in the original version
        assert len(oc) == len(cc), "the common rows differ in count"  # same keys on both sides
        arrays = lambda d: (d["y"].to_numpy(), d["score"].to_numpy(), universe.get_indexer(d["artist_id"].to_numpy()))  # labels, scores and artist codes
        draws = bootstrap_sets({"o_full": arrays(o), "o_common": arrays(oc), "c_common": arrays(cc)}, len(universe))  # same artist draws for the three sets
        p = {name: point_metrics(d["y"].to_numpy(), d["score"].to_numpy()) for name, d in (("o", o), ("c", c), ("oc", oc), ("cc", cc))}  # point metrics of the four sets
        full = draws["o_full"]  # draws of the original full table
        row = {"target": target, "config": config, "group": group, "n_orig": len(o), "n_coll": len(c), "n_common": len(oc)}  # identifiers and counts
        for name, label in (("o", "orig"), ("c", "coll")):  # point metrics of each full table
            row.update({f"{label}_{m}": v for m, v in p[name].items()})  # pr_auc, roc_auc, baseline, gain
        for i, m in enumerate(["pr_auc", "roc_auc", "baseline"]):  # intervals of the original full table
            row[f"orig_{m}_lo"], row[f"orig_{m}_hi"] = interval(full[:, i])  # percentile interval
        row["orig_gain_lo"], row["orig_gain_hi"] = interval(full[:, 0] - full[:, 2])  # gain interval
        row.update({"orig_common_pr_auc": p["oc"]["pr_auc"], "coll_common_pr_auc": p["cc"]["pr_auc"], "orig_common_gain": p["oc"]["gain"], "coll_common_gain": p["cc"]["gain"]})  # common-row points
        row["diff_pr_auc"] = p["cc"]["pr_auc"] - p["oc"]["pr_auc"]  # collapsed minus original
        row["diff_pr_auc_lo"], row["diff_pr_auc_hi"] = interval(draws["c_common"][:, 0] - draws["o_common"][:, 0])  # paired PR-AUC interval
        row["diff_gain"] = p["cc"]["gain"] - p["oc"]["gain"]  # gain difference
        d_gain = (draws["c_common"][:, 0] - draws["c_common"][:, 2]) - (draws["o_common"][:, 0] - draws["o_common"][:, 2])  # paired gain draws
        row["diff_gain_lo"], row["diff_gain_hi"] = interval(d_gain)  # paired gain interval
        out.append(row)  # keep the group
    return out  # rows of this configuration


def verdicts(results: pd.DataFrame) -> pd.DataFrame:
    """Decision rule fixed before running, one row per target.

    Primary: material if the collapsed pooled PR-AUC is outside the original pooled 95% interval (any configuration),
    or if the stage 0 versus final choice (mean dev PR-AUC over folds) differs. Secondary: paired difference interval excludes zero.
    """
    rows = []  # one dict per target
    for target, t in results.groupby("target", sort=False):  # each target
        row = {"target": target}  # verdict row
        for tag, folds in (("", ["fold1", "fold2", "fold3", "fold4", "fold5"]), ("_no_fold5", ["fold1", "fold2", "fold3", "fold4"])):  # with and without the look-ahead fold
            pooled = t[t["group"] == ("pooled" if tag == "" else "pooled_no_fold5")].set_index("config")  # pooled rows by configuration
            outside = (pooled["coll_pr_auc"] < pooled["orig_pr_auc_lo"]) | (pooled["coll_pr_auc"] > pooled["orig_pr_auc_hi"])  # collapsed point outside the original interval
            mean = t[t["group"].isin(folds)].groupby("config")[["orig_pr_auc", "coll_pr_auc"]].mean()  # mean dev PR-AUC over folds
            row[f"outside_ci{tag}"] = ",".join(f"{c}={bool(v)}" for c, v in outside.items())  # per configuration
            row[f"choice_orig{tag}"], row[f"choice_coll{tag}"] = mean["orig_pr_auc"].idxmax(), mean["coll_pr_auc"].idxmax()  # configuration with the higher mean
            row[f"material{tag}"] = bool(outside.any() or row[f"choice_orig{tag}"] != row[f"choice_coll{tag}"])  # primary verdict
            row[f"paired_ci_excludes_zero{tag}"] = ",".join(f"{c}={bool(v)}" for c, v in ((pooled["diff_pr_auc_lo"] > 0) | (pooled["diff_pr_auc_hi"] < 0)).items())  # secondary reading
        rows.append(row)  # keep the target
    return pd.DataFrame(rows)  # verdict table
