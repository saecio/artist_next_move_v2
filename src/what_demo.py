"""Readers and small helpers for the holdout and illustrative sections of notebooks/what_models.ipynb.

Everything here only reads files saved by scripts/run_xgb_what_holdout.py and the cleaned Works table.
Nothing is fitted, scored or recomputed, and the sealed holdout is not evaluated again.
"""

from pathlib import Path  # path handling for the results folder

import numpy as np  # NaN handling and conditional columns
import pandas as pd  # dataframes

# Fixed list of the illustrative section: artists picked by fame and known stylistic reinvention, never by outcome,
# and not changed after seeing scores. The three random artists are a seed-1 draw among artists with at least 3 dated
# Works before 2018 and at least 3 eligible holdout rows. Rosalia is left out on purpose: two artist ids share the name.
NAMED = "named"  # artists picked by name
RANDOM = "random draw (seed 1)"  # artists drawn at random
CAVEAT = "named, 1 eligible row"  # artists with a single eligible holdout row, shown with that caveat
DEMO_ARTISTS = {
    5894905: NAMED, 1124645: NAMED, 2171152: NAMED, 52835: NAMED, 18839: NAMED, 29735: NAMED,  # Bad Bunny, Taylor Swift, The Weeknd, Beyonce, Metallica, Coldplay
    2390402: RANDOM, 2055531: RANDOM, 4341597: RANDOM,  # the seed-1 random artists
    1778977: CAVEAT, 84752: CAVEAT, 251595: CAVEAT,  # Kendrick Lamar, AC/DC, Iron Maiden
}
HOLDOUT_TABLES = ("headline", "by_year", "threshold_summary", "threshold_by_class", "top10_lift")  # saved tables, row_scores.csv is read apart


def read_holdout(results_dir) -> dict:
    """Read the saved tables of the single holdout evaluation, one dataframe per file."""
    folder = Path(results_dir)  # results/holdout_what
    return {name: pd.read_csv(folder / f"{name}.csv") for name in HOLDOUT_TABLES}  # undefined values were saved empty and come back as NaN


def accuracy_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Accuracy next to its two references, with a note where a model flags no row at all."""
    table = summary.copy()  # keep the saved table untouched
    table["note"] = np.where(table["flagged_share"] == 0, "no rows flagged", "")  # a score that never reaches the threshold
    return table


def class_table(by_class: pd.DataFrame) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, with undefined values spelled out."""
    table = by_class.copy()  # keep the saved table untouched
    table["note"] = np.where(table[["precision", "f1"]].isna().any(axis=1), "no rows flagged", "")  # precision and F1 are undefined when no row is flagged
    return table.round(4).astype(object).fillna("undefined")  # NaN printed as the word


def development_range(compare_csv, baseline_c_report: pd.DataFrame, headline: pd.DataFrame) -> pd.DataFrame:
    """Holdout PR-AUC and ROC-AUC next to the lowest and highest value over the five validation folds."""
    folds = pd.read_csv(compare_csv)  # per-fold results of the adoption evaluation, stage 0 columns
    sources = {"stage 0": (folds["stage0_pr_auc"], folds["stage0_roc_auc"]),  # stage 0 per-fold PR-AUC and ROC-AUC
               "baseline (c)": (baseline_c_report["pr_auc"], baseline_c_report["roc_auc"])}  # baseline (c) per-fold values of the gate
    held = headline.set_index("model")  # holdout values by model
    rows = [{"model": model, "validation_pr_auc_min": pr.min(), "validation_pr_auc_max": pr.max(), "holdout_pr_auc": held.loc[model, "pr_auc"],
             "validation_roc_auc_min": roc.min(), "validation_roc_auc_max": roc.max(), "holdout_roc_auc": held.loc[model, "roc_auc"]}
            for model, (pr, roc) in sources.items()]  # one row per model
    return pd.DataFrame(rows)


def illustrative_cases(results_dir, works_path, thresholds: list, artists: dict = DEMO_ARTISTS) -> pd.DataFrame:
    """Saved stage 0 score, true label and flags of the fixed artists, one row per artist-year of the holdout.

    A row is flagged when score >= threshold. The marker uses the lowest threshold (the training positive rate):
    hit (flagged and a new style), missed (not flagged and a new style), false alarm (flagged and no new style),
    correct no (not flagged and no new style). No score is recomputed and nothing is refitted.
    """
    scores = pd.read_csv(Path(results_dir) / "row_scores.csv")  # one row per holdout row, both models
    scores["artist_id"] = scores["artist_id"].astype("int64")  # saved as float, joined as integer
    cases = scores[scores["artist_id"].isin(list(artists))].copy()  # the fixed artists only
    names = pd.read_parquet(works_path, columns=["artist_id", "artist"])  # name column of the cleaned Works table, nothing else
    names = names.dropna(subset=["artist_id"])  # Works without an artist id cannot match a fixed artist
    names["artist_id"] = names["artist_id"].astype("int64")  # float in the Works table too
    names = names[names["artist_id"].isin(list(artists))].drop_duplicates("artist_id")  # one name per artist id
    cases = cases.merge(names, on="artist_id", how="left", validate="m:1")  # attach the artist name
    counts = cases.groupby("artist_id").size()  # eligible holdout rows per artist
    assert set(counts.index) == set(artists), f"artists without holdout rows: {set(artists) - set(counts.index)}"  # every fixed artist has rows
    assert (counts[[a for a, g in artists.items() if g == CAVEAT]] == 1).all(), "a caveat artist does not have exactly one eligible row"  # the caveat is true
    assert cases["artist"].notna().all(), "an artist id has no name in the Works table"  # the join found every name
    cases["group"] = cases["artist_id"].map(artists)  # named, random draw or caveat
    cases["new_style"] = np.where(cases["label"] == 1, "yes", "no")  # true label, shown whatever it is
    for threshold in thresholds:  # one flag column per fixed threshold
        cases[f"flagged at {threshold}"] = np.where(cases["stage0_score"] >= threshold, "yes", "no")  # cutoff rule: score >= threshold
    flagged = cases["stage0_score"] >= min(thresholds)  # flag at the training positive rate
    new = cases["label"] == 1  # a new style did appear
    cases["marker"] = np.select([new & flagged, new & ~flagged, ~new & flagged], ["hit", "missed", "false alarm"], default="correct no")  # agreement of flag and label
    columns = ["group", "artist", "artist_id", "year", "stage0_score", "new_style"] + [f"flagged at {t}" for t in thresholds] + ["marker"]  # display order
    return cases[columns].sort_values(["artist", "year"]).reset_index(drop=True)  # sorted by artist, then year


def group_summary_lines(cases: pd.DataFrame, threshold: float) -> list:
    """One line per artist group, and one for all: new-style artist-years scored above the threshold and flagged without one."""
    flagged = cases["stage0_score"] >= threshold  # cutoff rule: score >= threshold
    new = cases["new_style"] == "yes"  # a new style did appear
    lines = []  # one text line per group
    for group, index in [*cases.groupby("group").groups.items(), ("all fixed artists", cases.index)]:  # each group, then everything
        n_new, n_none = int(new[index].sum()), int((~new[index]).sum())  # artist-years with and without a new style
        lines.append(f"{group}: {cases.loc[index, 'artist_id'].nunique()} artists, {len(index)} artist-years; "
                     f"{int((new & flagged)[index].sum())} of {n_new} with a new style scored at or above {threshold}; "
                     f"{int((~new & flagged)[index].sum())} of {n_none} without one were flagged anyway")  # counts, no percentages
    return lines


# Fixed list of the live demo: artists picked by fame, never by outcome. Rosalia is left out on purpose: two artist ids share the name.
LIVE_ARTISTS = {
    5894905: "Bad Bunny", 52835: "Beyonce", 1124645: "Taylor Swift", 2171152: "The Weeknd", 1967268: "Tyler, The Creator",  # artist id -> label of the list
    1778977: "Kendrick Lamar", 3317315: "Lorde", 1245513: "Tame Impala", 7592689: "Olivia Rodrigo", 4378284: "Sabrina Carpenter",  # the label is only for the printed report
    4110740: "Mitski", 18839: "Metallica", 29735: "Coldplay", 321128: "Rihanna", 2013868: "Frank Ocean",  # the artist name in the CSV comes from the Works table
}
LIVE_MIN_HISTORY = 3  # minimum dated Works before the last active year, as in the WHAT table


def last_active_rows(artist_years: pd.DataFrame, ids: list, min_history: int = LIVE_MIN_HISTORY) -> tuple:
    """Last active year row of each artist, built directly from the artist-year table (the WHAT table drops unlabelled rows).

    Returns the rows with at least `min_history` dated Works before that year, and a list of (artist_id, reason) for the others.
    """
    rows, missing = [], []  # kept rows and the reasons for the others
    for artist_id in ids:  # one artist at a time
        own = artist_years[artist_years["artist_id"] == artist_id]  # all artist-years of the artist
        if own.empty:  # no dated Work at all
            missing.append((artist_id, "no artist-year row"))  # reported, not hidden
            continue  # next artist
        last = own.loc[own["year"].idxmax()]  # the last active year
        if last["history_count"] < min_history:  # too little history for the model
            missing.append((artist_id, f"history_count {int(last['history_count'])} in {int(last['year'])}"))  # reported, not hidden
            continue  # next artist
        rows.append(last)  # keep the row
    return pd.DataFrame(rows).reset_index(drop=True), missing  # rows as a dataframe


def last_holdout_scores(results_dir, ids: list) -> pd.DataFrame:
    """Last holdout year and saved stage 0 score of each artist that has a holdout row, read from row_scores.csv, nothing recomputed."""
    scores = pd.read_csv(Path(results_dir) / "row_scores.csv", usecols=["artist_id", "year", "stage0_score"])  # saved holdout scores
    scores["artist_id"] = scores["artist_id"].astype("int64")  # saved as float, compared as integer
    scores = scores[scores["artist_id"].isin(ids)]  # the fixed artists only
    last = scores.loc[scores.groupby("artist_id")["year"].idxmax()]  # the last holdout year of each artist
    return last.rename(columns={"year": "last_holdout_year", "stage0_score": "last_holdout_score"}).reset_index(drop=True)  # column names of the CSV


def read_live(csv_path, top10_path) -> tuple:
    """Saved live scores sorted by score, with the two reference points: the training rate p and the holdout top 10% cutoff."""
    live = pd.read_csv(csv_path).sort_values("score", ascending=False).reset_index(drop=True)  # highest score first
    p = float(live["training_rate_p"].iloc[0])  # training positive rate, saved by the script
    top = pd.read_csv(top10_path)  # saved top 10% table of the holdout evaluation
    cutoff = float(top.loc[top["model"] == "stage 0", "score_cutoff"].iloc[0])  # stage 0 score at the 90th percentile of the holdout scores
    return live, p, cutoff  # table and reference points
