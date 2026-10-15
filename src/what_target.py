"""
Artist-year version of the WHAT target (new_style).

The Work-level `new_style` in src/snapshots.py compares an artist's next Work
with the styles seen up to the current Work. Because about 57% of those
transitions go to a Work of the SAME year, and the order inside a year is a
seeded pseudo-random order (the data only knows the year), a large part of
that label depends on an ordering that is not real chronology.

This module builds the alternative unit: one row per artist-year. The label
asks whether the artist introduces at least one style, in the Works of their
next active year, that is not in the vocabulary seen through the end of the
current year. Nothing here depends on the order of Works inside a year.

This module does not read any files and does not know about Discogs. It takes
the same Works table as `src.snapshots.build_snapshots` (one row per Work,
shaped like `data/processed/works_checkpoint.parquet`).

Usage:
    from what_target import build_artist_years
    artist_years = build_artist_years(works)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score  # PR-AUC of the descriptive genre view, ROC-AUC of the diagnostics

from backtesting import GAP_YEARS, HOLDOUT_END_YEAR, HOLDOUT_START_YEAR, N_FOLD_TRANCHES, TRANCHE_YEARS, _sorted_inputs, _weighted_average_precision, evaluate_folds, make_what_folds  # shared fold and holdout constants, the WHAT folds and the per-fold evaluation, defined once in the WHEN module
from baselines import KILL_GATE_CONFIDENCE, N_BOOTSTRAP, add_what_buckets, fit_what_baseline, fit_what_lookup_baseline, score_what_baseline, score_what_lookup_baseline  # same bootstrap settings and baselines as the kill gate
from models import GENRES, WHAT_FEATURES, WHAT_TABLE_INPUTS, make_what_xgboost  # fixed genre vocabulary of the WHEN models, and the WHAT reference model
from eda import PRIOR_WORK_BUCKET_LABELS, VOLUME_BUCKET_EDGES, VOLUME_BUCKET_LABELS, _bucket_by_prior_works  # same history and volume buckets as the Work-level EDA
from snapshots import _split_styles, classify_work_type  # same list-to-set conversion and Work type as the Work-level snapshots


def build_artist_years(works: pd.DataFrame) -> pd.DataFrame:
    """Turn a one row per Work table into one row per artist-year.

    Works with no `artist_id` or no known `year` get no row, same as in
    `build_snapshots`. The undated Works of an artist still count, but only
    as a starting vocabulary (see the note below).

    Returns
    -------
    One row per (artist, active year), sorted by artist and year, with columns:

    - `artist_id`, `year`: the artist-year.
    - `next_active_year`: the artist's next year with at least one dated
      Work, missing for the last one.
    - `styles_seen`: set of every style used by the artist in their Works up
      to the end of `year` (all Works of `year` included), plus the styles of
      their undated Works.
    - `next_year_styles`: set of the styles used in the Works of
      `next_active_year`, empty when there is no next active year or those
      Works carry no style tags.
    - `new_style_year`: 1 if `next_year_styles` has at least one style that is
      not in `styles_seen`, 0 if not, missing when `next_year_styles` is empty
      (no next active year, or the next year's Works have no style tags).
    - `history_count`: how many dated Works the artist had strictly before
      `year`, so every Work of `year` is excluded.
    - `n_works_year`: how many dated Works the artist has in `year`.
    - `n_works_next_year`: how many dated Works the artist has in
      `next_active_year`, missing when there is no next active year (nullable
      integer). Not a model input (it describes the future); it is there to
      measure how much of the label rests on a single Work.

    A note on the undated-Works style baseline:

    Same convention as the Work-level `new_style`: every style used on an
    undated Work is folded into the artist's seen styles from their first
    artist-year onward. We cannot know whether an undated Work came before or
    after the dated ones, so treating its styles as already seen avoids a
    common false "new" at the cost of a rare true one.
    """
    # A Work with no artist_id cannot belong to any artist's timeline.
    works = works[works["artist_id"].notna()]
    # A Work with no year cannot be placed on a timeline.
    undated = works["year"].isna()

    # Every style used on any undated Work, one set per artist (the baseline).
    undated_styles_by_artist = works[undated].groupby("artist_id")["styles"].apply(
        lambda styles_column: set().union(*(_split_styles(s) for s in styles_column))
    )

    # From here on only dated Works, with year as a plain integer.
    dated = works[~undated][["artist_id", "year", "styles"]].copy()
    dated["year"] = dated["year"].astype(int)
    # Turn each Work's tag list into a set once, so the union below is cheap.
    dated["styles"] = dated["styles"].map(_split_styles)

    # One row per artist-year: its Work count and the union of its Works' styles.
    artist_years = dated.groupby(["artist_id", "year"]).agg(
        n_works_year=("styles", "size"),
        year_styles=("styles", lambda column: set().union(*column)),
    ).reset_index()  # already sorted by artist_id then year, which the loop below relies on

    # Dated Works strictly before each year: running total minus this year's own count.
    artist_years["history_count"] = (
        artist_years.groupby("artist_id")["n_works_year"].cumsum() - artist_years["n_works_year"]
    )
    # The artist's next active year (missing for the last one), and the next row's styles.
    by_artist = artist_years.groupby("artist_id", sort=False)
    artist_years["next_active_year"] = by_artist["year"].shift(-1)
    next_year_styles = by_artist["year_styles"].shift(-1)
    # Dated Works in that next active year, nullable integer so the last year stays missing.
    artist_years["n_works_next_year"] = by_artist["n_works_year"].shift(-1).astype("Int64")

    # The running vocabulary needs one pass in Python, resetting at each new artist.
    artist_ids = artist_years["artist_id"].to_numpy()
    year_styles = artist_years["year_styles"].tolist()
    next_styles_list = next_year_styles.tolist()
    styles_seen = []  # vocabulary through the end of each artist-year
    new_style_year = np.full(len(artist_years), np.nan)
    current_artist = None
    seen: set = set()
    for i in range(len(artist_years)):
        # A new artist starts from their undated-Works styles, not from empty.
        if artist_ids[i] != current_artist:
            current_artist = artist_ids[i]
            seen = set(undated_styles_by_artist.get(current_artist, set()))
        # Every style of this year is known at the end of it, so it counts as seen.
        seen |= year_styles[i]
        # A copy, because `seen` keeps growing on the next iteration.
        styles_seen.append(set(seen))
        # Missing next row (last year of the artist) is NaN, a real set otherwise.
        next_set = next_styles_list[i] if isinstance(next_styles_list[i], set) else set()
        # Label only when the next year has styles, otherwise "new" is unknowable.
        if next_set:
            new_style_year[i] = float(len(next_set - seen) > 0)
    artist_years["styles_seen"] = styles_seen
    artist_years["next_year_styles"] = [s if isinstance(s, set) else set() for s in next_styles_list]
    artist_years["new_style_year"] = new_style_year

    # Column order: identity, features, label.
    return artist_years[[
        "artist_id", "year", "next_active_year", "styles_seen", "next_year_styles",
        "new_style_year", "history_count", "n_works_year", "n_works_next_year",
    ]]


def _assert_sorted_by_artist_year(work_snapshots: pd.DataFrame) -> None:
    """Stop if the Work-level snapshots are not in file order by artist and then year.

    Several functions here use the next or last row of an artist, which is only
    meaningful when the rows of one artist are consecutive and sorted by year.
    """
    # Row-to-row differences of the two sort keys.
    d_artist = work_snapshots["artist_id"].diff()
    d_year = work_snapshots["year"].diff()
    # A row is in order when it opens the file, starts a later artist, or has a year not before the previous one.
    in_order = d_artist.isna() | (d_artist > 0) | ((d_artist == 0) & (d_year >= 0))
    assert in_order.all(), "work_snapshots is not sorted by artist_id then year"


def crosscheck_levels(work_snapshots: pd.DataFrame, artist_years: pd.DataFrame, holdout_start: int = HOLDOUT_START_YEAR) -> pd.DataFrame:
    """Check that the artist-year label is consistent with the Work-level label.

    `work_snapshots` is `data/processed/snapshots.parquet` (needs `artist_id`,
    `year` and `new_style`, in file order, which is the seeded within-year
    order). `artist_years` is the output of `build_artist_years`. The last Work
    of each artist-year (in file order) is joined to its artist-year row, and
    only development years (`year` < `holdout_start`) with both labels present
    are compared.

    Returns one row per check with `check`, `rows` (the population the check
    runs on), `count` (rows that meet the condition) and `share` (count / rows):

    - `dev_last_work_rows`: last Works of development years (count = rows).
    - `dropped_only_new_style_nan`, `dropped_only_new_style_year_nan`,
      `dropped_both_nan`: rows left out of the comparison, by which label is NaN.
    - `implication`: rows with Work-level 1; count = those where the
      artist-year label is 0 (expected 0).
    - `equality_single_next_work`: rows whose next active year has exactly one
      Work; count = mismatches between the two labels (expected 0).
    - `info_next_2`, `info_next_3_4`, `info_next_5_plus`: rows by number of
      Works in the next active year (2, 3-4, 5+); count = those with
      artist-year 1 and Work-level 0 (what the Work-level label cannot see).

    If the implication or the equality has any violation, up to 5 example rows
    are stored in `result.attrs["examples"]` (empty frame otherwise), with
    both labels and the artist-year style sets. Nothing is fixed here.
    """
    # The last-Work logic relies on file order, so check it is sorted by artist then year.
    _assert_sorted_by_artist_year(work_snapshots)

    # Development years only, and the last Work of each artist-year in file order.
    development = work_snapshots[work_snapshots["year"] < holdout_start]
    last_work = development.drop_duplicates(["artist_id", "year"], keep="last")[["artist_id", "year", "new_style"]]
    # One artist-year row per last Work; validate and the length check catch any unmatched or duplicated key.
    joined = last_work.merge(artist_years, on=["artist_id", "year"], how="inner", validate="one_to_one")
    assert len(joined) == len(last_work), "some last Works have no artist-year row"

    # Which label is missing on each row, to report the dropped rows by cause.
    work_nan = joined["new_style"].isna()
    year_nan = joined["new_style_year"].isna()
    # Rows where both labels exist are the only ones compared.
    both = joined[~work_nan & ~year_nan]
    # Work counts of the next year are never missing here (a label needs a next year); plain ints for comparisons.
    next_n = both["n_works_next_year"].astype(int)

    # Check 1: Work-level 1 must imply artist-year 1; violations are the 1 -> 0 rows.
    work_one = both["new_style"] == 1
    implication_violation = work_one & (both["new_style_year"] == 0)
    # Check 2: with a single Work in the next year the two labels must be identical.
    single = next_n == 1
    equality_violation = single & (both["new_style"] != both["new_style_year"])
    # Check 3: artist-year 1 with Work-level 0, by size of the next active year (2, 3-4, 5+).
    hidden = (both["new_style_year"] == 1) & (both["new_style"] == 0)
    bucket = pd.cut(next_n, bins=[1, 2, 4, np.inf], labels=["2", "3_4", "5_plus"])

    # Each row of the report: name, population size, rows meeting the condition.
    n_dev = len(joined)
    rows = [
        ("dev_last_work_rows", n_dev, n_dev),
        ("dropped_only_new_style_nan", n_dev, int((work_nan & ~year_nan).sum())),
        ("dropped_only_new_style_year_nan", n_dev, int((~work_nan & year_nan).sum())),
        ("dropped_both_nan", n_dev, int((work_nan & year_nan).sum())),
        ("implication", int(work_one.sum()), int(implication_violation.sum())),
        ("equality_single_next_work", int(single.sum()), int(equality_violation.sum())),
    ]
    # The informational rows, one per bucket, in the same (name, population, count) shape.
    for label in ["2", "3_4", "5_plus"]:
        in_bucket = bucket == label
        rows.append((f"info_next_{label}", int(in_bucket.sum()), int((in_bucket & hidden).sum())))
    result = pd.DataFrame(rows, columns=["check", "rows", "count"])
    # Share of the population that meets the condition (0 when the population is empty).
    result["share"] = np.where(result["rows"] > 0, result["count"] / result["rows"].where(result["rows"] > 0), 0.0)

    # Examples of any violation of check 1 or 2, left for inspection, never fixed.
    violations = both[implication_violation | equality_violation]
    result.attrs["examples"] = violations[
        ["artist_id", "year", "new_style", "new_style_year", "n_works_next_year", "styles_seen", "next_year_styles"]
    ].head(5)
    return result


def _window_row(table: pd.DataFrame, first_year: int, last_year: int, sealed: bool) -> dict:
    """Eligibility and label statistics of one level on one window of years.

    `table` has `artist_id`, `year` and `label`, and only holds rows that
    already passed the history rule. When `sealed` is True only the row and
    artist counts are computed, the label column is never read.
    """
    # Rows whose year falls inside the window, both ends included.
    in_window = table[(table["year"] >= first_year) & (table["year"] <= last_year)]
    # Counts that do not look at any label.
    row = {"eligible_rows": len(in_window), "eligible_artists": in_window["artist_id"].nunique()}
    # A sealed window (the holdout) reports nothing about the label.
    if sealed:
        return {**row, "nan_share": np.nan, "labelled_rows": np.nan, "prevalence": np.nan}
    # Labelled rows are the ones where the label is not NaN.
    labelled = in_window["label"].dropna()
    # NaN share and prevalence are undefined (NaN) on an empty population.
    row["nan_share"] = 1 - len(labelled) / len(in_window) if len(in_window) > 0 else np.nan
    row["labelled_rows"] = len(labelled)
    row["prevalence"] = labelled.mean() if len(labelled) > 0 else np.nan
    return row


def prevalence_by_fold(work_snapshots: pd.DataFrame, artist_years: pd.DataFrame, min_previous_works: int = 3, holdout_start: int = HOLDOUT_START_YEAR) -> pd.DataFrame:
    """Compare the Work-level and the artist-year label on the same windows.

    `work_snapshots` is `data/processed/snapshots.parquet` (needs `artist_id`,
    `year`, `history_count` and `new_style`); `artist_years` is the output of
    `build_artist_years`. Both levels use the same rule: a row is eligible when
    its own `history_count` (Works before it for the Work level, dated Works
    strictly before the year for the artist-year level) is at least
    `min_previous_works`.

    Windows: `development` (every year before `holdout_start`), the five test
    tranches of the WHEN folds, and `holdout` (`holdout_start` to
    `HOLDOUT_END_YEAR`). The holdout row only reports `eligible_rows` and
    `eligible_artists`; its label is never read, so the window stays sealed.

    Returns a dataframe indexed by (`window`, `level`), with `level` either
    `work` (label `new_style`) or `artist_year` (label `new_style_year`), and
    columns `eligible_rows`, `eligible_artists`, `nan_share` (share of
    eligible rows with a NaN label), `labelled_rows` and `prevalence` (mean of
    the label among labelled rows).
    """
    # Each level reduced to the same three columns, already filtered by the history rule.
    levels = {
        "work": work_snapshots.rename(columns={"new_style": "label"}),
        "artist_year": artist_years.rename(columns={"new_style_year": "label"}),
    }
    levels = {
        name: table.loc[table["history_count"] >= min_previous_works, ["artist_id", "year", "label"]]
        for name, table in levels.items()
    }

    # Development as a whole, then each fold tranche, as (name, first year, last year, sealed).
    first_tranche = holdout_start - N_FOLD_TRANCHES * TRANCHE_YEARS
    windows = [("development", int(levels["work"]["year"].min()), holdout_start - 1, False)]
    for start in range(first_tranche, holdout_start, TRANCHE_YEARS):
        windows.append((f"{start}-{start + TRANCHE_YEARS - 1}", start, start + TRANCHE_YEARS - 1, False))
    # The sealed holdout window comes last.
    windows.append((f"holdout {holdout_start}-{HOLDOUT_END_YEAR}", holdout_start, HOLDOUT_END_YEAR, True))

    # One row per (window, level), in window order, work level first.
    records = []
    for name, first_year, last_year, sealed in windows:
        for level, table in levels.items():
            records.append({"window": name, "level": level, **_window_row(table, first_year, last_year, sealed)})
    return pd.DataFrame(records).set_index(["window", "level"])


def prevalence_by_history_bucket(work_snapshots: pd.DataFrame, artist_years: pd.DataFrame, min_previous_works: int = 3, holdout_start: int = HOLDOUT_START_YEAR) -> pd.DataFrame:
    """Compare how the label rate falls with accumulated history at both levels.

    Inputs and the eligibility rule are the same as in `prevalence_by_fold`.
    Only development years (`year` < `holdout_start`) are used, so the label of
    the holdout is never read. Buckets are the prior-work buckets of
    src/eda.py (`PRIOR_WORK_BUCKET_LABELS`, bucketed by the same helper, with
    `history_count` playing the role of `prior_work_count`).

    Returns a dataframe indexed by (`bucket`, `level`), `level` being `work` or
    `artist_year`, with columns `eligible_rows`, `row_share` (share of the
    level's eligible development rows), `nan_share` (share of the bucket's rows
    with a NaN label), `labelled_rows` and `prevalence`.
    """
    # Each level reduced to the columns the eda helper and the statistics need.
    levels = {
        "work": work_snapshots.rename(columns={"new_style": "label"}),
        "artist_year": artist_years.rename(columns={"new_style_year": "label"}),
    }
    records = []
    stats = {}
    for name, table in levels.items():
        # Development years only, before anything touches the label.
        development = table[table["year"] < holdout_start]
        # The eda helper expects `prior_work_count`; it also applies the history rule.
        bucketed = _bucket_by_prior_works(development.rename(columns={"history_count": "prior_work_count"}), min_previous_works)
        # One group per bucket; observed=False keeps an empty bucket as a row of zeros.
        grouped = bucketed.groupby("prior_work_bucket", observed=False)["label"]
        # size counts every row, count only the non-NaN labels, mean is the prevalence.
        stats[name] = pd.DataFrame({"eligible_rows": grouped.size(), "labelled_rows": grouped.count(), "prevalence": grouped.mean()})
        # Share of the level's eligible rows, and share of the bucket's rows without a label.
        stats[name]["row_share"] = stats[name]["eligible_rows"] / stats[name]["eligible_rows"].sum()
        stats[name]["nan_share"] = 1 - stats[name]["labelled_rows"] / stats[name]["eligible_rows"].where(stats[name]["eligible_rows"] > 0)
    # One row per (bucket, level), buckets in ascending order, work level first.
    for bucket in PRIOR_WORK_BUCKET_LABELS:
        for name in levels:
            records.append({"bucket": bucket, "level": name, **stats[name].loc[bucket].to_dict()})
    return pd.DataFrame(records).set_index(["bucket", "level"])[
        ["eligible_rows", "row_share", "nan_share", "labelled_rows", "prevalence"]
    ]


def prevalence_by_next_year_volume(artist_years: pd.DataFrame, min_previous_works: int = 3, holdout_start: int = HOLDOUT_START_YEAR) -> dict:
    """Measure how much the artist-year label depends on next-year volume.

    `n_works_next_year` is unknown at prediction time, so it is used only as a
    diagnostic. Population: artist-year rows of development years (`year` <
    `holdout_start`, filtered before any label or volume column is read) with
    `history_count` >= `min_previous_works` and a non-NaN `new_style_year`.

    Returns a dict of three dataframes:

    - `by_volume`: per bucket of `n_works_next_year` (1, 2, 3-4, 5+),
      `labelled_rows`, `row_share` and `prevalence`.
    - `cross`: rows are the prior-work buckets of src/eda.py, columns are
      (`labelled_rows` | `prevalence`, volume bucket).
    - `volume_predictability`: per bucket of `n_works_year` (known at the
      cutoff), `rows` and `share_next_year_single` (share of rows with exactly
      one Work in the next active year); an `all` row holds the Spearman
      correlation between `n_works_year` and `n_works_next_year` in `spearman`.
    """
    # Development years only, before any other column is read.
    development = artist_years[artist_years["year"] < holdout_start]
    # History rule and prior-work buckets from the eda helper, same as the history-bucket function.
    bucketed = _bucket_by_prior_works(development.rename(columns={"history_count": "prior_work_count"}), min_previous_works)
    # Only labelled rows: the label is the thing being diagnosed.
    rows = bucketed[bucketed["new_style_year"].notna()].copy()
    # Plain ints for the volumes (a labelled row always has a next year), then their buckets.
    rows["n_next"] = rows["n_works_next_year"].astype(int)
    rows["volume_bucket"] = pd.cut(rows["n_next"], bins=VOLUME_BUCKET_EDGES, labels=VOLUME_BUCKET_LABELS)
    rows["volume_now_bucket"] = pd.cut(rows["n_works_year"], bins=VOLUME_BUCKET_EDGES, labels=VOLUME_BUCKET_LABELS)

    # 1. Label rate by next-year volume.
    by_volume = rows.groupby("volume_bucket", observed=False)["new_style_year"].agg(labelled_rows="size", prevalence="mean")
    by_volume["row_share"] = by_volume["labelled_rows"] / by_volume["labelled_rows"].sum()
    by_volume = by_volume[["labelled_rows", "row_share", "prevalence"]]

    # 2. Same rates with history held fixed; observed=False keeps empty cells.
    cells = rows.groupby(["prior_work_bucket", "volume_bucket"], observed=False)["new_style_year"]
    cross = pd.concat({"labelled_rows": cells.size().unstack(), "prevalence": cells.mean().unstack()}, axis=1)

    # 3. How well the volume at the cutoff predicts a single-Work next year.
    single = (rows["n_next"] == 1).groupby(rows["volume_now_bucket"], observed=False)
    predictability = pd.DataFrame({"rows": single.size(), "share_next_year_single": single.mean()})
    # Spearman over all rows, stored on its own row because it is not per bucket.
    predictability.loc["all"] = [len(rows), (rows["n_next"] == 1).mean()]
    predictability["spearman"] = np.nan
    predictability.loc["all", "spearman"] = rows["n_works_year"].corr(rows["n_next"], method="spearman")
    return {"by_volume": by_volume, "cross": cross, "volume_predictability": predictability}


def threshold_sensitivity(artist_years: pd.DataFrame, thresholds: tuple = (2, 3, 5), holdout_start: int = HOLDOUT_START_YEAR) -> pd.DataFrame:
    """Check how the artist-year label changes with the minimum history rule.

    Only development years (`year` < `holdout_start`) are used, the table is
    cut before any history or label column is read. For each threshold, a row
    is eligible when `history_count` >= threshold.

    Returns a dataframe indexed by `min_previous_works` with columns
    `eligible_rows`, `eligible_artists`, `artists_removed` (development artists
    with no eligible row, against all artists with a development artist-year),
    `nan_share`, `labelled_rows`, `prevalence`, and one `prevalence_<first>_<last>`
    column per WHEN test tranche (the same five tranches as `prevalence_by_fold`).
    """
    # Development years only, before any other column is read.
    development = artist_years[artist_years["year"] < holdout_start]
    # Every artist with at least one development artist-year, the reference for "removed".
    all_artists = development["artist_id"].nunique()
    # The five WHEN test tranches, as (first year, last year).
    first_tranche = holdout_start - N_FOLD_TRANCHES * TRANCHE_YEARS
    tranches = [(start, start + TRANCHE_YEARS - 1) for start in range(first_tranche, holdout_start, TRANCHE_YEARS)]

    records = []
    for threshold in thresholds:
        # Eligible rows for this threshold, in the (artist_id, year, label) shape the window helper expects.
        eligible = development.loc[development["history_count"] >= threshold, ["artist_id", "year", "new_style_year"]]
        eligible = eligible.rename(columns={"new_style_year": "label"})
        # All development years at once (first year 0 is below any real year).
        row = _window_row(eligible, 0, holdout_start - 1, sealed=False)
        row = {"min_previous_works": threshold, **row, "artists_removed": all_artists - row["eligible_artists"]}
        # Prevalence on each test tranche, to see stability over time.
        for first_year, last_year in tranches:
            row[f"prevalence_{first_year}_{last_year}"] = _window_row(eligible, first_year, last_year, sealed=False)["prevalence"]
        records.append(row)
    # Tidy column order: sizes first, then label statistics, then the tranches.
    result = pd.DataFrame(records).set_index("min_previous_works")
    return result[["eligible_rows", "eligible_artists", "artists_removed", "nan_share", "labelled_rows", "prevalence"]
                  + [c for c in result.columns if c.startswith("prevalence_")]]


# Columns of the WHAT modeling table that are known at the cutoff or are plain identifiers.
WHAT_TABLE_COLUMNS = ["artist_id", "year", "history_count", "n_works_year", "next_active_year", "new_style_year"]
# Columns that describe the future (the next active year), so they are diagnostics only and never model inputs.
DIAGNOSTIC_COLUMNS = ["n_works_next_year"]


def build_what_table(artist_years: pd.DataFrame, min_previous_works: int = 3, include_diagnostics: bool = False) -> pd.DataFrame:
    """Select the rows WHAT is trained and evaluated on, one row per artist-year.

    `artist_years` is the output of `build_artist_years`. A row is kept when
    `history_count` (dated Works strictly before the year) is at least
    `min_previous_works` and `new_style_year` is not NaN. All years are kept,
    splitting by year (development, holdout, recent) happens later, and no
    label statistic is computed here.

    Columns: `WHAT_TABLE_COLUMNS`. With `include_diagnostics=True` the columns of
    `DIAGNOSTIC_COLUMNS` (`n_works_next_year`, unknown at the cutoff) are added
    for analysis only; they must never reach a model. The style sets
    (`styles_seen`, `next_year_styles`) are never included.
    """
    # Enough previous Works, and a label that exists.
    keep = (artist_years["history_count"] >= min_previous_works) & artist_years["new_style_year"].notna()
    # The diagnostic column is added only when explicitly requested.
    columns = WHAT_TABLE_COLUMNS + (DIAGNOSTIC_COLUMNS if include_diagnostics else [])
    # Fresh 0..n-1 index, so the result does not depend on the row numbers of the input.
    return artist_years.loc[keep, columns].reset_index(drop=True)


def same_year_transition_share(work_snapshots: pd.DataFrame, min_previous_works: int = 3, holdout_start: int = HOLDOUT_START_YEAR) -> pd.DataFrame:
    """Measure how many Work-level transitions go to a Work of the same year.

    `work_snapshots` is `data/processed/snapshots.parquet` (needs `artist_id`,
    `year`, `history_count` and `new_style`, in file order). A transition is a
    row with `history_count` >= `min_previous_works` whose `new_style` is
    present, so the same population the Work-level label is built on. Only the
    presence of `new_style` is used to select rows, its values are never read,
    so the figures describe the structure of the data and are not a label
    statistic. For such a row the next Work is the next row of the same artist.

    Returns two rows, `development (year < holdout_start)` and `all years`, with
    `transitions`, `to_same_year` (next Work in the same year as the row) and
    `same_year_share`.
    """
    # The next row of an artist is only the next Work when the file is in order.
    _assert_sorted_by_artist_year(work_snapshots)
    # Year of the artist's next row (missing for an artist's last row).
    next_year = work_snapshots.groupby("artist_id", sort=False)["year"].shift(-1)
    # Transitions: enough history and a label present (presence only, values are never used).
    eligible = (work_snapshots["history_count"] >= min_previous_works) & work_snapshots["new_style"].notna()
    # Whether the next Work falls in the same year as the row.
    same_year = next_year == work_snapshots["year"]
    # Development years and all years, as two labelled populations.
    populations = {
        f"development (year < {holdout_start})": eligible & (work_snapshots["year"] < holdout_start),
        "all years": eligible,
    }
    # One row per population: how many transitions, and how many go to the same year.
    rows = [(name, int(mask.sum()), int((mask & same_year).sum())) for name, mask in populations.items()]
    result = pd.DataFrame(rows, columns=["population", "transitions", "to_same_year"]).set_index("population")
    # Share of the transitions that go to a Work of the same year.
    result["same_year_share"] = result["to_same_year"] / result["transitions"]
    return result


# Genre tags that are a data artifact, not a genre (8 Works carry only the tag "1"); never used as a primary genre.
FAKE_GENRE_TAGS = ("1",)
# A genre needs at least this many eligible rows to get its own row in the genre breakdown (smaller groups are too noisy to compare).
MIN_GENRE_ROWS = 1000
# Quantiles of the extra delay (years) between the immediate next active year and the next year with style tags.
DELAY_QUANTILES = [0.5, 0.75, 0.9, 0.99]


def _loss_shares(flags: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    """Rows and shares of the two label-loss flags for each group of `key`.

    `flags` has one row per eligible artist-year with the boolean columns
    `no_styles` (label missing because the next year has no style tags) and
    `recoverable` (that, and a later year with style tags exists).
    """
    # Group size, then the mean of each boolean column is its share of the group.
    grouped = flags.groupby(key, observed=False)
    return pd.DataFrame({
        "eligible_rows": grouped.size(),
        "cause_c_share": grouped["no_styles"].mean(),
        "recovered_share": grouped["recoverable"].mean(),
    })


def _primary_genre(works: pd.DataFrame) -> pd.Series:
    """Primary genre of each artist: the most frequent genre tag across their dated Works.

    `FAKE_GENRE_TAGS` are removed first, and ties are broken alphabetically
    (asserted). Returns a series indexed by `artist_id`; an artist with no valid
    tag left has no entry.
    """
    # Dated Works of artists that exist, with one row per genre tag.
    dated = works[works["artist_id"].notna() & works["year"].notna()][["artist_id", "genres"]]
    tags = dated.explode("genres").dropna(subset=["genres"])
    tags = tags[~tags["genres"].isin(FAKE_GENRE_TAGS)]
    # Tag counts per artist, most frequent first, alphabetical among equal counts; the first row of each artist is its primary genre.
    counts = tags.groupby(["artist_id", "genres"]).size().rename("n").reset_index()
    ordered = counts.sort_values(["artist_id", "n", "genres"], ascending=[True, False, True])
    primary = ordered.drop_duplicates("artist_id").set_index("artist_id")["genres"]
    # Independent check of the tie-break: the alphabetically first of the tags with the maximum count.
    top = counts[counts["n"] == counts.groupby("artist_id")["n"].transform("max")]
    assert primary.equals(top.groupby("artist_id")["genres"].min().reindex(primary.index)), "primary genre tie-break is not alphabetical"
    return primary


def label_loss_breakdown(works: pd.DataFrame, artist_years: pd.DataFrame, min_previous_works: int = 3, holdout_start: int = HOLDOUT_START_YEAR) -> dict:
    """Explain why eligible artist-years lose their label and what a tolerant definition would recover.

    Population: artist-years with `year` < `holdout_start` and `history_count`
    >= `min_previous_works` (the set whose labelled rows are the WHAT table).
    `artist_years` is the output of `build_artist_years`; `works` is the Works
    table and is used only to get each artist's primary genre. A development
    row can have its next styled year after `holdout_start` (as the current
    label of a 2017 row already does); such rows are counted, but their label is
    left out of every prevalence so no statistic uses style tags of holdout years.

    The label of a row is missing for one of two reasons besides being present:
    (b) there is no next active year, or (c) the next active year exists but its
    Works carry no style tags. For (c) rows, the next styled year is the first
    later active year whose Works carry at least one style tag; the label such a
    row would get is 1 if that year has a style absent from `styles_seen`, the
    same convention as the current target.

    Returns a dict of dataframes:
    - `causes`: rows and share of the eligible population that are labelled,
      without next active year, or with a next year without styles (the three add
      up to `eligible_total`, asserted).
    - `recovery`: for the (c) rows, rows recoverable and not recoverable, rows
      whose next styled year is at or after `holdout_start` (counted, label
      left out), rows whose label is used, quantiles of the extra delay in years
      (years only), and the label prevalence of the recovered rows used next to
      the currently labelled rows and to both together.
    - `by_window`: eligible rows, (c) share and recovered share for the five WHEN
      test tranches and for all earlier development years.
    - `by_genre`: the same by primary genre (most frequent genre tag over the
      artist's dated Works without `FAKE_GENRE_TAGS`, ties broken alphabetically,
      asserted), for genres with at least `MIN_GENRE_ROWS` eligible rows, after
      the row `all genres`. Artists left with no valid genre form the row
      `(no valid genre)`, always shown; their counts are in `attrs`.
    - `by_history_bucket`: the same by prior-work bucket of src/eda.py.
    """
    # Positions and per-artist shifts below need the rows sorted by artist then year.
    _assert_sorted_by_artist_year(artist_years)
    # Fresh 0..n-1 index, so a row number and an index label are the same thing.
    rows = artist_years.reset_index(drop=True)
    artist = rows["artist_id"]

    # Whether the year of the NEXT row has style tags; the own year of a row has them if the previous row says so.
    next_has_styles = rows["next_year_styles"].map(bool)
    own_has_styles = next_has_styles.groupby(artist).shift(1, fill_value=False)
    # Row number of each row whose own year has style tags (missing otherwise).
    position = pd.Series(np.arange(len(rows)))
    styled_position = position.where(own_has_styles)
    # First styled year strictly after each row's year: the next row's value, filled backwards inside the artist.
    next_styled_position = styled_position.groupby(artist).shift(-1).groupby(artist).bfill()

    # The three mutually exclusive outcomes of the label.
    labelled = rows["new_style_year"].notna()
    no_next = rows["next_active_year"].isna()
    no_styles = ~no_next & ~labelled
    # Eligible development rows, the population of every table.
    eligible = (rows["year"] < holdout_start) & (rows["history_count"] >= min_previous_works)
    # A (c) row is recoverable when some later active year has style tags.
    recoverable = no_styles & next_styled_position.notna()

    # 1. Causes; the three must add up to the eligible total.
    cause_rows = {
        "labelled": int((eligible & labelled).sum()),
        "no_next_active_year": int((eligible & no_next).sum()),
        "next_year_without_styles": int((eligible & no_styles).sum()),
    }
    eligible_total = int(eligible.sum())
    assert sum(cause_rows.values()) == eligible_total, "causes do not add up to the eligible rows"
    causes = pd.DataFrame({"rows": {**cause_rows, "eligible_total": eligible_total}})
    causes["share"] = causes["rows"] / eligible_total

    # 2. Recovery: the label each recoverable (c) row would get, and the extra delay.
    seen_sets = rows["styles_seen"].tolist()
    # Styles of the year of row j are the next_year_styles of row j - 1 (always the same artist for a later year).
    styles_of_year = [set()] + rows["next_year_styles"].tolist()[:-1]
    years = rows["year"].to_numpy()
    next_active = rows["next_active_year"].to_numpy()
    recovered_label = []
    delay = []
    styled_years = []
    for i in np.flatnonzero((eligible & recoverable).to_numpy()):
        # Row number of the next styled year, then its styles against the vocabulary seen through the current year.
        j = int(next_styled_position[i])
        # A styled year at or after holdout_start would put holdout style tags into a label statistic, so its label is left out (NaN).
        recovered_label.append(float(len(styles_of_year[j] - seen_sets[i]) > 0) if years[j] < holdout_start else np.nan)
        # Years between the immediate next active year and the next styled year.
        delay.append(years[j] - next_active[i])
        styled_years.append(years[j])
    recovered_label = pd.Series(recovered_label, dtype=float)
    delay = pd.Series(delay, dtype=float)
    # Recovered rows whose label is used: the next styled year is a development year.
    used_label = recovered_label.dropna()
    labelled_prevalence = float(rows.loc[eligible & labelled, "new_style_year"].mean())
    n_c = cause_rows["next_year_without_styles"]
    recovery = {
        "cause_c_rows": n_c,
        "recoverable_rows": len(recovered_label),
        "not_recoverable_rows": n_c - len(recovered_label),
        "recoverable_styled_year_at_or_after_holdout_start": int(sum(y >= holdout_start for y in styled_years)),
        "recoverable_rows_in_prevalence": len(used_label),
    }
    for q in DELAY_QUANTILES:
        recovery[f"extra_delay_years_p{int(q * 100)}"] = float(delay.quantile(q)) if len(delay) else np.nan
    recovery["prevalence_recovered_rows"] = float(used_label.mean()) if len(used_label) else np.nan
    recovery["prevalence_currently_labelled"] = labelled_prevalence
    # Prevalence over the labelled rows and the recovered rows whose label is used, together.
    labelled_n = cause_rows["labelled"]
    recovery["prevalence_labelled_plus_recovered"] = (labelled_prevalence * labelled_n + used_label.sum()) / (labelled_n + len(used_label))
    recovery = pd.DataFrame({"value": recovery})

    # Flags of the eligible rows only, shared by the three breakdowns below.
    flags = pd.DataFrame({"no_styles": no_styles, "recoverable": recoverable})[eligible]

    # 3. By window: the five WHEN test tranches and everything before the first one.
    first_tranche = holdout_start - N_FOLD_TRANCHES * TRANCHE_YEARS
    edges = [0] + [first_tranche - 1 + k * TRANCHE_YEARS for k in range(N_FOLD_TRANCHES + 1)]
    labels = [f"before {first_tranche}"] + [f"{edges[k] + 1}-{edges[k + 1]}" for k in range(1, N_FOLD_TRANCHES + 1)]
    window = pd.cut(rows.loc[eligible, "year"], bins=edges, labels=labels)
    by_window = _loss_shares(flags, window)

    # 4. By primary genre, from the dated Works of each artist, without the fake tags.
    primary = _primary_genre(works)
    # Genre of each eligible row; artists with no valid genre left get their own labelled group.
    genre = rows.loc[eligible, "artist_id"].map(primary).fillna("(no valid genre)")
    by_genre = _loss_shares(flags, genre)
    # Genres with enough rows, largest first; the no-valid-genre group is kept whatever its size, and goes last.
    no_genre = by_genre.loc[["(no valid genre)"]] if "(no valid genre)" in by_genre.index else by_genre.iloc[0:0]
    kept = by_genre.drop(index="(no valid genre)", errors="ignore")
    kept = kept[kept["eligible_rows"] >= MIN_GENRE_ROWS].sort_values("eligible_rows", ascending=False)
    overall = pd.DataFrame({"eligible_rows": [len(flags)], "cause_c_share": [flags["no_styles"].mean()], "recovered_share": [flags["recoverable"].mean()]}, index=["all genres"])
    by_genre = pd.concat([overall, kept, no_genre])
    # Artists left without a valid genre, among all artists with dated Works and among the eligible ones.
    all_artists = works.loc[works["artist_id"].notna() & works["year"].notna(), "artist_id"].unique()
    eligible_artists = rows.loc[eligible, "artist_id"].unique()
    by_genre.attrs["artists_without_valid_genre"] = {
        "all_artists": int((~pd.Series(all_artists).isin(primary.index)).sum()),
        "eligible_artists": int((~pd.Series(eligible_artists).isin(primary.index)).sum()),
    }

    # 5. By prior-work bucket, with the eda helper (it also applies the history rule, a no-op here).
    bucket_input = flags.assign(prior_work_count=rows.loc[eligible, "history_count"])
    bucketed = _bucket_by_prior_works(bucket_input, min_previous_works)
    by_history_bucket = _loss_shares(bucketed[["no_styles", "recoverable"]], bucketed["prior_work_bucket"])

    return {"causes": causes, "recovery": recovery, "by_window": by_window, "by_genre": by_genre, "by_history_bucket": by_history_bucket}


def example_artist_years(artist_years: pd.DataFrame, min_previous_works: int = 3, holdout_start: int = HOLDOUT_START_YEAR) -> pd.DataFrame:
    """Return the artist-year rows of one real example artist, chosen by a fixed rule.

    `artist_years` is the output of `build_artist_years`. The example is the
    first `artist_id` (in ascending order) that meets all of:
    - 5 or 6 active years in total;
    - every active year before `holdout_start`, so no row, and no style set,
      comes from the holdout years (the artist is picked from development
      years only);
    - at least one row that is eligible and labelled (`history_count` >=
      `min_previous_works` and `new_style_year` not NaN), the rows that
      `build_what_table` keeps;
    - at least one such row with `new_style_year` = 1.
    The choice is deterministic, so the same example is shown on every run.

    Returns that artist's rows in year order with `artist_id`, `year`,
    `history_count`, `n_works_year`, `styles_seen` and `next_year_styles` (the
    style sets as sorted, comma-separated text), `next_active_year`,
    `new_style_year` and `in_what_table` (True for the rows that survive the
    WHAT table filter).
    """
    # Only artists whose last active year is a development year; later years are never looked at again.
    last_year = artist_years.groupby("artist_id")["year"].transform("max")
    pool = artist_years[last_year < holdout_start].copy()
    # The rows the WHAT table keeps: enough previous Works and a label that exists.
    pool["in_what_table"] = (pool["history_count"] >= min_previous_works) & pool["new_style_year"].notna()
    # A kept row whose label is a positive.
    pool["kept_positive"] = pool["in_what_table"] & (pool["new_style_year"] == 1)
    # One row per artist: active years, and whether some kept row exists and some kept row is a positive.
    per_artist = pool.groupby("artist_id").agg(
        active_years=("year", "size"),
        has_kept_row=("in_what_table", "any"),
        has_kept_positive=("kept_positive", "any"),
    )
    # The rule: 5 to 6 active years, a row in the WHAT table, and a positive among those rows.
    meets_rule = per_artist["active_years"].between(5, 6) & per_artist["has_kept_row"] & per_artist["has_kept_positive"]
    assert meets_rule.any(), "no artist meets the example rule"
    # Smallest artist_id among those that meet it (the index is sorted by the groupby).
    chosen = meets_rule[meets_rule].index[0]
    rows = pool[pool["artist_id"] == chosen].copy()
    # Style sets as sorted text, so the table reads and prints well.
    rows["styles_seen"] = rows["styles_seen"].map(lambda styles: ", ".join(sorted(styles)))
    rows["next_year_styles"] = rows["next_year_styles"].map(lambda styles: ", ".join(sorted(styles)))
    return rows[["artist_id", "year", "history_count", "n_works_year", "styles_seen", "next_active_year", "next_year_styles", "new_style_year", "in_what_table"]].reset_index(drop=True)


def fixed_gap_comparison(table: pd.DataFrame, n_folds: int = N_FOLD_TRANCHES, test_years: int = TRANCHE_YEARS, gap_years: int = GAP_YEARS, holdout_start: int = HOLDOUT_START_YEAR) -> pd.DataFrame:
    """Show, per fold, what the exact label-resolution rule changes against the fixed gap used for WHEN.

    `table` is the output of `build_what_table`. The folds come from
    `make_what_folds`; the fixed-gap alternative is derived from the same
    development rows: a train row is kept by the fixed gap if `year` <
    `start` - `gap_years`. Only `artist_id`, `year`, `history_count` and
    `next_active_year` are passed on, so no label column is read and no label
    statistic can leak; development years only.

    Returns one row per fold (index `fold`) with `test_years`,
    `train_rows_exact`, `train_rows_fixed_gap`, `dropped_by_exact_rule`,
    `dropped_by_fixed_gap`, `fixed_gap_rows_resolving_at_or_after_start` (fixed-gap
    train rows whose label resolves at or after the test start, with their share
    of the fixed-gap train rows in `fixed_gap_resolving_share`),
    `rows_kept_by_exact_rule_but_dropped_by_gap` (recent rows whose label
    resolved in time), and `net_train_difference` (exact minus fixed gap, with
    its share of the fixed-gap train rows in `net_train_difference_share`).
    """
    # Label-free view of the table, so nothing below can read a label.
    rows = table[["artist_id", "year", "history_count", "next_active_year"]]
    # Development years only, and the exact folds from the shared function.
    development = rows[rows["year"] < holdout_start]
    folds = make_what_folds(rows, n_folds, test_years, holdout_start)
    records = []
    for fold in folds:
        # First test year of this fold, from the fold's own label.
        start = int(fold["test_years"].split("-")[0])
        # Rows before the test tranche, and the ones the fixed gap keeps (a whole gap before the test).
        before = development[development["year"] < start]
        fixed_train = before[before["year"] < start - gap_years]
        # Fixed-gap train rows whose next active year is at or after the test start: their label is not resolved yet.
        resolving = int((fixed_train["next_active_year"] >= start).sum())
        # Exact-rule train rows that sit inside the gap: recent rows the fixed gap would throw away.
        kept_by_exact = int((fold["train"]["year"] >= start - gap_years).sum())
        # Bookkeeping: exact train = fixed train - unresolved rows + recent resolved rows.
        assert fold["train_rows"] == len(fixed_train) - resolving + kept_by_exact, "exact and fixed-gap counts do not reconcile"
        records.append({
            "fold": fold["fold"],
            "test_years": fold["test_years"],
            "train_rows_exact": fold["train_rows"],
            "train_rows_fixed_gap": len(fixed_train),
            "dropped_by_exact_rule": fold["dropped_by_resolution_rule"],
            "dropped_by_fixed_gap": len(before) - len(fixed_train),
            "fixed_gap_rows_resolving_at_or_after_start": resolving,
            "fixed_gap_resolving_share": resolving / len(fixed_train),
            "rows_kept_by_exact_rule_but_dropped_by_gap": kept_by_exact,
            "net_train_difference": fold["train_rows"] - len(fixed_train),
            "net_train_difference_share": (fold["train_rows"] - len(fixed_train)) / len(fixed_train),
        })
    return pd.DataFrame(records).set_index("fold")


def run_what_baselines(table: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP, confidence: float = KILL_GATE_CONFIDENCE) -> dict:
    """Run the three WHAT baselines on the five folds and measure the lifts between them.

    `table` is the output of `build_what_table`. The folds come from
    `make_what_folds` (development years only), the bucket columns are added to
    each fold's train and test rows with `add_what_buckets`, and every baseline is
    scored with `evaluate_folds` on the target `new_style_year`: (a) the constant
    train rate, (b) the lookup on `history_bucket`, (c) the lookup on
    `history_bucket` and `volume_bucket`, the reference of the kill gate.

    Returns a dict of two dataframes:
    - `ladder`: one row per (`baseline`, `fold`) with `test_years`, `pr_auc`,
      `pr_auc_lo`, `pr_auc_hi`, `ci_half_width_points` (PR-AUC points),
      `positive_rate`, `roc_auc`, `train_rows` and `test_rows`.
    - `lifts`: one row per fold with `lift_c_over_a`, `lift_c_over_b` and
      `lift_b_over_a` (PR-AUC points), `flag_ci_overlap_c_vs_b` (True when the
      CIs of (c) and (b) overlap), `reference_ci_half_width_points` (half-width
      of the CI of (c)) and `two_half_widths_points` (twice that: how large a
      difference must be for two CIs of that width to separate).
    """
    target = "new_style_year"
    # Folds, with the two bucket columns added once to each fold's train and test rows (the helper asserts nothing is dropped).
    folds = make_what_folds(table)
    for fold in folds:
        fold["train"] = add_what_buckets(fold["train"])
        fold["test"] = add_what_buckets(fold["test"])
    # (fit, score) pair of each baseline; the constant one reads the Work-level column name, so the label is renamed for it.
    baselines = {
        "a_constant": (lambda train: fit_what_baseline(train.rename(columns={target: "new_style"})), score_what_baseline),
        "b_history": (lambda train: fit_what_lookup_baseline(train, ["history_bucket"]), score_what_lookup_baseline),
        "c_history_volume": (lambda train: fit_what_lookup_baseline(train, ["history_bucket", "volume_bucket"]), score_what_lookup_baseline),
    }
    # One evaluation per baseline, stacked into one table.
    ladder = pd.concat(
        {name: evaluate_folds(folds, target, fit_fn, score_fn, n_bootstrap, confidence) for name, (fit_fn, score_fn) in baselines.items()},
        names=["baseline"],
    )
    # Half of the CI width, in PR-AUC points.
    ladder["ci_half_width_points"] = (ladder["pr_auc_hi"] - ladder["pr_auc_lo"]) / 2 * 100
    ladder = ladder[["test_years", "pr_auc", "pr_auc_lo", "pr_auc_hi", "ci_half_width_points", "positive_rate", "roc_auc", "train_rows", "test_rows"]]

    # Lifts between baselines, fold by fold, in PR-AUC points.
    a, b, c = (ladder.loc[name] for name in baselines)
    lifts = pd.DataFrame({
        "test_years": c["test_years"],
        "lift_c_over_a": (c["pr_auc"] - a["pr_auc"]) * 100,
        "lift_c_over_b": (c["pr_auc"] - b["pr_auc"]) * 100,
        "lift_b_over_a": (b["pr_auc"] - a["pr_auc"]) * 100,
        # The CIs overlap when neither interval lies entirely above the other.
        "flag_ci_overlap_c_vs_b": (c["pr_auc_lo"] <= b["pr_auc_hi"]) & (b["pr_auc_lo"] <= c["pr_auc_hi"]),
        "reference_ci_half_width_points": c["ci_half_width_points"],
        "two_half_widths_points": 2 * c["ci_half_width_points"],
    })
    return {"ladder": ladder, "lifts": lifts}


def build_artist_year_genres(works: pd.DataFrame) -> pd.DataFrame:
    """Genre counterpart of `build_artist_years`: does the next active year bring a new genre.

    Used only for a descriptive measurement of genre change, not as a target
    for the kill gate. The row set, the sort, `history_count` and
    `next_active_year` are the ones of `build_artist_years`; the tags are taken
    from the `genres` column instead of `styles`, and `FAKE_GENRE_TAGS` are
    removed before anything else. A Work whose only tag is a fake one keeps its
    place in `n_works_year` and `history_count` but adds nothing to the
    vocabulary.

    Returns one row per (artist, active year), sorted by artist and year, with:

    - `artist_id`, `year`, `history_count` (dated Works strictly before `year`),
      `n_works_year` and `next_active_year`, as in `build_artist_years`;
    - `genres_seen`: set of every genre used by the artist up to the end of
      `year`, plus the genres of their undated Works (same convention as styles);
    - `next_year_genres`: set of the genres of the Works of `next_active_year`,
      empty when there is no next active year;
    - `new_genre_year`: 1 if `next_year_genres` has a genre not in `genres_seen`,
      0 if not, missing when `next_year_genres` is empty (no next active year, or
      the next year's Works carry no valid genre).
    """
    # A Work with no artist_id cannot belong to any artist's timeline.
    works = works[works["artist_id"].notna()]
    # A Work with no year cannot be placed on a timeline.
    undated = works["year"].isna()

    # Every valid genre used on any undated Work, one set per artist (the baseline).
    undated_genres_by_artist = works[undated].groupby("artist_id")["genres"].apply(
        lambda genres_column: set().union(*(_split_styles(g) - set(FAKE_GENRE_TAGS) for g in genres_column))
    )

    # From here on only dated Works, with year as a plain integer.
    dated = works[~undated][["artist_id", "year", "genres"]].copy()
    dated["year"] = dated["year"].astype(int)
    # Turn each Work's tag list into a set once, without the fake tags.
    dated["genres"] = dated["genres"].map(lambda g: _split_styles(g) - set(FAKE_GENRE_TAGS))

    # One row per artist-year: its Work count and the union of its Works' genres.
    artist_years = dated.groupby(["artist_id", "year"]).agg(
        n_works_year=("genres", "size"),
        year_genres=("genres", lambda column: set().union(*column)),
    ).reset_index()  # already sorted by artist_id then year, which the loop below relies on

    # Dated Works strictly before each year: running total minus this year's own count.
    artist_years["history_count"] = (
        artist_years.groupby("artist_id")["n_works_year"].cumsum() - artist_years["n_works_year"]
    )
    # The artist's next active year (missing for the last one), and the next row's genres.
    by_artist = artist_years.groupby("artist_id", sort=False)
    artist_years["next_active_year"] = by_artist["year"].shift(-1)
    next_year_genres = by_artist["year_genres"].shift(-1)

    # The running vocabulary needs one pass in Python, resetting at each new artist.
    # This loop mirrors build_artist_years line for line on the genres column; the
    # alignment assertion run against build_artist_years guards against divergence.
    artist_ids = artist_years["artist_id"].to_numpy()
    year_genres = artist_years["year_genres"].tolist()
    next_genres_list = next_year_genres.tolist()
    genres_seen = []  # vocabulary through the end of each artist-year
    new_genre_year = np.full(len(artist_years), np.nan)
    current_artist = None
    seen: set = set()
    for i in range(len(artist_years)):
        # A new artist starts from their undated-Works genres, not from empty.
        if artist_ids[i] != current_artist:
            current_artist = artist_ids[i]
            seen = set(undated_genres_by_artist.get(current_artist, set()))
        # Every genre of this year is known at the end of it, so it counts as seen.
        seen |= year_genres[i]
        # A copy, because `seen` keeps growing on the next iteration.
        genres_seen.append(set(seen))
        # Missing next row (last year of the artist) is NaN, a real set otherwise.
        next_set = next_genres_list[i] if isinstance(next_genres_list[i], set) else set()
        # Label only when the next year has a valid genre, otherwise "new" is unknowable.
        if next_set:
            new_genre_year[i] = float(len(next_set - seen) > 0)
    artist_years["genres_seen"] = genres_seen
    artist_years["next_year_genres"] = [s if isinstance(s, set) else set() for s in next_genres_list]
    artist_years["new_genre_year"] = new_genre_year

    # Column order: identity, features, label.
    return artist_years[[
        "artist_id", "year", "history_count", "n_works_year", "next_active_year",
        "genres_seen", "next_year_genres", "new_genre_year",
    ]]


def _two_label_prevalence(frame: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    """Rows and prevalence of both labels for each group of `key`.

    `frame` has one row per eligible artist-year with `new_style_year` and
    `new_genre_year`; `key` assigns each row to a group. A row without a label
    counts in `eligible_rows` but not in that label's `labelled_rows`.
    """
    # One group per key value; observed=False keeps empty buckets as rows.
    grouped = frame.groupby(key, observed=False)
    return pd.DataFrame({
        "eligible_rows": grouped.size(),
        "style_labelled_rows": grouped["new_style_year"].count(),
        "style_prevalence": grouped["new_style_year"].mean(),
        "genre_labelled_rows": grouped["new_genre_year"].count(),
        "genre_prevalence": grouped["new_genre_year"].mean(),
    })


def genre_change_summary(works: pd.DataFrame, artist_years: pd.DataFrame, artist_year_genres: pd.DataFrame, min_previous_works: int = 3, holdout_start: int = HOLDOUT_START_YEAR) -> dict:
    """Describe how often the next active year brings a new genre, next to the style label.

    `artist_years` is the output of `build_artist_years` and
    `artist_year_genres` the output of `build_artist_year_genres`; they must
    have the same rows (asserted). Descriptive only, not a target for the kill
    gate. Population: artist-years with `year` < `holdout_start` and
    `history_count` >= `min_previous_works` (the eligible set of
    `label_loss_breakdown`). The tables are cut to it before any label column is
    read, so no label statistic uses a holdout year.

    Returns a dict of dataframes:
    - `overall`: per label, eligible rows, labelled rows, unlabelled rows and prevalence.
    - `by_window`: both prevalences for the five WHEN test tranches.
    - `by_genre`: both prevalences by primary genre (`_primary_genre`), for genres
      with at least `MIN_GENRE_ROWS` eligible rows, after the row `all genres`.
    - `by_history_bucket`: both prevalences by prior-work bucket of src/eda.py.
    - `crosstab`: on rows where both labels exist, `rows`, `base` and `share` for
      the four cells of style against genre, the two conditional probabilities
      (genre = 1 given style = 1, style = 1 given genre = 1) and the three-level
      split (no new style and no new genre, new style without new genre, new genre
      with or without new style).
    - `style_missing_rows`: prevalence of `new_genre_year` on the rows where the
      genre label exists and the style label is missing (the next active year has
      no style tags), next to the rows where both exist.
    - `by_next_year_volume`: both prevalences by number of Works in the next
      active year (buckets of src/eda.py), over eligible rows with a next active
      year; a diagnostic, unknown at the cutoff.
    """
    # The two tables must describe exactly the same artist-years.
    keys = ["artist_id", "year", "history_count", "n_works_year"]
    assert artist_years[keys].reset_index(drop=True).equals(artist_year_genres[keys].reset_index(drop=True)), "style and genre tables are not row-aligned"
    # Eligible development rows first, before any label column is read.
    eligible = ((artist_years["year"] < holdout_start) & (artist_years["history_count"] >= min_previous_works)).to_numpy()
    frame = artist_years.loc[eligible, ["artist_id", "year", "history_count", "n_works_next_year", "new_style_year"]].reset_index(drop=True)
    frame["new_genre_year"] = artist_year_genres.loc[eligible, "new_genre_year"].to_numpy()
    # Primary genre of the artist; an artist with no valid genre gets its own group.
    frame["primary_genre"] = frame["artist_id"].map(_primary_genre(works)).fillna("(no valid genre)")
    style, genre = frame["new_style_year"], frame["new_genre_year"]

    # 1. Overall: eligible rows, labelled rows and prevalence of each label.
    overall = pd.DataFrame({
        "eligible_rows": len(frame),
        "labelled_rows": [style.count(), genre.count()],
        "unlabelled_rows": [style.isna().sum(), genre.isna().sum()],
        "prevalence": [style.mean(), genre.mean()],
    }, index=["new_style_year", "new_genre_year"])
    assert (overall["labelled_rows"] + overall["unlabelled_rows"] == overall["eligible_rows"]).all(), "overall rows do not add up"

    # 2. By window: the five WHEN test tranches (rows before the first tranche are not shown).
    first_tranche = holdout_start - N_FOLD_TRANCHES * TRANCHE_YEARS
    edges = [first_tranche - 1 + k * TRANCHE_YEARS for k in range(N_FOLD_TRANCHES + 1)]
    labels = [f"{edges[k] + 1}-{edges[k + 1]}" for k in range(N_FOLD_TRANCHES)]
    by_window = _two_label_prevalence(frame, pd.cut(frame["year"], bins=edges, labels=labels))

    # 3. By primary genre: genres with enough rows, largest first, after the overall row.
    by_genre = _two_label_prevalence(frame, frame["primary_genre"])
    by_genre = by_genre[by_genre["eligible_rows"] >= MIN_GENRE_ROWS].sort_values("eligible_rows", ascending=False)
    all_genres = _two_label_prevalence(frame, pd.Series("all genres", index=frame.index))
    by_genre = pd.concat([all_genres, by_genre])

    # 4. By prior-work bucket, with the eda helper (it also applies the history rule, a no-op here).
    bucketed = _bucket_by_prior_works(frame.rename(columns={"history_count": "prior_work_count"}), min_previous_works)
    assert bucketed.index.equals(frame.index), "bucketing changed the rows"
    by_history_bucket = _two_label_prevalence(frame, bucketed["prior_work_bucket"])

    # 5. Cross-tab on the rows where both labels exist.
    both = frame[style.notna() & genre.notna()]
    cell = lambda s, g: int(((both["new_style_year"] == s) & (both["new_genre_year"] == g)).sum())
    n_both = len(both)
    style_one = int((both["new_style_year"] == 1).sum())
    genre_one = int((both["new_genre_year"] == 1).sum())
    crosstab = pd.DataFrame([
        ("style 0, genre 0", cell(0, 0), n_both),
        ("style 0, genre 1", cell(0, 1), n_both),
        ("style 1, genre 0", cell(1, 0), n_both),
        ("style 1, genre 1", cell(1, 1), n_both),
        ("P(genre = 1 | style = 1)", cell(1, 1), style_one),
        ("P(style = 1 | genre = 1)", cell(1, 1), genre_one),
        ("no new style and no new genre", cell(0, 0), n_both),
        ("new style without new genre", cell(1, 0), n_both),
        ("new genre (with or without new style)", genre_one, n_both),
    ], columns=["measure", "rows", "base"]).set_index("measure")
    crosstab["share"] = crosstab["rows"] / crosstab["base"]
    # The three levels cover every compared row exactly once.
    assert crosstab.loc[["no new style and no new genre", "new style without new genre", "new genre (with or without new style)"], "rows"].sum() == n_both, "three-level split does not add up"

    # 6. Rows with a genre label but no style label, next to the rows where both exist.
    missing = frame[genre.notna() & style.isna()]
    style_missing_rows = pd.DataFrame({
        "rows": [len(missing), len(both)],
        "genre_prevalence": [missing["new_genre_year"].mean(), both["new_genre_year"].mean()],
    }, index=["genre label only (next year without style tags)", "both labels exist"])

    # 7. By next-year volume, over rows with a next active year; a diagnostic, never a feature.
    volume = pd.cut(frame["n_works_next_year"].astype(float), bins=VOLUME_BUCKET_EDGES, labels=VOLUME_BUCKET_LABELS)
    by_next_year_volume = _two_label_prevalence(frame[volume.notna()], volume[volume.notna()])

    return {
        "overall": overall, "by_window": by_window, "by_genre": by_genre, "by_history_bucket": by_history_bucket,
        "crosstab": crosstab, "style_missing_rows": style_missing_rows, "by_next_year_volume": by_next_year_volume,
    }


def build_what_features(works: pd.DataFrame) -> pd.DataFrame:
    """Model inputs for WHAT, one row per artist-year, computed at the end of each year.

    `works` is the Works table (one row per Work, shaped like
    `data/processed/works_checkpoint.parquet`). The rows and the keys are those of
    `build_artist_years`: only dated Works with an artist define them. Every value
    of a row at year t uses only Works dated t or earlier. Undated Works enter no
    feature, which is a deliberate difference from the target label, where the
    styles of undated Works count as already seen. Missing values stay NaN, nothing
    is imputed. `history_count`, `n_works_year`, the label and `n_works_next_year`
    are not included (the first two come from `build_what_table`).

    Returns one row per (artist, active year), sorted by artist and year, with:

    - `artist_id`, `year`;
    - `n_styles_seen`: distinct styles used through the end of `year` (dated Works);
    - `n_styles_year`: distinct styles in the Works of `year`;
    - `n_new_styles_year`: styles of `year` absent from every earlier dated year; a
      year is novelty-eligible when it has style tags and the dated vocabulary of
      earlier years is not empty, so the value is missing in the first active year
      and in the first year with style tags (every style would be new by
      definition), 0 for a later year without style tags;
    - `styles_per_work`: `n_styles_seen` over the dated Works through `year`;
    - `new_style_rate_so_far`: mean of an indicator over the novelty-eligible years up
      to and including `year`; the indicator is 1 if the year has a style absent from
      the dated vocabulary of earlier years, else 0; years that are not
      novelty-eligible are ignored, not counted as 0; missing when there is no such
      year yet;
    - `years_since_new_style`: `year` minus the last novelty-eligible year with
      `n_new_styles_year` > 0 (missing if none yet);
    - `years_since_prev_active_year`: `year` minus the previous active year (missing
      in the first active year);
    - `years_active_so_far`: `year` minus the first active year;
    - `top_style_share`: share of style tag occurrences through `year` (one count per
      Work and style) that belong to the most frequent style (missing without tags);
    - `n_genres_seen`: distinct genres through `year`, without `FAKE_GENRE_TAGS`;
    - `genre_<name>` for each genre of `models.GENRES`: True if a Work of `year`
      carries the genre;
    - `albums_share_year`, `singles_share_year`: share of the Works of `year` that
      are album-like or single-like, with the classification of the snapshots.
    """
    # A Work with no artist_id or no year cannot be placed on a timeline, and undated Works enter no feature.
    works = works[works["artist_id"].notna() & works["year"].notna()]
    # Only the columns the features need, with year as a plain integer.
    dated = works[["artist_id", "year", "styles", "genres", "format_descriptions"]].copy()
    dated["year"] = dated["year"].astype(int)
    # Tag lists as sets once, so the unions below are cheap; the fake genre tags are removed first.
    dated["styles"] = dated["styles"].map(_split_styles)
    dated["genres"] = dated["genres"].map(lambda g: _split_styles(g) - set(FAKE_GENRE_TAGS))
    # Work type with the classification of the snapshots, as two booleans so the sums below are fast.
    work_type = dated["format_descriptions"].map(classify_work_type)
    dated["is_album"] = work_type == "album"
    dated["is_single"] = work_type == "single"

    # One row per artist-year: Work count, each Work's style set, union of genres, album and single counts.
    rows = dated.groupby(["artist_id", "year"]).agg(
        n_works=("styles", "size"),
        work_styles=("styles", list),
        year_genres=("genres", lambda column: set().union(*column)),
        n_albums=("is_album", "sum"),
        n_singles=("is_single", "sum"),
    ).reset_index()  # already sorted by artist_id then year, which the loop below relies on

    # Arrays for the pass over the rows, one entry per artist-year.
    n = len(rows)
    artist_ids = rows["artist_id"].to_numpy()
    years = rows["year"].to_numpy()
    n_works = rows["n_works"].to_numpy()
    work_styles = rows["work_styles"].tolist()
    year_genres = rows["year_genres"].tolist()
    # Feature arrays start as NaN, so a value that is never set stays missing.
    n_styles_seen = np.full(n, np.nan)
    n_styles_year = np.full(n, np.nan)
    n_new_styles_year = np.full(n, np.nan)
    styles_per_work = np.full(n, np.nan)
    new_style_rate = np.full(n, np.nan)
    years_since_new = np.full(n, np.nan)
    years_since_prev = np.full(n, np.nan)
    years_active = np.full(n, np.nan)
    top_share = np.full(n, np.nan)
    n_genres_seen = np.full(n, np.nan)

    # One pass in Python, resetting the running state at each new artist; each row is written before the state absorbs its year.
    current_artist = None
    for i in range(n):
        # A new artist starts from empty state: only dated Works count.
        if artist_ids[i] != current_artist:
            current_artist = artist_ids[i]
            vocabulary: set = set()  # distinct styles of earlier and current dated years
            tag_counts: dict = {}  # occurrences of each style, one per Work
            tag_total = 0  # all style tag occurrences so far
            tag_top = 0  # count of the most frequent style so far
            works_so_far = 0  # dated Works through the current year
            genres_seen: set = set()  # distinct genres through the current year
            tagged_years = 0  # novelty-eligible years: style tags and a non-empty vocabulary of earlier years
            new_years = 0  # of those, years with a style absent from earlier years
            last_new_year = None  # last novelty-eligible year with a new style
            first_year = years[i]
            previous_year = None
        # Styles of this year as one set, and those not used in any earlier dated year.
        year_styles = set().union(*work_styles[i])
        new_styles = year_styles - vocabulary
        # Features that compare this year with earlier years exist only after the first active year.
        if previous_year is not None:
            years_since_prev[i] = years[i] - previous_year
            if not year_styles:
                # A later year without style tags has no new style and adds no indicator.
                n_new_styles_year[i] = 0
            elif vocabulary:
                # Novelty-eligible year: tags, and earlier years gave a non-empty vocabulary to compare with.
                n_new_styles_year[i] = len(new_styles)
                tagged_years += 1
                new_years += len(new_styles) > 0
                if new_styles:
                    last_new_year = years[i]
            # Otherwise this is the first year with style tags: every style is new by definition, so it stays missing.
        # The state absorbs this year: vocabulary, tag counts, Works and genres.
        vocabulary |= year_styles
        for work in work_styles[i]:
            for style in work:
                tag_counts[style] = tag_counts.get(style, 0) + 1
                tag_total += 1
                tag_top = max(tag_top, tag_counts[style])
        works_so_far += n_works[i]
        genres_seen |= year_genres[i]
        # Features that describe the state through the end of this year.
        n_styles_seen[i] = len(vocabulary)
        n_styles_year[i] = len(year_styles)
        styles_per_work[i] = len(vocabulary) / works_so_far
        if tagged_years:
            new_style_rate[i] = new_years / tagged_years
        if last_new_year is not None:
            years_since_new[i] = years[i] - last_new_year
        years_active[i] = years[i] - first_year
        if tag_total:
            top_share[i] = tag_top / tag_total
        n_genres_seen[i] = len(genres_seen)
        previous_year = years[i]

    # Output in the column order of the docstring: keys, style features, genre features, Work type shares.
    features = pd.DataFrame({
        "artist_id": rows["artist_id"], "year": rows["year"],
        "n_styles_seen": n_styles_seen, "n_styles_year": n_styles_year, "n_new_styles_year": n_new_styles_year,
        "styles_per_work": styles_per_work, "new_style_rate_so_far": new_style_rate,
        "years_since_new_style": years_since_new, "years_since_prev_active_year": years_since_prev,
        "years_active_so_far": years_active, "top_style_share": top_share, "n_genres_seen": n_genres_seen,
    })
    # One True / False column per genre of the WHEN vocabulary, named like the WHEN models name theirs.
    for genre in GENRES:
        features[f"genre_{genre}"] = [genre in genres for genres in year_genres]
    # Share of this year's Works that are album-like or single-like.
    features["albums_share_year"] = rows["n_albums"] / rows["n_works"]
    features["singles_share_year"] = rows["n_singles"] / rows["n_works"]
    return features


def _artist_sample(works: pd.DataFrame, n_artists: int, seed: int) -> pd.DataFrame:
    """All Works of a seeded random sample of `n_artists` artists, as in `backtesting.point_in_time_check`."""
    # Random artist sample, the same draw as the WHEN check for the same seed.
    artists = works["artist_id"].dropna().drop_duplicates().sample(n_artists, random_state=seed)
    # Every Work of the sampled artists.
    return works[works["artist_id"].isin(artists)]


def _compare_builds(full: pd.DataFrame, cut: pd.DataFrame, columns: list, kind: str) -> pd.DataFrame:
    """Count, per column, the rows that differ between two builds with the same (artist_id, year) rows.

    NaN equal to NaN counts as the same. A differing row where exactly one side is
    NaN is also counted in `differ_nan_vs_value`: a value that appeared or vanished.
    """
    # Same rows in the same order, so the columns line up row by row.
    full = full.set_index(["artist_id", "year"])
    cut = cut.set_index(["artist_id", "year"])
    assert full.index.equals(cut.index), "the two builds do not have the same rows"
    records = []
    for column in columns:
        # Floats make booleans, integers and NaN comparable in the same way.
        a = full[column].astype(float)
        b = cut[column].astype(float)
        # Equal values, or both missing, are the same; everything else differs.
        differs = ~((a == b) | (a.isna() & b.isna()))
        # Of the differing rows, those where one side is missing and the other is not.
        one_missing = differs & (a.isna() | b.isna())
        records.append((column, kind, len(a), int(differs.sum()), int(one_missing.sum())))
    return pd.DataFrame(records, columns=["column", "kind", "rows_compared", "rows_differ", "differ_nan_vs_value"]).set_index("column")


def what_point_in_time_check(works: pd.DataFrame, cutoff_year: int, n_artists: int = 30000, seed: int = 1, features_fn=build_what_features) -> pd.DataFrame:
    """Count, per WHAT feature, the rows whose value changes when the future is removed.

    Mirrors `backtesting.point_in_time_check`. Builds the features and the
    artist-year table twice for a seeded random sample of artists: once from all
    their Works and once from the Works dated up to `cutoff_year` (undated Works
    are kept, they have no year to cut on). Then compares both builds on the rows
    with `year` <= `cutoff_year`. The set of (artist_id, year) rows with
    `year` <= `cutoff_year` must be identical in both builds, asserted before any
    column is compared. A feature that only uses Works up to its own year must be
    identical in both (0 rows differ).

    Returns one row per column with `kind` (`feature` or `positive control`),
    `rows_compared`, `rows_differ` and `differ_nan_vs_value` (differing rows where
    exactly one side is missing). The positive controls are `new_style_year` and
    `n_works_next_year`, which look forward by design, so they must differ; this is
    asserted, so the check fails loudly if it ever stops detecting a leak.
    `features_fn` defaults to `build_what_features`; it only exists so the check can
    be tested on a feature built on purpose with a forward look.
    """
    # All Works of the sampled artists, and the same Works without the future.
    sample = _artist_sample(works, n_artists, seed)
    truncated = sample[sample["year"].isna() | (sample["year"] <= cutoff_year)]
    builds = {}
    for name, source in [("full", sample), ("cut", truncated)]:
        # Features and artist-year table, restricted to the rows inside the truncated window.
        features = features_fn(source)
        artist_years = build_artist_years(source)
        builds[name] = (features[features["year"] <= cutoff_year].reset_index(drop=True), artist_years[artist_years["year"] <= cutoff_year].reset_index(drop=True))
    (full_features, full_years), (cut_features, cut_years) = builds["full"], builds["cut"]
    # Truncation only removes later rows: the rows up to the cutoff must be the same in both builds.
    keys = ["artist_id", "year"]
    assert full_features[keys].equals(cut_features[keys]), "feature rows differ between the full and the truncated build"
    assert full_years[keys].equals(cut_years[keys]), "artist-year rows differ between the full and the truncated build"
    # Features, then the two positive controls from the artist-year table.
    feature_columns = [c for c in full_features.columns if c not in keys]
    table = pd.concat([
        _compare_builds(full_features, cut_features, feature_columns, "feature"),
        _compare_builds(full_years, cut_years, ["new_style_year", "n_works_next_year"], "positive control"),
    ])
    # The controls look forward by design: if they do not differ, the check cannot see leaks.
    assert (table.loc[["new_style_year", "n_works_next_year"], "rows_differ"] > 0).all(), "a positive control did not differ"
    return table


def what_undated_check(works: pd.DataFrame, n_artists: int = 30000, seed: int = 1, features_fn=build_what_features) -> pd.DataFrame:
    """Check that undated Works enter no WHAT feature.

    Builds the features for the same seeded sample of artists as
    `what_point_in_time_check`, once with all their Works and once without their
    undated Works, and compares every feature column on every row. The rows must
    be the same in both builds (asserted). Returns one row per feature with
    `rows_compared`, `rows_differ` and `differ_nan_vs_value`; all must be 0.
    """
    # All Works of the sampled artists, and the same Works without the undated ones.
    sample = _artist_sample(works, n_artists, seed)
    dated_only = sample[sample["year"].notna()]
    full = features_fn(sample)
    cut = features_fn(dated_only)
    # Undated Works define no row, so the rows must be the same.
    keys = ["artist_id", "year"]
    assert full[keys].equals(cut[keys]), "feature rows differ with and without undated Works"
    return _compare_builds(full, cut, [c for c in full.columns if c not in keys], "feature")


# Gate pass rule (docs/decisions.md, "Kill gate pass rule for WHAT"), fixed before any WHAT model result existed.
GATE_MIN_FOLDS = 4  # folds that must meet rule (a), out of the 5 development folds
GATE_MARGIN_POINTS = 1.0  # smallest PR-AUC improvement over baseline (c), in absolute points, that counts as practically relevant
GATE_TOLERANCE_POINTS = 1e-9  # only absorbs floating point error in (model - baseline) * 100, far below any real difference
# PR-AUC of baseline (c) per fold, logged in the baseline ladder entry of docs/decisions.md; the gate must reproduce it.
WHAT_BASELINE_C_PR_AUC = [0.3474, 0.3479, 0.3625, 0.3698, 0.3597]


def _apply_gate_rule(folds: pd.DataFrame, min_folds: int = GATE_MIN_FOLDS, margin_points: float = GATE_MARGIN_POINTS) -> tuple:
    """Apply the logged pass rule to a per-fold table, on unrounded values.

    `folds` has one row per fold with `model_pr_auc`, `model_pr_auc_lo`,
    `model_pr_auc_hi` and the same three columns with prefix `baseline_c_`.
    Rule (a): in at least `min_folds` folds the lower bound of the model interval is
    above the upper bound of the baseline interval AND the improvement is at least
    `margin_points` (absolute PR-AUC points). Rule (b): in no fold the model interval
    lies entirely below the baseline interval. Nothing is rounded here.

    Returns `(folds_with_flags, verdict)`: the table with `improvement_points`,
    `ci_separated`, `rule_a_fold` and `rule_b_fold_fails`, and a one-row dataframe with
    the verdict, the folds meeting (a), the folds failing (b) and the improvement per fold.
    """
    table = folds.copy()
    # Improvement in absolute PR-AUC points, unrounded.
    table["improvement_points"] = (table["model_pr_auc"] - table["baseline_c_pr_auc"]) * 100
    # Strictly separated intervals: the model's lower bound is above the baseline's upper bound.
    table["ci_separated"] = table["model_pr_auc_lo"] > table["baseline_c_pr_auc_hi"]
    # Rule (a) in this fold: separated intervals and an improvement of at least the margin.
    table["rule_a_fold"] = table["ci_separated"] & (table["improvement_points"] >= margin_points - GATE_TOLERANCE_POINTS)
    # Rule (b) is broken in this fold when the model's interval lies entirely below the baseline's.
    table["rule_b_fold_fails"] = table["model_pr_auc_hi"] < table["baseline_c_pr_auc_lo"]
    folds_a = int(table["rule_a_fold"].sum())
    folds_b = int(table["rule_b_fold_fails"].sum())
    rule_a_met = folds_a >= min_folds
    rule_b_met = folds_b == 0
    verdict = {
        "result": "PASS" if rule_a_met and rule_b_met else "FAIL",
        "folds_passing_a": folds_a, "min_folds_required": min_folds, "rule_a_met": rule_a_met,
        "folds_failing_b": folds_b, "rule_b_met": rule_b_met, "margin_points": margin_points,
    }
    # The improvement of each fold, so the verdict can be audited without the fold table.
    for fold, value in table["improvement_points"].items():
        verdict[f"improvement_points_fold_{fold}"] = value
    return table, pd.DataFrame([verdict], index=["verdict"])


def _record_scores(score_fn, store: list):
    """Wrap a score function so that the scores it returns are also appended to `store`."""
    def recorded(test, fitted):
        """Score the test rows with `score_fn` and keep a copy of the scores in `store`. Inputs: the test dataframe and the fitted model. Output: the array of scores, one per test row."""
        scores = np.asarray(score_fn(test, fitted))  # the scores evaluate_folds will use
        assert len(scores) == len(test), "a score function must return one score per test row"
        store.append(scores)
        return scores
    return recorded


def run_what_gate(works: pd.DataFrame, table: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP, confidence: float = KILL_GATE_CONFIDENCE, make_model=make_what_xgboost, features: pd.DataFrame = None, reference_pr_auc: list = WHAT_BASELINE_C_PR_AUC) -> dict:
    """Run the WHAT kill gate: the reference model against baseline (c) on the five development folds.

    `table` is the output of `build_what_table`, `works` the Works table. The folds come
    from `make_what_folds` (development years only, the holdout is never touched). The
    model comes from `make_model(features)`, with `features` built once by
    `build_what_features(works)` unless given (the audit shows they are point-in-time
    safe). Model and baseline (c) go through `evaluate_folds` with the same target,
    bootstrap size, confidence and default seed as `run_what_baselines`, so the intervals
    are comparable. One model, one configuration: nothing is tuned here.

    Two sanity checks stop the run with an assertion: baseline (c) must reproduce
    `reference_pr_auc` per fold to 4 decimals (skipped when it is None), and the model
    must not give constant scores in any fold. `evaluate_folds` calls the score
    function once per fold, on the full test set; that is asserted, so the recorded
    scores are the ones behind the intervals.

    Returns a dict of dataframes:
    - `folds`: per fold, train rows, test rows, test artists, distinct model scores,
      test positive rate, PR-AUC with interval for the model and for baseline (c),
      `improvement_points`, `ci_separated`, `rule_a_fold` and `rule_b_fold_fails`.
    - `verdict`: one row with the result, the folds meeting rule (a), the folds failing
      rule (b) and the improvement per fold (see `_apply_gate_rule`).
    - `by_genre`: DESCRIPTIVE ONLY, no gate decision. PR-AUC of the model and of baseline
      (c) by primary genre, pooled over the five disjoint test tranches, with rows and
      positive rate, after the row `all genres`. The primary genre is taken from the
      artist's full Works (`_primary_genre`) for stratification only and is never a
      model input.
    - `scores`: the row-level scores behind the intervals, one row per test row of the
      five folds, with `artist_id`, `year`, `fold`, `label`, `model_score` and `c_score`
      (baseline (c)). The diagnostics read this table, so they use the same scores.
    """
    target = "new_style_year"
    # Features once, on the full Works table.
    if features is None:
        features = build_what_features(works)
    # Folds with the bucket columns baseline (c) needs, added once to each train and test table.
    folds = make_what_folds(table)
    for fold in folds:
        fold["train"] = add_what_buckets(fold["train"])
        fold["test"] = add_what_buckets(fold["test"])
    model_fit, model_score = make_model(features)
    baseline_fit = lambda train: fit_what_lookup_baseline(train, ["history_bucket", "volume_bucket"])
    # Each run records the scores of its test sets, in fold order.
    recorded = {"model": [], "baseline_c": []}
    results = {}
    for name, fit_fn, score_fn in [("model", model_fit, model_score), ("baseline_c", baseline_fit, score_what_lookup_baseline)]:
        results[name] = evaluate_folds(folds, target, fit_fn, _record_scores(score_fn, recorded[name]), n_bootstrap, confidence)
        assert len(recorded[name]) == len(folds), "the score function was not called exactly once per fold"
    model, baseline = results["model"], results["baseline_c"]

    # Sanity check 1: baseline (c) is the same as in the baseline ladder.
    if reference_pr_auc is not None:
        assert baseline["pr_auc"].round(4).tolist() == list(reference_pr_auc), f"baseline (c) does not reproduce the ladder: {baseline['pr_auc'].round(4).tolist()}"
    # Sanity check 2: the model ranks rows, it does not output one constant.
    distinct = [len(np.unique(scores)) for scores in recorded["model"]]
    assert min(distinct) > 1, f"the model gave constant scores in a fold: {distinct}"

    # Per-fold table, then the pass rule on the unrounded values.
    per_fold = pd.DataFrame({
        "test_years": model["test_years"],
        "train_rows": model["train_rows"], "test_rows": model["test_rows"],
        "test_artists": [fold["test_artists"] for fold in folds],
        "distinct_model_scores": distinct,
        "positive_rate": model["positive_rate"],
        "model_pr_auc": model["pr_auc"], "model_pr_auc_lo": model["pr_auc_lo"], "model_pr_auc_hi": model["pr_auc_hi"],
        "baseline_c_pr_auc": baseline["pr_auc"], "baseline_c_pr_auc_lo": baseline["pr_auc_lo"], "baseline_c_pr_auc_hi": baseline["pr_auc_hi"],
    })
    per_fold, verdict = _apply_gate_rule(per_fold)

    # Descriptive genre view: the recorded scores of the five test tranches, pooled.
    scores = pd.concat([
        pd.DataFrame({
            "artist_id": fold["test"]["artist_id"].to_numpy(), "year": fold["test"]["year"].to_numpy(), "fold": fold["fold"],
            "label": fold["test"][target].to_numpy(), "model_score": scores_m, "c_score": scores_b,
        })
        for fold, scores_m, scores_b in zip(folds, recorded["model"], recorded["baseline_c"])
    ], ignore_index=True)
    pooled = scores.rename(columns={"label": "y", "model_score": "model", "c_score": "baseline_c"})
    pooled["primary_genre"] = pooled["artist_id"].map(_primary_genre(works)).fillna("(no valid genre)")
    rows = []
    for genre, group in [("all genres", pooled)] + list(pooled.groupby("primary_genre")):
        model_ap = average_precision_score(group["y"], group["model"])
        baseline_ap = average_precision_score(group["y"], group["baseline_c"])
        rows.append((genre, len(group), group["y"].mean(), model_ap, baseline_ap, (model_ap - baseline_ap) * 100))
    by_genre = pd.DataFrame(rows, columns=["primary_genre", "rows", "positive_rate", "model_pr_auc", "baseline_c_pr_auc", "difference_points"]).set_index("primary_genre")
    # Genres with enough rows, largest first, the overall row first and the no-genre group (if any) last.
    keep = by_genre[(by_genre["rows"] >= MIN_GENRE_ROWS) & ~by_genre.index.isin(["all genres", "(no valid genre)"])].sort_values("rows", ascending=False)
    by_genre = pd.concat([by_genre.loc[["all genres"]], keep, by_genre.loc[by_genre.index.isin(["(no valid genre)"])]])

    return {"folds": per_fold, "verdict": verdict, "by_genre": by_genre, "scores": scores}


def _gate_pr_auc(label, scores) -> float:
    """PR-AUC with the tie-aware function of the gate (`evaluate_folds`); NaN when one class only."""
    # Same sorted inputs and the same function as backtesting.evaluate_folds.
    return _weighted_average_precision(*_sorted_inputs(np.asarray(label, dtype=float), np.asarray(scores, dtype=float)))


def _check_scores_match_gate(gate: dict) -> None:
    """Stop unless the per-fold PR-AUC of the row-level scores equals the gate table to 6 decimals."""
    for fold, rows in gate["scores"].groupby("fold"):
        # Model and baseline (c), each against its column of the gate table.
        for score_column, gate_column in [("model_score", "model_pr_auc"), ("c_score", "baseline_c_pr_auc")]:
            from_scores = _gate_pr_auc(rows["label"], rows[score_column])
            assert abs(from_scores - gate["folds"].loc[fold, gate_column]) < 1e-6, f"the scores of fold {fold} do not reproduce the gate table ({gate_column})"


def _subset_metrics(rows: pd.DataFrame, scope_rows: int) -> dict:
    """Rows, share and PR-AUC / ROC-AUC of the model and of baseline (c) on a subset of the gate's test rows.

    `rows` has `label`, `model_score` and `c_score`; `scope_rows` is the size of the
    fold (or of the pooled folds) the subset belongs to. PR-AUC and ROC-AUC are NaN
    when the subset is empty or has one class only. Differences are in points.
    """
    label = rows["label"].to_numpy(dtype=float)
    both_classes = len(rows) > 0 and 0 < label.sum() < len(label)
    record = {"rows": len(rows), "share": len(rows) / scope_rows, "positive_rate": label.mean() if len(rows) else np.nan}
    # The same measures for the model and for baseline (c).
    for prefix, column in [("model", "model_score"), ("c", "c_score")]:
        record[f"{prefix}_pr_auc"] = _gate_pr_auc(label, rows[column]) if both_classes else np.nan
        record[f"{prefix}_roc_auc"] = roc_auc_score(label, rows[column]) if both_classes else np.nan
    record["pr_difference_points"] = (record["model_pr_auc"] - record["c_pr_auc"]) * 100
    record["roc_difference_points"] = (record["model_roc_auc"] - record["c_roc_auc"]) * 100
    return record


def gate_diagnostic_returning_artists(gate: dict, table: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic D1 (post-hoc, it cannot change the gate verdict): new versus returning artists.

    `gate` is the output of `run_what_gate` and `table` the output of
    `build_what_table`. The row-level scores of the gate are split, fold by fold, by
    whether the artist has any row in that fold's train table (`returning`) or not
    (`new`). The folds come from `make_what_folds`, so only development years are
    used. The scores must reproduce the gate table (asserted) and belong to the same
    test rows as the folds (asserted). No model is refitted.

    Returns a dataframe indexed by (`scope`, `subset`), scope being `fold k` or
    `pooled` and subset `new`, `returning` or `all`, with `rows`, `share` (of the
    scope), `positive_rate`, PR-AUC and ROC-AUC of the model and of baseline (c)
    (`model_pr_auc`, `c_pr_auc`, `model_roc_auc`, `c_roc_auc`) and the differences in
    points. ROC-AUC is shown next to PR-AUC because PR-AUC depends on the base rate
    of each subset.
    """
    _check_scores_match_gate(gate)
    flagged = []
    for fold in make_what_folds(table):
        rows = gate["scores"][gate["scores"]["fold"] == fold["fold"]]
        # The recorded scores are those of this fold's test rows, in the same order.
        assert len(rows) == fold["test_rows"], "the scores do not have the rows of the fold's test table"
        assert (rows["artist_id"].to_numpy() == fold["test"]["artist_id"].to_numpy()).all() and (rows["year"].to_numpy() == fold["test"]["year"].to_numpy()).all(), "the scores are not in the order of the fold's test table"
        # Returning: the artist has at least one row in this fold's own train table.
        flagged.append(rows.assign(returning=rows["artist_id"].isin(set(fold["train"]["artist_id"])).to_numpy()))
    scores = pd.concat(flagged, ignore_index=True)
    records = {}
    # Each fold, then all folds pooled.
    scopes = [(f"fold {k}", rows) for k, rows in scores.groupby("fold")] + [("pooled", scores)]
    for scope, rows in scopes:
        for subset, part in [("new", rows[~rows["returning"]]), ("returning", rows[rows["returning"]]), ("all", rows)]:
            records[(scope, subset)] = _subset_metrics(part, len(rows))
    return pd.DataFrame(records).T.rename_axis(["scope", "subset"])


def gate_diagnostic_next_year_volume(gate: dict, table: pd.DataFrame) -> dict:
    """Diagnostic D2 (post-hoc, it cannot change the gate verdict): the gate scores within next-year volume.

    `gate` is the output of `run_what_gate` and `table` the output of
    `build_what_table(..., include_diagnostics=True)`, which carries
    `n_works_next_year`. That column is joined to the row-level scores on
    (artist_id, year) only to stratify them (the row count must not change and no
    key may be missing, both asserted); it is a diagnostic, unknown at the cutoff,
    and must never be a model input (asserted against `WHAT_FEATURES`). Buckets are
    `VOLUME_BUCKET_EDGES` / `VOLUME_BUCKET_LABELS`. No model is refitted and only
    development years are used (the scores come from the development folds).

    Returns a dict of dataframes:
    - `by_bucket`: indexed by (`scope`, `bucket`), scope being `fold k` or `pooled` and
      bucket a volume bucket or `all`, with the measures of `_subset_metrics`: `rows`,
      `share` of the scope, `positive_rate`, PR-AUC and ROC-AUC of the model and of
      baseline (c) and their differences in points.
    - `ranking`: indexed by `scope`, the ROC-AUC of the model and of (c) on all rows
      (`*_roc_all`), the mean of the within-bucket ROC-AUCs (`*_roc_within_mean`) and
      the same mean weighted by bucket rows (`*_roc_within_weighted`), over the buckets
      that have both classes, and `*_gap_points` (all rows minus the weighted
      within-bucket value): how much of the ranking comes from separating buckets,
      that is volume, and not from ranking inside a bucket.
    """
    assert "n_works_next_year" not in WHAT_FEATURES, "n_works_next_year must never be a model input"
    _check_scores_match_gate(gate)
    keys = ["artist_id", "year"]
    scores = gate["scores"]
    # Join the diagnostic column on the keys with a reindex, so rows can neither be added nor reordered.
    lookup = table.set_index(keys)["n_works_next_year"]
    assert lookup.index.is_unique, "the table has repeated (artist_id, year) keys"
    wanted = pd.MultiIndex.from_frame(scores[keys])
    assert wanted.isin(lookup.index).all(), "some scored rows have no next-year Work count"
    joined = lookup.reindex(wanted)
    assert len(joined) == len(scores), "the join changed the row count"
    assert joined.notna().all(), "a scored row has no next-year Work count"
    scores = scores.assign(bucket=pd.cut(joined.to_numpy().astype(float), bins=VOLUME_BUCKET_EDGES, labels=VOLUME_BUCKET_LABELS))
    scopes = [(f"fold {k}", rows) for k, rows in scores.groupby("fold")] + [("pooled", scores)]
    records, ranking = {}, {}
    for scope, rows in scopes:
        within = {}  # ROC-AUC of each bucket that has both classes
        for bucket in list(VOLUME_BUCKET_LABELS) + ["all"]:
            part = rows if bucket == "all" else rows[rows["bucket"] == bucket]
            records[(scope, bucket)] = _subset_metrics(part, len(rows))
            if bucket != "all":
                within[bucket] = records[(scope, bucket)]
        # Within-bucket ROC-AUC against the ROC-AUC on all rows, for the model and for (c).
        record = {}
        for prefix in ["model", "c"]:
            valid = {b: m for b, m in within.items() if not np.isnan(m[f"{prefix}_roc_auc"])}
            roc = np.array([m[f"{prefix}_roc_auc"] for m in valid.values()])
            weight = np.array([m["rows"] for m in valid.values()], dtype=float)
            record[f"{prefix}_roc_all"] = records[(scope, "all")][f"{prefix}_roc_auc"]
            record[f"{prefix}_roc_within_mean"] = roc.mean()
            record[f"{prefix}_roc_within_weighted"] = (roc * weight).sum() / weight.sum()
            record[f"{prefix}_gap_points"] = (record[f"{prefix}_roc_all"] - record[f"{prefix}_roc_within_weighted"]) * 100
        ranking[scope] = record
    return {
        "by_bucket": pd.DataFrame(records).T.rename_axis(["scope", "bucket"]),
        "ranking": pd.DataFrame(ranking).T.rename_axis("scope"),
    }


# Feature groups of the ablation diagnostic (D4). Together they cover the 25 inputs of WHAT_FEATURES exactly once.
TABLE_INPUTS = WHAT_TABLE_INPUTS  # history_count and n_works_year, the inputs the baselines already use
VOCABULARY = ["n_styles_seen", "n_styles_year", "styles_per_work", "top_style_share", "n_genres_seen"]  # how wide the artist's style vocabulary is
NOVELTY = ["n_new_styles_year", "new_style_rate_so_far", "years_since_new_style"]  # how often and how recently the artist added styles
TIMING_FORMAT = ["years_since_prev_active_year", "years_active_so_far", "albums_share_year", "singles_share_year"]  # activity timing and release format
GENRE_FLAGS = [f"genre_{g}" for g in GENRES]  # one True / False column per genre of the year's Works
ABLATION_GROUPS = {"table_inputs": TABLE_INPUTS, "vocabulary": VOCABULARY, "novelty": NOVELTY, "timing_format": TIMING_FORMAT, "genre_flags": GENRE_FLAGS}


def gate_diagnostic_feature_ablation(gate: dict, table: pd.DataFrame, features: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP, confidence: float = KILL_GATE_CONFIDENCE, make_model=make_what_xgboost, groups: dict = None, all_features: list = None) -> dict:
    """Diagnostic D4 (post-hoc, it cannot change the gate verdict): feature-group ablation with refits.

    `gate` is the output of `run_what_gate`, `table` the output of `build_what_table` and
    `features` the output of `build_what_features`. The groups (`ABLATION_GROUPS` by
    default) must cover `all_features` (`WHAT_FEATURES`) exactly once, asserted. Runs: the
    full model (reused from the gate, not refitted; its row-level scores must reproduce the
    gate table, asserted), the full set minus each group except `table_inputs`, and
    `table_inputs` alone. Every refit uses the same folds, bootstrap size, confidence and
    seed as the gate, one configuration of `make_model`, through `evaluate_folds` with the
    recorded scores of the gate path (one score call per fold, asserted). Development
    years only; nothing is tuned.

    Returns a dict of dataframes:
    - `folds`: indexed by (`run`, `fold`), `n_features`, `pr_auc` with `pr_auc_lo` and
      `pr_auc_hi`, `ci_half_width_points`, `diff_vs_c_points` and `diff_vs_full_points`.
    - `summary`: indexed by `run` (including `full model (gate)` and `baseline (c)`),
      `n_features`, the mean over folds of PR-AUC and of the two differences, the pooled
      ROC-AUC on all test rows (`pooled_roc_auc_all_rows`) and its difference versus the
      full model in points.
    """
    groups = ABLATION_GROUPS if groups is None else groups
    all_features = WHAT_FEATURES if all_features is None else all_features
    # The groups must be a partition of the inputs: every input once, nothing else.
    listed = [name for columns in groups.values() for name in columns]
    assert len(listed) == len(set(listed)) and set(listed) == set(all_features) and len(listed) == len(all_features), "the groups do not cover the inputs exactly once"
    _check_scores_match_gate(gate)
    target = "new_style_year"
    folds = make_what_folds(table)
    labels = np.concatenate([fold["test"][target].to_numpy(dtype=float) for fold in folds])
    # Runs in report order: each is a list of input columns, in the order of all_features.
    runs = {f"minus {name}": [c for c in all_features if c not in columns] for name, columns in groups.items() if name != "table_inputs"}
    runs["table_inputs only"] = [c for c in all_features if c in groups["table_inputs"]]
    # The full model and baseline (c) come from the gate: point estimates, intervals and row-level scores.
    gate_folds = gate["folds"]
    results = {
        "full model (gate)": (gate_folds[["test_years"]].assign(pr_auc=gate_folds["model_pr_auc"], pr_auc_lo=gate_folds["model_pr_auc_lo"], pr_auc_hi=gate_folds["model_pr_auc_hi"]), gate["scores"]["model_score"].to_numpy(), len(all_features)),
    }
    assert (gate["scores"]["label"].to_numpy(dtype=float) == labels).all(), "the gate scores are not in the order of the fold test tables"
    for name, columns in runs.items():
        fit_fn, score_fn = make_model(features, columns)
        recorded = []
        evaluated = evaluate_folds(folds, target, fit_fn, _record_scores(score_fn, recorded), n_bootstrap, confidence)
        assert len(recorded) == len(folds), "the score function was not called exactly once per fold"
        results[name] = (evaluated[["test_years", "pr_auc", "pr_auc_lo", "pr_auc_hi"]], np.concatenate(recorded), len(columns))
    # Baseline (c) goes last, also from the gate.
    results["baseline (c)"] = (gate_folds[["test_years"]].assign(pr_auc=gate_folds["baseline_c_pr_auc"], pr_auc_lo=gate_folds["baseline_c_pr_auc_lo"], pr_auc_hi=gate_folds["baseline_c_pr_auc_hi"]), gate["scores"]["c_score"].to_numpy(), np.nan)
    # Per-fold table with the two differences, then the summary with the pooled ROC-AUC.
    full_pr = results["full model (gate)"][0]["pr_auc"]
    c_pr = results["baseline (c)"][0]["pr_auc"]
    parts, summary = {}, {}
    full_roc = roc_auc_score(labels, results["full model (gate)"][1])
    for name, (evaluated, scores, n_features) in results.items():
        per_fold = evaluated.copy()
        per_fold["n_features"] = n_features
        per_fold["ci_half_width_points"] = (per_fold["pr_auc_hi"] - per_fold["pr_auc_lo"]) / 2 * 100
        per_fold["diff_vs_c_points"] = (per_fold["pr_auc"] - c_pr) * 100
        per_fold["diff_vs_full_points"] = (per_fold["pr_auc"] - full_pr) * 100
        parts[name] = per_fold[["n_features", "pr_auc", "pr_auc_lo", "pr_auc_hi", "ci_half_width_points", "diff_vs_c_points", "diff_vs_full_points"]]
        roc = roc_auc_score(labels, scores)
        summary[name] = {"n_features": n_features, "mean_pr_auc": per_fold["pr_auc"].mean(), "mean_diff_vs_c_points": per_fold["diff_vs_c_points"].mean(),
                         "mean_diff_vs_full_points": per_fold["diff_vs_full_points"].mean(), "pooled_roc_auc_all_rows": roc, "roc_diff_vs_full_points": (roc - full_roc) * 100}
    return {"folds": pd.concat(parts, names=["run", "fold"]), "summary": pd.DataFrame(summary).T.rename_axis("run")}
