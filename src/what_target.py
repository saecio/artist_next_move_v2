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

from eda import PRIOR_WORK_BUCKET_LABELS, _bucket_by_prior_works  # same history buckets as the Work-level EDA
from snapshots import _split_styles  # same list-to-set conversion as the Work-level target


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


# First year of the sealed holdout (same value as HOLDOUT_START_YEAR in
# src/backtesting.py, repeated here so this module does not import matplotlib).
# The cross-check only uses development years, strictly before it.
HOLDOUT_START_YEAR = 2018


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
    d_artist = work_snapshots["artist_id"].diff()
    d_year = work_snapshots["year"].diff()
    # A row is in order when it opens the file, starts a later artist, or has a year not before the previous one.
    in_order = d_artist.isna() | (d_artist > 0) | ((d_artist == 0) & (d_year >= 0))
    assert in_order.all(), "work_snapshots is not sorted by artist_id then year"

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


# Last year of the sealed holdout window (same value as HOLDOUT_END_YEAR in
# src/backtesting.py; 2024 and later are not used by any evaluation).
HOLDOUT_END_YEAR = 2023
# Test tranches of the WHEN folds: 5 consecutive windows of 5 years that end
# right before the holdout (1993-1997 ... 2013-2017 for a holdout starting in 2018).
N_FOLD_TRANCHES = 5
TRANCHE_YEARS = 5


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


# Bucket edges for the number of Works in one year (right-inclusive): 1, 2, 3-4, 5+.
VOLUME_BUCKET_EDGES = [0, 1, 2, 4, float("inf")]
VOLUME_BUCKET_LABELS = ["1", "2", "3-4", "5+"]


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
