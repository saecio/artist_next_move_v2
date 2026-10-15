"""
Build temporal snapshots and the WHEN / WHAT targets from a Works table.

A "snapshot" here means: for one artist, at one point in their career
(right after one of their Works), what did their history look like up to
that point, and what happened afterwards. This is the piece
docs/decisions.md calls mandatory: every prediction in this project has to
be made as if standing at that point in time, using only what had already
happened, never information from the future.

The unit is the individual Work, but the WHEN target looks at years, not at
Works. About 63% of the dates in the data are year-only, so several Works in
the same year have no reliable order between them, and "next Work" would
often be a same-year sibling (a gap of 0 years that says nothing about
forecasting). So WHEN measures the gap to the artist's next year with
activity after the snapshot's own year.

This module reads no data files (only config.yaml, through `config.py`)
and does not know about Discogs. It only takes a Works table shaped like notebook 01's real output checkpoint
(`data/processed/works_checkpoint.parquet`), one row per Work, and turns it
into one row per snapshot.

Usage:
    from src.snapshots import build_snapshots
    snapshots = build_snapshots(works)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG  # project settings read from config.yaml

# The Discogs dump is dated 2026-09-01, so 2026 is only partly observed. The
# last year that is fully observed (config.yaml, years.last_complete) is 2025,
# and it is the boundary used to decide whether "no next active year" is a
# real 0 or simply not known yet.
LAST_COMPLETE_YEAR = CONFIG["years"]["last_complete"]

# Seed for the order of Works inside one artist-year. The data only knows the
# year, so that order is not real chronology; sorting same-year Works by
# work_id would put every master before every synthetic Work (synthetic ids
# carry a 10_000_000_000 offset), and the "so far" counts would partly encode
# work_source. Instead the order follows a hash of work_id plus this seed: it
# looks random, but it is identical on every run and does not depend on the
# row order of the input. It is the project seed of config.yaml (project.SEED).
WITHIN_YEAR_ORDER_SEED = CONFIG["project"]["SEED"]

# Discogs format_descriptions tags that mark an album-like or a single-like
# release. Everything else (including no tag at all) is "other".
ALBUM_FORMATS = {"Album", "LP", "EP", "Mini-Album"}
SINGLE_FORMATS = {"Single", "Maxi-Single"}


def _split_styles(styles_value) -> set:
    """Turn a list-valued cell (styles, format_descriptions) into a set of tags.

    `styles_value` is whatever the cell actually holds: a list (or numpy
    array) of tags, `None`, or NaN when nothing was recorded at all. Any of
    those missing forms becomes an empty set, not a set containing a stray
    tag, so "no tags" and "one real tag" are never confused with each other.
    """
    # None and NaN (a bare float, pandas' missing-value marker here) both
    # mean "nothing recorded"; a list/array is never a float, so this
    # check cannot misfire on a real, non-empty list.
    if styles_value is None or isinstance(styles_value, float):
        return set()
    # An empty list/array also means "nothing recorded", same outcome
    # as the None/NaN case above, just a different shape for it.
    if len(styles_value) == 0:
        return set()
    # Otherwise it is a real list (or array) of tags, turn it into a set.
    return set(styles_value)


def classify_work_type(format_descriptions) -> str:
    """Normalize a Work's format_descriptions into "album", "single" or "other".

    Album-like (Album, LP, EP, Mini-Album) wins over single-like (Single,
    Maxi-Single) when a release carries both kinds of tag, and anything
    without either kind, including an empty list, is "other".
    """
    # Same list-to-set conversion used for styles, so missing values are safe.
    tags = _split_styles(format_descriptions)
    # Album-like tags take precedence.
    if tags & ALBUM_FORMATS:
        return "album"
    # Then single-like tags.
    if tags & SINGLE_FORMATS:
        return "single"
    # Everything else, including no tags at all.
    return "other"


def build_snapshots(works: pd.DataFrame, last_complete_year: int = LAST_COMPLETE_YEAR) -> pd.DataFrame:
    """Turn a one row per Work table into one row per temporal snapshot.

    Every Work with a known year becomes one snapshot, anchored at that
    Work, for the artist who made it. The snapshot describes what was
    knowable at that point (how much history the artist already had) and,
    separately, the targets this project is trying to predict (WHEN and
    WHAT), built only from what actually happened afterwards in the data.

    Works with a missing `year` never get a snapshot: they cannot be placed
    in time. The count dropped is returned in
    `result.attrs["dropped_missing_year"]`, rather than silently lowering
    every count downstream. Works with no `artist_id` are dropped the same
    way and counted in `result.attrs["dropped_missing_artist"]`.

    Parameters
    ----------
    works:
        One row per Work, shaped like notebook 01's real output checkpoint:
        `artist_id`, `work_id`, `year`, `work_source`, `genres`, `styles`
        and `format_descriptions` (the last three are lists of tags).
    last_complete_year:
        The last calendar year fully observed in the data, used for the
        censoring rule below.

    Returns
    -------
    One row per snapshot, sorted by artist and year, with same-year Works in
    a seeded pseudo-random order (see `WITHIN_YEAR_ORDER_SEED`; identical on
    every run), with columns:

    - `artist_id`, `work_id`, `year`, `cutoff_year`: `cutoff_year` is the
      same value as `year`, kept as its own column because it is the
      boundary between "known" and "not yet known".
    - `work_source`, `genres`: passed through from the Work.
    - `works_so_far_this_year`: how many Works the artist published in the
      same year as this snapshot up to and including this one, in the sort
      order. This is the productivity feature that is safe as a model input.
    - `works_this_year`: how many Works the artist published in the whole
      year of this snapshot. DESCRIPTIVE / EDA ONLY, NOT SAFE AS A MODEL
      INPUT: it counts same-year Works that come after this one in the sort
      order, which leaks information from after the snapshot.
    - `history_count`: how many earlier Works this artist already had,
      strictly before this one in the sort order.
    - `years_since_last`: years since the artist's previous Work, missing
      for an artist's very first known Work. 0 when the previous Work is
      from the same year.
    - `years_active_so_far`: years between the artist's very first known
      Work and this one, 0 for a first Work.
    - `historical_rate`: `history_count` divided by `years_active_so_far`.
      Missing when no years have elapsed, dividing by zero years has no
      sensible answer, so it is left unknown rather than forced to 0.
    - `next_1y`, `next_2y`: 1 if the artist's next year with activity after
      this Work's year is within 1 (or 2) years of it, 0 if we can be
      confident it was not, and missing if we have not observed enough time
      afterwards to be sure (see the censoring note below). Same-year Works
      are never the "next", so the gap is always >= 1.
    - `new_style`: 1 if the artist's next Work (the next row in the sort
      order, which can be a same-year Work) used at least one style tag
      never seen in any of their Works up to and including this one, 0 if
      not, missing if there is no next Work, or if the next Work has no
      style tags recorded.
    - `artist_undated_work_count`: how many of this artist's Works have no
      known year, the same number on every snapshot row of that artist.
      There is no reliable signal to say which gap an undated Work falls
      into, so it is reported at the artist level.
    - `year_missing`: True when `artist_undated_work_count` > 0, the same
      artist-level flag `src/eda.py` uses.
    - `is_album_like`, `is_single_like`: this Work's type (see
      `classify_work_type`; both False means "other").
    - `albums_so_far_this_year`, `singles_so_far_this_year`: how many
      album-like and single-like Works the artist published in this
      snapshot's year up to and including this one. Safe as model inputs.
    - `albums_this_year`, `singles_this_year`: the same counts over the whole
      year. DESCRIPTIVE / EDA ONLY, NOT SAFE AS MODEL INPUTS, same leak as
      `works_this_year`.

    A note on the undated-Works style baseline:

    An artist's undated Works never get their own snapshot, but a style tag
    can still have been used on one of them. Without accounting for this, a
    style used only on an undated Work and then again on a dated one would
    wrongly look "new". So every undated Work's styles are folded into the
    artist's seen styles as a starting baseline. This is a deliberate,
    conservative choice: it trades away a rare true "new style" positive
    for avoiding a much more common false one.

    A note on censoring `next_1y` / `next_2y`:

    If an artist's last active year is 2024 and the data is only complete
    through 2025, we cannot tell whether they stopped or just have not
    released anything yet inside a 2 year window. So with no next active
    year, the targets are 0 only when enough fully observed years follow
    the snapshot (`last_complete_year`), and missing otherwise. 2026 is
    only partly observed (the dump is dated 2026-09-01), so it is never
    treated as a complete year, though a Work found in it still counts as
    a real next active year.
    """
    # A Work with no artist_id cannot belong to any artist's timeline (95 in
    # the real checkpoint), so it is dropped up front and counted, not lost
    # silently inside a groupby.
    has_artist = works["artist_id"].notna()
    dropped_missing_artist = int((~has_artist).sum())
    works = works[has_artist]

    # A Work with no known year cannot be placed on a timeline.
    year_missing_work = works["year"].isna()
    # How many Works were dropped for that reason, reported on the result.
    dropped_missing_year = int(year_missing_work.sum())

    # Kept aside so the two aggregates below still see every undated Work.
    undated_works = works[year_missing_work]
    # How many undated Works each artist has, reported on every snapshot row.
    undated_work_count_by_artist = undated_works.groupby("artist_id").size()
    # Every style used on any of an artist's undated Works, one set per artist.
    undated_styles_by_artist = undated_works.groupby("artist_id")["styles"].apply(
        lambda styles_column: set().union(*(_split_styles(s) for s in styles_column))
    )

    # From here on only dated Works, with year as a plain integer.
    snapshots = works[~year_missing_work].copy()
    snapshots["year"] = snapshots["year"].astype(int)
    # Pseudo-random but reproducible rank of each Work inside its year: a hash
    # of work_id shifted by the fixed seed, so it ignores the input row order.
    snapshots["within_year_rank"] = pd.util.hash_array(snapshots["work_id"].to_numpy() + WITHIN_YEAR_ORDER_SEED)
    # Chronological order per artist, then year, then the seeded rank; work_id
    # only settles the (practically impossible) case of equal hashes.
    snapshots = snapshots.sort_values(["artist_id", "year", "within_year_rank", "work_id"]).reset_index(drop=True)

    # Per-artist views of the sorted table, for the vectorised features.
    by_artist = snapshots.groupby("artist_id", sort=False)
    # Same value as year, kept under its own name as the known/unknown boundary.
    snapshots["cutoff_year"] = snapshots["year"]

    # Works strictly before this one in the sort order.
    snapshots["history_count"] = by_artist.cumcount()
    # Years since the previous Work (0 for a same-year sibling) and since the first.
    snapshots["years_since_last"] = snapshots["year"] - by_artist["year"].shift(1)
    snapshots["years_active_so_far"] = snapshots["year"] - by_artist["year"].transform("first")
    # Rate of Works per elapsed year, unknown when no years have elapsed.
    snapshots["historical_rate"] = np.where(
        snapshots["years_active_so_far"] > 0,
        snapshots["history_count"] / snapshots["years_active_so_far"].where(snapshots["years_active_so_far"] > 0),
        np.nan,
    )

    # Year-level activity: the distinct years each artist has at least one Work in.
    active_years = snapshots[["artist_id", "year"]].drop_duplicates()
    # The artist's next active year after each of those years (NaN for the last).
    active_years["next_active_year"] = active_years.groupby("artist_id")["year"].shift(-1)
    # Back onto every Work; a left merge keeps the sorted order of snapshots.
    snapshots = snapshots.merge(active_years, on=["artist_id", "year"], how="left")

    # Works in the snapshot's own year: up to and including this one (model-safe),
    # and the whole year (descriptive only, it counts later same-year Works).
    snapshots["works_so_far_this_year"] = snapshots.groupby(["artist_id", "year"]).cumcount() + 1
    snapshots["works_this_year"] = snapshots.groupby(["artist_id", "year"])["work_id"].transform("size")

    # WHEN targets: gap to the next active year, always >= 1 by construction.
    has_next = snapshots["next_active_year"].notna()
    gap_years = snapshots["next_active_year"] - snapshots["year"]
    # Fully observed years that follow this snapshot, for the censoring rule.
    observed_after = last_complete_year - snapshots["year"]
    # With a next active year the target is measured; without one it is a
    # confident 0 only when enough complete years followed, else unknown.
    snapshots["next_1y"] = np.where(has_next, (gap_years <= 1).astype(float), np.where(observed_after >= 1, 0.0, np.nan))
    snapshots["next_2y"] = np.where(has_next, (gap_years <= 2).astype(float), np.where(observed_after >= 2, 0.0, np.nan))

    # WHAT target needs a running set of styles per artist, so one pass in Python.
    artist_ids = snapshots["artist_id"].to_numpy()
    work_styles = snapshots["styles"].map(_split_styles).tolist()
    new_style = np.full(len(snapshots), np.nan)
    current_artist = None
    styles_seen_so_far: set = set()
    for i in range(len(snapshots)):
        # A new artist starts from their undated-Works styles, not from empty.
        if artist_ids[i] != current_artist:
            current_artist = artist_ids[i]
            styles_seen_so_far = set(undated_styles_by_artist.get(current_artist, set()))
        # This Work's styles are known at its cutoff, so they count as seen.
        styles_seen_so_far |= work_styles[i]
        # Compare against the next Work only when it exists, is the same
        # artist, and has styles recorded (otherwise "new" is unknowable).
        if i + 1 < len(snapshots) and artist_ids[i + 1] == current_artist and work_styles[i + 1]:
            new_style[i] = float(len(work_styles[i + 1] - styles_seen_so_far) > 0)
    snapshots["new_style"] = new_style

    # Artist-level undated-Works count and flag, repeated on every row.
    snapshots["artist_undated_work_count"] = (
        snapshots["artist_id"].map(undated_work_count_by_artist).fillna(0).astype(int)
    )
    snapshots["year_missing"] = snapshots["artist_undated_work_count"] > 0

    # Work type of each Work, then the same counts summed over its year.
    work_type = snapshots["format_descriptions"].map(classify_work_type)
    snapshots["is_album_like"] = work_type == "album"
    snapshots["is_single_like"] = work_type == "single"
    year_groups = snapshots.groupby(["artist_id", "year"])
    # Running counts up to and including this Work (model-safe) and whole-year
    # counts (descriptive only, they include later same-year Works).
    snapshots["albums_so_far_this_year"] = year_groups["is_album_like"].cumsum()
    snapshots["singles_so_far_this_year"] = year_groups["is_single_like"].cumsum()
    snapshots["albums_this_year"] = year_groups["is_album_like"].transform("sum")
    snapshots["singles_this_year"] = year_groups["is_single_like"].transform("sum")

    # Column order: identity, passthrough, features, targets, flags.
    snapshots = snapshots[[
        "artist_id", "work_id", "year", "cutoff_year", "work_source", "genres",
        "works_so_far_this_year", "albums_so_far_this_year", "singles_so_far_this_year",
        "is_album_like", "is_single_like", "works_this_year", "albums_this_year", "singles_this_year",
        "history_count", "years_since_last", "years_active_so_far", "historical_rate",
        "next_1y", "next_2y", "new_style", "artist_undated_work_count", "year_missing",
    ]]
    # Bookkeeping that would otherwise be silently lost.
    snapshots.attrs["dropped_missing_year"] = dropped_missing_year
    snapshots.attrs["dropped_missing_artist"] = dropped_missing_artist
    snapshots.attrs["last_complete_year"] = last_complete_year
    return snapshots
