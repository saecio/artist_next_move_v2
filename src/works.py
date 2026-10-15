"""
Build the one row per Work table out of a Releases table.

A Work (a Discogs Master, in this project's current definition) can have
many Releases pointing at it: different countries, different formats,
reissues. Everything downstream of this file (temporal snapshots, WHEN,
WHAT) needs exactly one row per Work, not one row per Release, so this
module does that reduction once, in one place, instead of every notebook
repeating its own version of the same logic.

This is the same reduction that was first tried out by hand in an early
feasibility notebook that is no longer in the repository. It is a plain
function, so it can be reused on the full releases.parquet without
copying and pasting notebook cells again.

Usage:
    from src.works import build_works_table
    works = build_works_table(releases)
"""

from __future__ import annotations

import re  # tokenizing titles for the fuzzy-dedup number/roman-numeral guard

import pandas as pd
from rapidfuzz import fuzz  # similarity scoring used by the fuzzy-dedup filter

# Discogs' own top level genre tags for music that is out of scope for
# this project's WHEN and WHAT questions, not because it is lower value,
# but because both questions assume an artist has their own evolving,
# datable body of original work, and that assumption does not hold well
# for these two genres: Classical is mostly performances of compositions
# written long before the performer was credited, often by many different
# performers, and Brass & Military is dominated by ensembles and military
# bands rather than a single artist's own trajectory. This is a scope
# decision, not a quality judgement, and it is kept as an explicit,
# editable list right here rather than buried inside a function, so it is
# easy to question or extend later.
DEFAULT_EXCLUDED_GENRES = {"Classical", "Brass & Military"}

# The two placeholder artist credits Discogs uses when a release is not
# really tied to one identifiable artist. Neither one is a real person or
# group, so a row credited to either of these is flagged, not treated the
# same as a normal, named artist.
PLACEHOLDER_ARTIST_NAMES = {"Various", "Unknown Artist", "No Artist"}

# Word-chunk tokenizer, unicode-aware so accented letters stay inside a token.
# Used by the fuzzy-dedup number/roman-numeral guard below.
TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# A token made up only of roman-numeral letters (heuristic, short words
# like "mix" can also match this, see strip_numeric_content).
ROMAN_NUMERAL_RE = re.compile(r"^[ivxlcdm]+$")

# The offset added to a release_id to build a synthetic work's work_id
# (see notebook 01, where standalone_candidates["work_id"] is built this
# way). Repeated here, as its own constant, so the catalog-number filter
# below can invert it (work_id -> release_id) without guessing the value.
SYNTHETIC_WORK_ID_OFFSET = 10_000_000_000

# Catalog-number strings that really mean "no catalog number was ever
# entered", not an actual match. Treating two releases as the same just
# because both have a blank catno would create false positives (two
# unrelated self-releases with nothing in common except both being
# uncatalogued), so these values are never accepted as real evidence.
EMPTY_CATALOG_NUMBERS = {"none", "", "n/a"}


def strip_numeric_content(token: str) -> str:
    """Strip the numeric part out of a single token, so only its wording remains.

    A token that is entirely a roman numeral (e.g. "ii") is dropped
    outright, since it carries no wording on its own. Otherwise any
    digit run embedded anywhere in the token is removed (e.g. "mix4" ->
    "mix", "s01e03" -> "se"), so a number fused onto a word is treated
    the same as one that's its own token.
    """
    if ROMAN_NUMERAL_RE.fullmatch(token):  # the whole token is a sequence marker on its own
        return ""  # nothing left, it carries no wording
    return re.sub(r"\d+", "", token)  # drop any embedded digit run, keep the rest of the word as is


def differs_only_by_number_or_roman(title_a: str, title_b: str) -> bool:
    """Check whether two titles' wording is identical once numeric content is stripped.

    Tokenizes both titles, and if their raw token sets already match
    there is no numbered-series question to ask (that's a punctuation-only
    difference, not this guard's concern). Otherwise, numeric content is
    stripped from every token (standalone numbers, roman numerals, and
    digits fused onto a word all count) and the two stripped token sets
    are compared: if those now match, the only real difference was a
    sequence marker ("Vol. 2" vs "Vol. 3", "mix4" vs "mix7"), not wording.

    Known accepted limitation: "Vol." and "Volume" tokenize as different
    words, so a volume-number pair spelled out with both forms is not
    caught here (see docs/decisions.md).
    """
    tokens_a = set(TOKEN_RE.findall(title_a.lower()))  # this title's word tokens, lowercased
    tokens_b = set(TOKEN_RE.findall(title_b.lower()))  # the other title's word tokens, lowercased
    if tokens_a == tokens_b:  # no wording difference at all, not a numbered-series case
        return False  # leave this one for the caller to handle normally

    stripped_a = {strip_numeric_content(token) for token in tokens_a} - {""}  # this title's wording with numbers removed
    stripped_b = {strip_numeric_content(token) for token in tokens_b} - {""}  # the other title's wording with numbers removed
    return stripped_a == stripped_b  # true only if the only real difference was numeric content


def remove_fuzzy_duplicate_titles(
    works_df: pd.DataFrame,
    score_threshold: float = 90,
    window_size: int = 10,
) -> pd.DataFrame:
    """Collapse wording-level near-duplicate titles within the same artist_id.

    Catches near-duplicates the earlier exact-match title_normalized
    dedup cannot catch (typos, accents, punctuation, "and"/"&" swaps),
    without merging distinct numbered-series entries (different volumes,
    episodes, parts). The threshold, guard and this known limitation were
    all confirmed by EDA in notebooks/01_cleaning_eda.ipynb (bands 85-99
    sampled at full scale, with and without the guard) and logged in
    docs/decisions.md.

    Within each artist_id, titles are compared only against their next
    `window_size` neighbors in alphabetical order (not every other title
    in the group), which keeps the cost roughly O(n * window_size) per
    artist instead of O(n^2) and needs no per-artist size cap.

    A pair is treated as a true duplicate only if its token_sort_ratio
    score is >= `score_threshold` and it is not a numbered-series pair
    per `differs_only_by_number_or_roman`. Duplicates are collapsed by
    keeping the earlier `year` (missing year treated as the latest
    possible, ties broken by the smaller `work_id`).
    """
    works_df = works_df.reset_index(drop=True)  # fresh positional index, so iloc-style access below lines up with row order
    rows_to_drop = set()  # work_id values marked as the losing half of a duplicate pair

    for artist_id, group in works_df.groupby("artist_id"):  # comparisons only ever run within one artist's own titles
        titles = group["title"].tolist()  # that artist's titles, in their original row order
        work_ids = group["work_id"].tolist()  # matching work_id for each title, same order
        years = group["year"].tolist()  # matching year for each title, same order

        sort_order = sorted(range(len(titles)), key=lambda idx: titles[idx].lower())  # row positions sorted alphabetically, so near-duplicate titles land next to each other
        n = len(sort_order)  # this artist's group size, used to bound the comparison window

        for pos in range(n):  # walk the alphabetically sorted titles once
            i = sort_order[pos]  # original row position of the title at this sorted position
            if work_ids[i] in rows_to_drop:  # already marked as a duplicate loser by an earlier pair
                continue  # nothing new to learn from a title that is already being removed

            for pos2 in range(pos + 1, min(pos + 1 + window_size, n)):  # only the next window_size titles in sorted order
                j = sort_order[pos2]  # original row position of the nearby title being compared
                if work_ids[j] in rows_to_drop:  # already marked as a duplicate loser by an earlier pair
                    continue  # nothing new to learn by comparing against a title that is already being removed

                score = fuzz.token_sort_ratio(titles[i], titles[j])  # similarity between this one pair only
                if score < score_threshold or score >= 100:  # below the chosen threshold, or an exact match (handled upstream already)
                    continue  # not a pair this filter acts on

                if differs_only_by_number_or_roman(titles[i], titles[j]):  # a numbered-series pair (different volume/episode/part), not a true duplicate
                    continue  # leave both titles in place

                # True duplicate: keep the earlier year, missing year sorts as the
                # latest possible so a real year always wins over a missing one,
                # ties broken by the smaller work_id.
                key_i = (float("inf") if pd.isna(years[i]) else years[i], work_ids[i])  # this title's (year, work_id) sort key
                key_j = (float("inf") if pd.isna(years[j]) else years[j], work_ids[j])  # the other title's (year, work_id) sort key
                loser = i if key_i > key_j else j  # the later/worse-tie-broken key loses

                rows_to_drop.add(work_ids[loser])  # mark the losing half of this pair for removal
                if loser == i:  # i itself is being removed
                    break  # stop comparing it against further neighbors, it is going away either way

    return works_df[~works_df["work_id"].isin(rows_to_drop)].reset_index(drop=True)  # drop every marked duplicate, fresh index for downstream use


def _add_release_year(releases: pd.DataFrame) -> pd.DataFrame:
    """Add a `release_year` column, trimmed down from the `released` text field.

    Discogs allows three levels of date precision inside `released`: a
    full date ("1994-02-28"), a year and month only ("1994-02-00"), or
    just a year ("1994"). Comparing these as if they were all equally
    precise is misleading, so only the year is ever trusted here, the
    same decision already explained and tested in notebook 04. A value
    that does not start with a real four digit year (including a missing
    value) becomes a missing `release_year`, never a guessed one. A year past
    the snapshot date (2026, same bound notebook 01 already applied to
    standalone_candidates) is also treated as missing, it marks a
    pre-announced release rather than a real date.
    """
    releases = releases.copy()

    # Every one of the three date shapes above starts with a four digit
    # year, so reading just the first four characters is safe.
    first_four_characters = releases["released"].fillna("").str.slice(0, 4)
    looks_like_a_year = first_four_characters.str.fullmatch(r"[0-9]{4}")

    releases["release_year"] = pd.to_numeric(
        first_four_characters.where(looks_like_a_year), errors="coerce"
    )

    # Discogs uses "0000" as its own placeholder for "no real date", which
    # still passes the four digit check above and would otherwise become
    # the literal number 0 instead of a missing value.
    releases.loc[releases["release_year"] == 0, "release_year"] = pd.NA

    # Snapshot was pulled 2026-09-01, so a year past 2026 is a pre-announced
    # release, not an actual past date, the same exclusion notebook 01 already
    # applied directly to standalone_candidates, now shared by every caller.
    releases.loc[releases["release_year"] > 2026, "release_year"] = pd.NA

    return releases


def recover_missing_master_years(masters_df: pd.DataFrame, releases_df: pd.DataFrame) -> pd.Series:
    """Recover a Master's missing `year` from its own Releases, where possible.

    A Master's `year` sometimes never got filled in by Discogs, even though
    one or more of its Releases carry a usable date. This looks up only the
    Releases belonging to a Master with a missing year, parses each one with
    the same rule `_add_release_year` already applies everywhere else, and
    takes the earliest valid year as the recovered value. A Master stays
    missing when none of its Releases have a parseable, non-zero year.
    """
    # Only Masters with a missing year need a lookup at all, this keeps the
    # Release scan limited to the rows that can actually change anything.
    missing_year_master_ids = set(masters_df.loc[masters_df["year"].isna(), "master_id"])

    # Releases are only relevant here if they belong to one of those Masters,
    # the rest of releases_df (tens of millions of rows) is never touched.
    candidate_releases = releases_df[releases_df["master_id"].isin(missing_year_master_ids)].copy()

    # Reuse the exact same year-parsing rule already trusted for every other
    # release year in this project, so a recovered year means the same thing.
    candidate_releases = _add_release_year(candidate_releases)

    # Drop Releases with no parseable year before grouping, a Master with
    # zero usable Releases must stay missing, not collapse to NaT-as-min.
    usable_releases = candidate_releases.dropna(subset=["release_year"])

    # Earliest valid Release year per Master, the recovered value.
    recovered_years = usable_releases.groupby("master_id")["release_year"].min()

    return recovered_years


def _pick_representative_release(real_work_releases: pd.DataFrame) -> pd.DataFrame:
    """Pick exactly one release per Work to represent that Work.

    Discogs already marks one release per Work as the main release
    (`is_main_release`), which is exactly what we want: the release
    Discogs itself considers the best representative of that Work. That
    flag is used first.

    Some Works, especially in a small sample, will not have their main
    release available at all (the sample simply did not happen to draw
    that particular row). For those, we fall back to the earliest release
    by `release_year` (not the raw date text, to avoid the precision
    problem notebook 03 found: a year only date can sort as "earlier"
    than a same year date with a real day attached, purely because of how
    the text sorts, not because of the real chronology). Ties in that
    fallback are broken by the smallest `release_id`, an arbitrary but
    fixed choice, so the result is the same every time this runs.
    """
    main_releases = real_work_releases[real_work_releases["is_main_release"] == True].copy()

    works_with_main_release = set(main_releases["master_id"])
    all_real_works = set(real_work_releases["master_id"])
    works_missing_main_release = all_real_works - works_with_main_release

    fallback_candidates = real_work_releases[
        real_work_releases["master_id"].isin(works_missing_main_release)
    ].copy()

    # Sorting by Work, then year, then release_id means the first row for
    # each Work after this sort is exactly the one we want to keep.
    fallback_candidates = fallback_candidates.sort_values(
        ["master_id", "release_year", "release_id"]
    )
    fallback_rows = fallback_candidates.drop_duplicates(subset="master_id", keep="first")

    # We record where each row came from, so anyone reading the final
    # table can see, per Work, whether it came from Discogs' own main
    # release flag or from this fallback rule.
    main_releases["representative_source"] = "discogs_main_release"
    fallback_rows["representative_source"] = "earliest_year_fallback"

    return pd.concat([main_releases, fallback_rows], ignore_index=True)


def build_works_table(releases: pd.DataFrame) -> pd.DataFrame:
    """Reduce a Releases table down to one row per Work.

    Parameters
    ----------
    releases:
        A table shaped like `releases.parquet` (or the demo sample of it),
        with at least these columns: `release_id`, `master_id`,
        `is_main_release`, `released`, `title`, `artist_ids`,
        `artist_names`, `genres`, `styles`.

    Returns
    -------
    One row per real Work (`master_id` greater than 0), with columns:
    `work_id`, `artist_id`, `artist`, `is_various_or_unknown_artist`,
    `title`, `year`, `genre`, `style`, `representative_source`.

    `genre` and `style` are kept as `|` separated text (the same format
    already used by masters_demo.csv), not trimmed to a single tag, since
    a Work can genuinely carry more than one of each and dropping the
    rest would throw away real information.
    """
    releases = _add_release_year(releases)

    # master_id 0 is Discogs' way of saying "this release is not linked
    # to any Work at all". There is no Work to reduce those rows into, so
    # they are set aside here, the same as in notebook 04.
    real_work_releases = releases[releases["master_id"] > 0].copy()

    representative = _pick_representative_release(real_work_releases)

    def first_artist_is_placeholder(artist_names_list) -> bool:
        """True when the first credited artist is a placeholder name such as Various, False for an empty list. Input: the list of artist names of a Master. Output: a bool."""
        # An empty artist list should not happen in practice, but this
        # guards against it instead of assuming it never will.
        if len(artist_names_list) == 0:
            return False
        return artist_names_list[0] in PLACEHOLDER_ARTIST_NAMES

    works = pd.DataFrame({
        # The Work's own id is just master_id under a clearer name. This
        # project already treats a Master as the Work, so no new
        # identifier is being invented here.
        "work_id": representative["master_id"],

        # The first credited artist's id and name. A Work can be credited
        # to more than one artist, but the fields asked for here are
        # singular, one artist per row.
        "artist_id": representative["artist_ids"].apply(lambda ids: ids[0]),
        "artist": representative["artist_names"].apply(lambda names: names[0]),

        # A flag, not a filter. Whoever uses this table can decide
        # whether to exclude placeholder credited Works, this function
        # does not decide that for them.
        "is_various_or_unknown_artist": representative["artist_names"].apply(
            first_artist_is_placeholder
        ),

        "title": representative["title"],
        "year": representative["release_year"],
        "genre": representative["genres"].apply(lambda tags: "|".join(tags)),
        "style": representative["styles"].apply(lambda tags: "|".join(tags)),
        "representative_source": representative["representative_source"],
    })

    return works


def _label_catno_pairs(labels) -> frozenset:
    """Turn one release's `labels` list of dicts into a comparable set.

    Each entry in `labels` is a dict like {"name": ..., "catno": ...}; a
    release can carry more than one (a co-release on two labels, for
    example), so the whole list is kept as a set of (name, catno) tuples
    rather than picking just one. A frozenset is used so two releases'
    label sets can be intersected directly to check for any shared pair.
    """
    if labels is None or (isinstance(labels, float) and pd.isna(labels)):  # no labels recorded at all (NaN shows up as a float, same situation handled for format_descriptions)
        return frozenset()  # nothing to compare against
    return frozenset((entry["name"], entry["catno"]) for entry in labels)  # one (name, catno) tuple per label credited on this release


def _has_real_catalog_number(label_catno_pair) -> bool:
    """Check whether a (name, catno) pair carries an actual catalog number.

    "none", "" and "n/a" are Discogs/community placeholders for "nothing
    was entered here", not a real identifier, so a pair using one of
    these must never count as evidence that two releases are the same
    item (see EMPTY_CATALOG_NUMBERS above).
    """
    _, catno = label_catno_pair  # the catalog-number half of the pair, the label name is not relevant to this check
    return catno is not None and str(catno).strip().lower() not in EMPTY_CATALOG_NUMBERS  # blank/placeholder catnos are not real evidence


def remove_catalog_number_duplicates(works_df: pd.DataFrame, releases_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse same-title, different-year Works that share a real catalog number.

    Targets the ambiguity the existing reissue-tag filter cannot see: a
    Work and a later Work by the same artist, same `title_normalized`,
    different `year`, where Discogs never tagged either release
    Reissue/Repress/Remaster. Scoped by EDA in notebooks/01_cleaning_eda.ipynb
    to exactly that population (55,642 (artist_id, title_normalized)
    groups, 122,569 rows), rather than every pair of same-artist works,
    which keeps the (expensive) label/catalog-number lookup small.

    Two Works in that population are treated as the same recording
    re-released only if their representative release (the one already
    used for format_descriptions, for a master Work) or their own
    release (for a synthetic Work) shares at least one (label name,
    catalog number) pair, and that pair carries a real catalog number,
    not a blank/placeholder one (see `_has_real_catalog_number`). A
    30-row sample at this threshold (docs/decisions.md) was uniformly
    clean; groups whose only shared pair has a blank catno are left
    alone, since that is not real evidence (2,943 groups in the EDA,
    deliberately excluded for that reason).

    A matched group can have more than two members (e.g. three Works,
    only two of which actually share a catalog number), so matches are
    resolved as connected components within each group, not just
    pairwise: every Work connected, directly or transitively, by a
    shared catalog number collapses into one survivor. The survivor is
    chosen the same way as `remove_fuzzy_duplicate_titles`: the earliest
    `year` (a missing year sorts as the latest possible, so a real year
    always wins), ties broken by the smaller `work_id`.

    Parameters
    ----------
    works_df:
        Must already carry `title_normalized` and `is_reissue_tagged`
        (both computed earlier in notebook 01, before this filter runs).
    releases_df:
        The raw Releases table (or a column subset of it), used to look
        up `labels` by `master_id` (via the same representative-release
        logic as `_pick_representative_release`) or by `release_id`
        (for synthetic Works, whose `work_id` is `release_id +
        SYNTHETIC_WORK_ID_OFFSET`).
    """
    # Step 1: find the ambiguous population this filter targets, exactly
    # as scoped by the EDA: same artist_id + title_normalized, 2 or more
    # distinct years, and no row in the group tagged reissue/repress/remaster.
    grouped = works_df.groupby(["artist_id", "title_normalized"])
    year_counts = grouped["year"].nunique(dropna=True)  # how many distinct years each group has
    multi_year_groups = year_counts[year_counts >= 2].index  # groups with a real different-year ambiguity

    any_tagged = grouped["is_reissue_tagged"].any()  # whether any row in the group already has the format_descriptions tag
    untagged_groups = any_tagged[~any_tagged].index  # groups the existing reissue filter has no visibility into

    target_groups = multi_year_groups.intersection(untagged_groups)  # the exact population this filter is scoped to

    group_index = works_df.set_index(["artist_id", "title_normalized"]).index  # row-aligned (artist_id, title_normalized) lookup
    candidate_rows = works_df[group_index.isin(target_groups)].copy()  # only the rows inside a target group get the expensive label lookup

    if candidate_rows.empty:  # nothing in scope (safety net, not expected given the EDA numbers)
        return works_df  # unchanged, no work_ids to resolve

    # Step 2: split the candidate rows by how to find their own release
    # record: a master Work's representative release is looked up by
    # master_id, a synthetic Work's single release is looked up directly
    # by release_id (its own work_id minus the synthetic offset).
    is_master = candidate_rows["work_source"] == "master"
    master_ids_needed = set(candidate_rows.loc[is_master, "work_id"])
    release_ids_needed = set(candidate_rows.loc[~is_master, "work_id"] - SYNTHETIC_WORK_ID_OFFSET)

    # Only the releases that could possibly matter are pulled out of the
    # (much larger) full releases table, so this stays cheap even though
    # releases_df itself can be tens of millions of rows.
    relevant_releases = releases_df[
        releases_df["master_id"].isin(master_ids_needed) | releases_df["release_id"].isin(release_ids_needed)
    ].copy()

    # Step 3: build a release_id -> label set lookup for the synthetic
    # side directly (one release per synthetic Work, no picking needed).
    synthetic_releases = relevant_releases[relevant_releases["release_id"].isin(release_ids_needed)]
    synthetic_label_sets = synthetic_releases.set_index("release_id")["labels"].apply(_label_catno_pairs)

    # Step 4: build a master_id -> label set lookup for the master side,
    # reusing the exact same representative-release rule already applied
    # to pick the release format_descriptions came from, so this stays
    # consistent with the rest of the pipeline rather than inventing a
    # second way to pick "the" release for a master.
    master_releases = relevant_releases[relevant_releases["master_id"].isin(master_ids_needed)].copy()
    master_releases = _add_release_year(master_releases)
    representative = _pick_representative_release(master_releases)
    representative = representative.drop_duplicates(subset="master_id", keep="first")  # one row per master_id, matching _pick_representative_release's own contract
    master_label_sets = representative.set_index("master_id")["labels"].apply(_label_catno_pairs)

    def label_set_for_row(row) -> frozenset:
        """Look up the label and catalog number set of one Work by its work_source (master or synthetic). Input: a row of the Works table. Output: a frozenset, empty when no release is found."""
        # Looks up this row's label set from whichever lookup applies to
        # its work_source; .get(..., frozenset()) means a Work with no
        # matching release found (should not happen, but not assumed)
        # simply has nothing to match on, instead of raising.
        if row["work_source"] == "master":
            return master_label_sets.get(row["work_id"], frozenset())
        return synthetic_label_sets.get(row["work_id"] - SYNTHETIC_WORK_ID_OFFSET, frozenset())

    candidate_rows["label_set"] = candidate_rows.apply(label_set_for_row, axis=1)

    # Step 5: within each target group, connect any two rows that share a
    # real catalog number, then collapse each connected component (not
    # just each pair) down to one survivor. Groups are small (almost
    # always 2-4 rows, per the EDA), so a plain O(n^2) comparison per
    # group is cheap; there is no need for anything fancier here.
    rows_to_drop = set()  # work_id values marked as the losing half of a collapsed component

    for _, group in candidate_rows.groupby(["artist_id", "title_normalized"]):
        label_sets = group["label_set"].tolist()
        work_ids = group["work_id"].tolist()
        years = group["year"].tolist()
        n = len(group)

        # Union-find over this group's local row positions (0..n-1), so a
        # three-way match (A-B share a catno, B-C share a different one)
        # still ends up as one component, not two separate pairs.
        parent = list(range(n))

        def find(i):
            """Find the root position of `i` in the union-find `parent` list. Input: a row position. Output: the root position of its component."""
            while parent[i] != i:  # walk up to this position's current root
                i = parent[i]
            return i

        def union(i, j):
            """Merge the components of positions `i` and `j` in the union-find `parent` list. Inputs: two row positions. Output: none, `parent` is updated."""
            root_i, root_j = find(i), find(j)
            if root_i != root_j:  # only link roots that are not already in the same component
                parent[root_i] = root_j

        for i in range(n):
            if not label_sets[i]:  # nothing to compare, this Work has no known label at all
                continue
            for j in range(i + 1, n):
                shared = label_sets[i] & label_sets[j]  # (name, catno) pairs both releases carry
                if any(_has_real_catalog_number(pair) for pair in shared):  # a real, non-blank catalog number is shared
                    union(i, j)

        # Group the local positions by their component root, so each
        # component with 2+ members can be collapsed to one survivor.
        components: dict[int, list[int]] = {}
        for i in range(n):
            components.setdefault(find(i), []).append(i)

        for member_positions in components.values():
            if len(member_positions) < 2:  # no match at all, this row is already fine on its own
                continue

            # Same tie-break as remove_fuzzy_duplicate_titles: earliest
            # year wins, missing year treated as the latest possible so a
            # real year always wins over a missing one, ties broken by
            # the smaller work_id.
            def sort_key(pos):
                """Sort key of a row position: its year (missing years last), then its work_id. Input: a row position. Output: a tuple."""
                year = years[pos]
                return (float("inf") if pd.isna(year) else year, work_ids[pos])

            survivor_pos = min(member_positions, key=sort_key)
            for pos in member_positions:
                if pos != survivor_pos:
                    rows_to_drop.add(work_ids[pos])

    return works_df[~works_df["work_id"].isin(rows_to_drop)].reset_index(drop=True)


def remove_mass_upload_artists(
    works_df: pd.DataFrame,
    min_work_count: int = 50,
    synthetic_share_threshold: float = 0.90,
    roman_numeral_share_threshold: float = 0.50,
    busiest_2yr_share_threshold: float = 0.80,
) -> pd.DataFrame:
    """Exclude artist_ids whose catalog reads as a bulk/algorithmic upload, not a discography.

    Found via EDA in notebooks/01_cleaning_eda.ipynb: "Nosferatu 1922"
    (artist_id 6391827, 1,854 works) stood out as a volume outlier with
    three signals together, none of which show up together for any
    genuinely prolific artist checked (Bing Crosby, Elvis Presley, Frank
    Sinatra, Duke Ellington): almost all of its works are synthetic
    (standalone releases, not Discogs Masters), almost all of its titles
    are bare roman numerals (e.g. "MCCCLXXIII", no actual wording), and
    almost all of them landed inside a single two-year window, not
    spread across a real career.

    This is written as a general rule on those three signals, not a
    hardcoded exclusion of artist_id 6391827, so it can catch another
    artist with the same signature without needing a code change. A
    `min_work_count` floor is required because, below it, the other two
    signals stop meaning anything: an artist with only 1-2 works almost
    always has every work inside one 2-year window, and a single
    roman-numeral-only title (a real but obscure stage name, e.g. "XIX")
    already clears a 50% share on its own. 50 was chosen as the floor
    because it sits at roughly the 99th percentile of works-per-artist
    (see the univariate EDA in notebooks/02_eda_visualization.ipynb),
    i.e. this filter only ever looks at the small population of
    unusually high-volume artists the signature was actually built to
    describe. Checked at that floor across the full works_df (not just
    the top few hundred by volume, since the three signals are cheap to
    compute): see docs/decisions.md for the resulting artist/row counts.

    Parameters
    ----------
    works_df:
        Must already carry `title_normalized` (computed earlier in
        notebook 01).
    """
    def artist_signature(group: pd.DataFrame) -> pd.Series:
        """Compute the signals of one artist_id group: work count, synthetic share, bare roman numeral title share and busiest 2-year share. Input: the Works of one artist. Output: a Series with those four values."""
        # One row's worth of the three signals for one artist_id's group
        # of works, returned together so groupby/apply can build one
        # signature table in a single pass instead of three separate
        # groupby calls.
        synthetic_share = (group["work_source"] == "synthetic").mean()  # share of this artist's works that are standalone releases, not Masters

        # A title counts as "bare roman numeral" only if the whole
        # normalized title is roman-numeral letters, not just a title
        # that happens to contain one (ROMAN_NUMERAL_RE anchors both
        # ends), so a real title like "Live In Rio" is never caught here.
        roman_numeral_share = group["title_normalized"].str.strip().apply(
            lambda title: bool(ROMAN_NUMERAL_RE.fullmatch(title))
        ).mean()

        years = group["year"].dropna()
        if len(years) == 0:  # no dated works at all, no window to compute
            busiest_2yr_share = 0.0  # can't be a year-concentration outlier with no years to concentrate
        else:
            # Count works per year, then take the best possible
            # consecutive 2-year total (a rolling sum over every year in
            # the artist's own min-max range, including years with zero
            # works, so a gap does not break the window), and express it
            # as a share of this artist's dated works.
            works_per_year = years.value_counts().sort_index()
            full_year_range = range(int(years.min()), int(years.max()) + 2)
            works_per_year = works_per_year.reindex(full_year_range, fill_value=0)
            busiest_2yr_count = works_per_year.rolling(2).sum().max()
            busiest_2yr_share = busiest_2yr_count / len(years)

        return pd.Series({
            "work_count": len(group),
            "synthetic_share": synthetic_share,
            "roman_numeral_share": roman_numeral_share,
            "busiest_2yr_share": busiest_2yr_share,
        })

    signatures = works_df.groupby("artist_id").apply(artist_signature, include_groups=False)

    # The actual rule: high volume (the floor that makes the other two
    # signals meaningful) and all three signals at once, not any single
    # one, since each signal alone has a legitimate explanation (a
    # prolific but normal artist can be mostly-synthetic or have one
    # busy two-year run; it's the combination that is the red flag).
    flagged_artist_ids = signatures[
        (signatures["work_count"] >= min_work_count)
        & (signatures["synthetic_share"] >= synthetic_share_threshold)
        & (signatures["roman_numeral_share"] >= roman_numeral_share_threshold)
        & (signatures["busiest_2yr_share"] >= busiest_2yr_share_threshold)
    ].index

    return works_df[~works_df["artist_id"].isin(flagged_artist_ids)].reset_index(drop=True)

