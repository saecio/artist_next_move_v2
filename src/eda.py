"""
Reusable plotting and summary functions for the EDA visualization notebook.

Notebooks only call these functions, they do not contain plotting or
analysis logic inline.
"""

from __future__ import annotations

import itertools  # all pairs of columns for the correlation table
import random  # local RNG for reservoir sampling in the similarity-band sampler

import matplotlib.pyplot as plt  # plotting library used for the histogram
import numpy as np  # log1p transform of the count columns
import pandas as pd  # used only for type hints here, data itself comes from the notebook
from rapidfuzz import fuzz, process  # fuzzy string matching, used for the near-duplicate title check
from scipy.stats import chi2_contingency, mannwhitneyu, normaltest, skew, kurtosis, spearmanr  # normality test + skew/kurtosis and Mann-Whitney U for the year-gap hypothesis, chi-square independence test for the prior-work buckets, rank correlation + p-value for the genre-age confound check
from works import differs_only_by_number_or_roman  # the production number/roman-numeral guard, reused here so the diagnostic matches the real filter

SERIES_COLOR = "#2a78d6"  # single series color, reused across charts in this module


def plot_year_distribution(works_df: pd.DataFrame, bin_width: int = 5) -> dict:
    """Plot a histogram of works_df['year'] and report the missing-year share.

    Returns a dict with the missing-year count and share, so the notebook
    can quote exact numbers in its interpretation without recomputing them.
    """
    years = works_df["year"].dropna()  # drop missing years, they can't be placed on the histogram
    missing_count = works_df["year"].isna().sum()  # count of rows with no year value
    missing_share = missing_count / len(works_df)  # missing count as a fraction of all works

    bins = range(int(years.min()), int(years.max()) + bin_width, bin_width)  # fixed-width bins spanning the full year range

    fig, ax = plt.subplots(figsize=(10, 5))  # one figure, one axis, wide aspect ratio for a time axis
    ax.hist(years, bins=bins, color=SERIES_COLOR, edgecolor="white", linewidth=0.5)  # draw the histogram with thin white gaps between bars
    ax.set_xlabel("Year")  # label the x-axis
    ax.set_ylabel("Number of works")  # label the y-axis
    ax.set_title("Distribution of works by year")  # chart title
    ax.spines["top"].set_visible(False)  # drop the top border, recessive chart frame
    ax.spines["right"].set_visible(False)  # drop the right border, recessive chart frame
    plt.show()  # render the chart in the notebook

    return {"missing_count": int(missing_count), "missing_share": float(missing_share)}  # exact numbers for the notebook's interpretation text


def missing_year_by_source(works_df: pd.DataFrame) -> pd.DataFrame:
    """Report the missing-year count and share per work_source (master vs synthetic)."""
    summary = works_df.groupby("work_source")["year"].agg(
        total="size", missing=lambda s: s.isna().sum()
    )  # group rows by work_source, count total rows and missing-year rows per group
    summary["missing_share"] = summary["missing"] / summary["total"]  # missing count as a fraction of each group's total
    return summary  # one row per work_source with total, missing, missing_share


def plot_works_per_artist(works_df: pd.DataFrame, top_n: int = 15) -> tuple[pd.Series, pd.DataFrame]:
    """Plot the distribution of works per artist and flag the top artists by count.

    Returns (summary_stats, top_artists): summary_stats is a Series of
    count/mean/median/percentiles/max, top_artists is a table of the
    top_n artists by work count, for a manual outlier check.
    """
    works_per_artist = works_df.groupby("artist_id").size()  # one row per artist_id, value is its work count

    fig, ax = plt.subplots(figsize=(10, 5))  # one figure, one axis, same shape as the year histogram
    ax.hist(works_per_artist, bins=50, color=SERIES_COLOR, edgecolor="white", linewidth=0.5)  # histogram of per-artist work counts
    ax.set_yscale("log")  # log scale on the y-axis, since a handful of bins dwarf the rest under a linear scale
    ax.set_xlabel("Number of works")  # label the x-axis
    ax.set_ylabel("Number of artists (log scale)")  # label the y-axis, noting the log scale
    ax.set_title("Distribution of works per artist")  # chart title
    ax.spines["top"].set_visible(False)  # drop the top border, recessive chart frame
    ax.spines["right"].set_visible(False)  # drop the right border, recessive chart frame
    plt.show()  # render the chart in the notebook

    summary_stats = works_per_artist.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])  # mean, std, percentiles, max in one call

    artist_names = works_df.drop_duplicates("artist_id").set_index("artist_id")["artist"]  # one name per artist_id, for labeling the top list
    top_artists = works_per_artist.sort_values(ascending=False).head(top_n).to_frame("work_count")  # top_n artist_ids ranked by work count
    top_artists["artist"] = artist_names.reindex(top_artists.index)  # attach the artist name to each top artist_id

    return summary_stats, top_artists[["artist", "work_count"]]  # stats for the writeup, top artists for the outlier check


def artist_volume_concentration(works_df: pd.DataFrame) -> dict:
    """Quantify how concentrated works_df is among a few high-volume artists.

    Reports, for works (work_count) and for potential snapshots
    (work_count - 1 per artist, one potential WHEN/WHAT transition per
    consecutive pair of works), what share of the total is held by the
    top 1% of artists, versus the bottom 50% (for works) and the median
    artist (for snapshots).
    """
    work_count = works_df.groupby("artist_id").size()  # one row per artist_id, value is its work count
    snapshot_count = (work_count - 1).clip(lower=0)  # potential snapshots per artist, floored at 0 for single-work artists

    n_artists = len(work_count)  # total number of distinct artists
    top_1pct_n = max(1, round(n_artists * 0.01))  # number of artists in the top 1% by count, at least 1

    sorted_work_count = work_count.sort_values(ascending=False)  # artists ranked highest to lowest by work count
    top_1pct_work_share = sorted_work_count.head(top_1pct_n).sum() / work_count.sum()  # share of all works held by the top 1%
    bottom_50pct_work_share = sorted_work_count.tail(n_artists // 2).sum() / work_count.sum()  # share of all works held by the bottom 50%

    sorted_snapshot_count = snapshot_count.sort_values(ascending=False)  # artists ranked highest to lowest by potential snapshots
    top_1pct_snapshot_share = sorted_snapshot_count.head(top_1pct_n).sum() / snapshot_count.sum()  # share of all potential snapshots held by the top 1%
    median_artist_snapshots = snapshot_count.median()  # potential snapshots contributed by a single median artist
    median_artist_snapshot_share = median_artist_snapshots / snapshot_count.sum()  # that single median artist's share of all potential snapshots

    return {
        "n_artists": int(n_artists),  # total distinct artists, for context
        "top_1pct_n_artists": int(top_1pct_n),  # how many artists make up that top 1%
        "top_1pct_work_share": float(top_1pct_work_share),  # share of total works from the top 1%
        "bottom_50pct_work_share": float(bottom_50pct_work_share),  # share of total works from the bottom 50%
        "top_1pct_snapshot_share": float(top_1pct_snapshot_share),  # share of total potential snapshots from the top 1%
        "median_artist_snapshots": float(median_artist_snapshots),  # potential snapshots for one median artist
        "median_artist_snapshot_share": float(median_artist_snapshot_share),  # that one median artist's share of total potential snapshots
    }


def plot_works_per_year(works_df: pd.DataFrame) -> pd.Series:
    """Plot total works per year as a time series (valid-year rows only).

    Returns the per-year counts, indexed by year, for reuse downstream.
    """
    works_per_year = works_df["year"].dropna().value_counts().sort_index()  # count per year, missing years dropped, sorted chronologically

    fig, ax = plt.subplots(figsize=(10, 5))  # one figure, one axis, same shape as the other charts
    ax.plot(works_per_year.index, works_per_year.values, color=SERIES_COLOR, linewidth=2)  # line chart, one point per year
    ax.set_xlabel("Year")  # label the x-axis
    ax.set_ylabel("Number of works")  # label the y-axis
    ax.set_title("Works per year")  # chart title
    ax.spines["top"].set_visible(False)  # drop the top border, recessive chart frame
    ax.spines["right"].set_visible(False)  # drop the right border, recessive chart frame
    plt.show()  # render the chart in the notebook

    return works_per_year  # per-year counts, reusable by later cells


def top_n_tags(works_df: pd.DataFrame, column: str, n: int = 15) -> pd.Series:
    """Compute and plot the top-N tags (genres or styles) by total work count.

    `column` is a works_df column holding a list of tags per row (e.g.
    "genres" or "styles"), each row can carry more than one tag. Returns
    the top-N counts, indexed by tag, so the same ranked list can be
    reused by later cells (e.g. trending each tag by year) without
    hardcoding it twice.
    """
    exploded_tags = works_df[column].explode()  # one row per tag occurrence, instead of one row per work
    top_counts = exploded_tags.value_counts().head(n)  # tag -> total work count, top n only

    fig, ax = plt.subplots(figsize=(10, 5))  # one figure, one axis, same shape as the other charts
    ax.bar(top_counts.index, top_counts.values, color=SERIES_COLOR)  # one bar per tag
    ax.set_xlabel(column)  # label the x-axis with the column name (genres or styles)
    ax.set_ylabel("Number of works")  # label the y-axis
    ax.set_title(f"Top {len(top_counts)} {column} by work count")  # chart title reflects how many tags were actually found, in case fewer than n exist
    ax.tick_params(axis="x", rotation=45)  # rotate tag labels so they don't overlap
    for label in ax.get_xticklabels():  # loop over the rotated tick labels
        label.set_horizontalalignment("right")  # right-align each rotated label under its bar
    ax.spines["top"].set_visible(False)  # drop the top border, recessive chart frame
    ax.spines["right"].set_visible(False)  # drop the right border, recessive chart frame
    plt.show()  # render the chart in the notebook

    return top_counts  # top-N tag -> count, reusable by later cells


def style_scale_check(works_df: pd.DataFrame, top_tags: pd.Series, column: str) -> dict:
    """Report how exhaustive a top-N tag ranking actually is.

    Unlike genres (a fixed, small taxonomy where top-15 is everything),
    styles is much higher-cardinality, so a top-N list is only a slice.
    Coverage is the share of works carrying at least one of top_tags, not
    the share of tag occurrences (one work can carry several tags, which
    would double count it).
    """
    exploded = works_df[column].explode()  # one row per tag occurrence, index repeated per source work
    in_top = exploded.isin(top_tags.index)  # True where that occurrence is one of the top-N tags
    covers_top_tag = in_top.groupby(exploded.index).any()  # one bool per original work, True if any of its tags is top-N

    return {
        "distinct_tag_count": exploded.nunique(),  # total distinct tags, not just the top-N
        "total_works": len(works_df),
        "works_covered_by_top_n": int(covers_top_tag.sum()),
        "coverage_share": covers_top_tag.mean(),  # share of works with >=1 top-N tag, not share of tag occurrences
    }


def plot_tag_trends(works_df: pd.DataFrame, top_tags: pd.Series, column: str) -> pd.DataFrame:
    """Plot each of top_tags' per-year work count as a small-multiples grid.

    `top_tags` is a Series of tag -> count (e.g. the output of top_n_tags),
    only its index (the tag names, in rank order) is used here. A small
    multiples grid is used instead of one overlaid line chart, since this
    many series would need that many distinct colors to stay readable.
    Returns a year x tag pivot table of work counts, for reuse downstream.
    """
    valid_year_works = works_df.dropna(subset=["year"])  # trending by year needs a real year, same treatment as the time series above
    exploded = valid_year_works[[column, "year"]].explode(column)  # one row per (tag, work) occurrence, year still attached
    exploded = exploded[exploded[column].isin(top_tags.index)]  # keep only the tags being trended, drop the long tail

    trend_table = exploded.groupby([column, "year"]).size().unstack(column, fill_value=0)  # year index, one column per tag, counts as values

    tags = list(top_tags.index)  # ordered list of tags to plot, same rank order as the top-N bar chart
    n_cols = 4  # fixed grid width
    n_rows = -(-len(tags) // n_cols)  # ceiling division, just enough rows to fit every tag

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharey=False)  # independent y-axis per panel, so each tag's own shape (rising/declining/stable) stays legible regardless of its overall volume
    axes = axes.flatten()  # flatten the 2D grid of axes into a flat list to iterate over

    for ax, tag in zip(axes, tags):  # one subplot per tag, in top-N order
        ax.plot(trend_table.index, trend_table[tag], color=SERIES_COLOR, linewidth=1.5)  # that tag's work count per year
        ax.set_title(tag, fontsize=10)  # subplot title is the tag name
        ax.spines["top"].set_visible(False)  # drop the top border, recessive chart frame
        ax.spines["right"].set_visible(False)  # drop the right border, recessive chart frame

    for ax in axes[len(tags):]:  # any leftover empty subplots past the last real tag
        ax.axis("off")  # hide unused grid cells instead of leaving them blank-but-framed

    fig.suptitle(f"{column.capitalize()} trend by year (top {len(tags)})")  # one title for the whole grid
    fig.supxlabel("Year")  # shared x-axis label
    fig.supylabel("Number of works")  # shared y-axis label
    fig.tight_layout()  # avoid subplot/label overlap
    plt.show()  # render the grid in the notebook

    return trend_table  # year x tag pivot table, reusable by later cells


def plot_genre_share_by_year(genre_trend_table: pd.DataFrame, min_year: int = 1960, exclude_tags: tuple = ("1",)) -> pd.DataFrame:
    """Plot each genre's share of top-genre work count per year, as a 100%-stacked area chart.

    `genre_trend_table` is the year x genre count pivot from plot_tag_trends,
    reused here instead of recomputed. `exclude_tags` drops junk tag columns
    (e.g. "1" fake-genre artifact) before computing shares, so they neither
    dilute real genres' shares nor take a stack band. Years before `min_year`
    are dropped before plotting (not from the returned table): per the
    univariate finding, total yearly volume is flat and low pre-1960, so a
    handful of works per genre there would swing share percentages wildly
    and read as a fake compositional trend. Colors are assigned explicitly
    (tab20, one per genre) instead of matplotlib's default 10-color cycle,
    which would otherwise repeat and collide once past 10 genres. Returns
    the full-range year x genre share table (each row sums to 1), for reuse
    downstream.
    """
    genre_trend_table = genre_trend_table.drop(columns=list(exclude_tags), errors="ignore")  # drop junk tag columns before normalizing shares
    share_table = genre_trend_table.div(genre_trend_table.sum(axis=1), axis=0)  # each year's row normalized to sum to 1 across genres
    plot_table = share_table.loc[min_year:]  # drop the noisy low-volume years from the plotted range only
    colors = plt.get_cmap("tab20").colors[: len(plot_table.columns)]  # one distinct color per genre, fixed order, no repeats up to 20 genres

    fig, ax = plt.subplots(figsize=(10, 5))  # one figure, one axis, same shape as the other charts
    ax.stackplot(plot_table.index, plot_table.T.values, labels=plot_table.columns, colors=colors)  # one stacked band per genre, summing to 1 at every year
    ax.set_xlabel("Year")  # label the x-axis
    ax.set_ylabel("Share of top-genre works")  # label the y-axis
    ax.set_title(f"Genre composition by year ({min_year}+)")  # chart title, states the restricted range
    ax.set_ylim(0, 1)  # fixed 0-1 range, since shares always sum to 1
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)  # legend outside the plot, one genre per top-N entry
    ax.spines["top"].set_visible(False)  # drop the top border, recessive chart frame
    ax.spines["right"].set_visible(False)  # drop the right border, recessive chart frame
    plt.show()  # render the chart in the notebook

    return share_table  # full-range year x genre share table, reusable by later cells


def missing_year_by_tag(works_df: pd.DataFrame, top_tags: pd.Series, column: str, exclude_tags: tuple = ("1",)) -> pd.DataFrame:
    """Plot missing-year share per tag, among the given top_tags.

    `exclude_tags` drops junk tag values (e.g. the "1" fake-genre artifact)
    before computing shares, so they don't distort the real tags' bars.
    Returns one row per tag with total/missing/missing_share, sorted by
    missing_share descending.
    """
    tags_to_check = [t for t in top_tags.index if t not in exclude_tags]  # real tags only, junk tags dropped
    exploded = works_df[[column, "year"]].explode(column)  # one row per (tag, work) occurrence, year still attached
    exploded = exploded[exploded[column].isin(tags_to_check)]  # keep only the tags being checked

    summary = exploded.groupby(column)["year"].agg(
        total="size", missing=lambda s: s.isna().sum()
    )  # total works and missing-year works per tag
    summary["missing_share"] = summary["missing"] / summary["total"]  # missing count as a fraction of each tag's total
    summary = summary.sort_values("missing_share", ascending=False)  # worst-missing tag first

    fig, ax = plt.subplots(figsize=(10, 5))  # one figure, one axis, same shape as the other charts
    ax.bar(summary.index, summary["missing_share"], color=SERIES_COLOR)  # one bar per tag
    ax.set_xlabel(column)  # label the x-axis
    ax.set_ylabel("Missing-year share")  # label the y-axis
    ax.set_title(f"Missing-year share by {column}")  # chart title
    ax.tick_params(axis="x", rotation=45)  # rotate tag labels so they don't overlap
    for label in ax.get_xticklabels():  # loop over the rotated tick labels
        label.set_horizontalalignment("right")  # right-align each rotated label under its bar
    ax.spines["top"].set_visible(False)  # drop the top border, recessive chart frame
    ax.spines["right"].set_visible(False)  # drop the right border, recessive chart frame
    plt.show()  # render the chart in the notebook

    return summary  # one row per tag with total, missing, missing_share


def tag_year_distribution(works_df: pd.DataFrame, top_tags: pd.Series, column: str, exclude_tags: tuple = ("1",)) -> pd.DataFrame:
    """Report min/10th-percentile/median year per tag, among non-missing-year rows only.

    Confound check for the missing-year-by-tag finding: a tag with a
    high missing-year rate could be genuinely worse-cataloged, or it could
    simply have an older catalog (more pre-digital-era works, which are
    more likely to lack a year regardless of tag). Returns one row per
    tag, sorted by median year ascending (oldest-skewing catalog first).
    """
    tags_to_check = [t for t in top_tags.index if t not in exclude_tags]  # real tags only, junk tags dropped
    exploded = works_df[[column, "year"]].explode(column)  # one row per (tag, work) occurrence, year still attached
    exploded = exploded[exploded[column].isin(tags_to_check)]  # keep only the tags being checked
    exploded = exploded.dropna(subset=["year"])  # distribution is among dated works only, missing-year rows have no year to place

    summary = exploded.groupby(column)["year"].agg(
        min="min", p10=lambda s: s.quantile(0.10), median="median"
    )  # earliest year, 10th percentile, median year per tag
    summary = summary.sort_values("median")  # oldest-skewing catalog first

    return summary  # one row per tag with min, p10, median year


def tag_confound_correlation(tag_confound_check: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation between missing-year share and each year-distribution stat.

    Takes the merged missing_share/min/p10/median table built in the
    notebook. Reports the p-value alongside each coefficient, since at a
    small number of tags a correlation number alone doesn't say whether the
    relationship could plausibly be noise.
    """
    rows = {}  # one row per year-distribution stat checked against missing_share
    for stat in ("min", "p10", "median"):
        r, p = spearmanr(tag_confound_check["missing_share"], tag_confound_check[stat])  # rank correlation + two-sided p-value
        rows[stat] = {"spearman_r": r, "p_value": p}

    return pd.DataFrame(rows).T  # one row per stat, columns spearman_r and p_value


def artist_style_transitions(works_df: pd.DataFrame) -> pd.DataFrame:
    """For each artist with 2+ known-year works, flag which consecutive transitions introduce a new style.

    Only non-missing-year works are used for ordering (a stated limitation:
    an artist's true chronology, including any missing-year works, is
    unknown). Works are sorted by year per artist, ties broken by work_id
    for a stable order. Returns one row per transition (not per artist),
    with prior_work_count (how many known-year works came before this one
    for that artist, so callers can filter by any min_previous_works
    threshold) and is_new_style (1 if this work's styles include at least
    one never seen in that artist's earlier known-year works, 0 otherwise).
    Also carries year_gap (years since the artist's previous known-year
    work), prior_style_count (distinct styles seen before this work) and
    year_missing (True if the artist has at least one undated work, which
    is invisible to the ordering above but marks an incomplete history).
    """
    undated_artists = set(works_df.loc[works_df["year"].isna(), "artist_id"])  # artists with at least one work missing a year
    dated = works_df.dropna(subset=["year"])[["artist_id", "work_id", "year", "styles", "genres"]]
    dated = dated.sort_values(["artist_id", "year", "work_id"])  # chronological per artist, work_id as a stable tiebreak for same-year works

    dated["genre"] = dated["genres"].map(lambda g: g[0] if len(g) == 1 else None)  # single genre label, None for multi-genre works so each transition falls in exactly one genre cell

    rows = []  # one row per transition, appended as artists are walked in chronological order
    for artist_id, group in dated.groupby("artist_id", sort=False):
        if len(group) < 2:
            continue  # no transition possible with fewer than 2 known-year works
        group_genres = group["genre"].tolist()  # genre label per work in chronological order, indexed alongside the style walk
        group_years = group["year"].tolist()  # year per work in chronological order, indexed alongside the style walk
        has_undated = artist_id in undated_artists  # same flag for every transition of this artist
        seen_styles: set = set(group.iloc[0]["styles"])  # running set of every style seen so far for this artist
        for i in range(1, len(group)):
            current_styles = set(group.iloc[i]["styles"])
            rows.append({
                "artist_id": artist_id,
                "prior_work_count": i,  # known-year works strictly before this one, for this artist
                "is_new_style": int(len(current_styles - seen_styles) > 0),
                "genre": group_genres[i],  # genre of the work being evaluated, None if it carries several
                "year_gap": group_years[i] - group_years[i - 1],  # years since the previous known-year work, 0 for same-year works
                "prior_style_count": len(seen_styles),  # distinct styles in this artist's history before this work
                "year_missing": has_undated,  # artist has undated works, so this history is incomplete
            })
            seen_styles |= current_styles  # fold this work's styles into the running history before moving to the next transition

    return pd.DataFrame(rows)  # one row per transition, columns artist_id, prior_work_count, is_new_style, genre, year_gap, prior_style_count, year_missing


def artist_style_diversity(works_df: pd.DataFrame) -> pd.Series:
    """Count distinct styles per artist, among artists with 2+ known-year works.

    Same eligible population as artist_style_transitions, so the two can be
    read side by side: this shows how much style variety artists typically
    accumulate, while artist_style_transitions shows how often that variety
    actually grows from one work to the next.
    """
    dated = works_df.dropna(subset=["year"])[["artist_id", "styles"]]
    eligible = dated.groupby("artist_id").size()
    eligible = eligible[eligible >= 2].index  # artists with 2+ known-year works, matching the transitions population

    exploded = dated[dated["artist_id"].isin(eligible)].explode("styles")  # one row per (artist, style) occurrence
    return exploded.groupby("artist_id")["styles"].nunique()  # distinct style count per eligible artist


PRIOR_WORK_BUCKET_EDGES = [0, 5, 10, 20, 50, float("inf")]  # right-inclusive bucket edges for prior_work_count
PRIOR_WORK_BUCKET_LABELS = ["3-5", "6-10", "11-20", "21-50", "51+"]  # bucket names, the lowest assumes min_previous_works=3
VOLUME_BUCKET_EDGES = [0, 1, 2, 4, float("inf")]  # right-inclusive bucket edges for the number of Works an artist has in one year
VOLUME_BUCKET_LABELS = ["1", "2", "3-4", "5+"]  # bucket names for that count


def what_base_rate(transitions: pd.DataFrame, min_previous_works: int) -> dict:
    """Share of transitions that introduce a new style, overall and for the modeling population."""
    restricted_mask = transitions["prior_work_count"] >= min_previous_works  # transitions with enough prior history to be modeled
    return {
        "n_transitions": len(transitions),  # every known-year transition
        "overall_base_rate": transitions["is_new_style"].mean(),  # new-style rate across all of them
        "n_restricted_transitions": int(restricted_mask.sum()),  # transitions that meet the history threshold
        "restricted_base_rate": transitions.loc[restricted_mask, "is_new_style"].mean(),  # new-style rate in the modeling population
    }


def what_cutoff_sensitivity(transitions: pd.DataFrame, cutoffs: list) -> pd.DataFrame:
    """Transition count and new-style rate at each min_previous_works cutoff."""
    rows = []  # one row per cutoff tested
    for cutoff in cutoffs:
        mask = transitions["prior_work_count"] >= cutoff  # transitions that meet this cutoff
        rows.append({
            "min_previous_works": cutoff,
            "n_transitions": int(mask.sum()),
            "new_style_rate": transitions.loc[mask, "is_new_style"].mean(),
        })
    return pd.DataFrame(rows)  # columns min_previous_works, n_transitions, new_style_rate


def _bucket_by_prior_works(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """Restrict to the modeling population and add an ordered prior_work_bucket column."""
    restricted = transitions[transitions["prior_work_count"] >= min_previous_works].copy()  # copy so the caller's frame is untouched
    restricted["prior_work_bucket"] = pd.cut(
        restricted["prior_work_count"], bins=PRIOR_WORK_BUCKET_EDGES, labels=PRIOR_WORK_BUCKET_LABELS
    )  # ordered categorical, so groupby and crosstab keep buckets in ascending order
    return restricted


def new_style_rate_by_prior_work_bucket(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """New-style introduction rate and sample size per prior-work-count bucket."""
    restricted = _bucket_by_prior_works(transitions, min_previous_works)  # modeling population with bucket labels
    grouped = restricted.groupby("prior_work_bucket", observed=True)["is_new_style"]  # one group per bucket
    return grouped.agg(n_transitions="count", new_style_count="sum", new_style_rate="mean")  # one row per bucket


def prior_work_bucket_chi_square(transitions: pd.DataFrame, min_previous_works: int) -> dict:
    """Chi-square test of independence between is_new_style and the prior-work bucket."""
    restricted = _bucket_by_prior_works(transitions, min_previous_works)  # modeling population with bucket labels
    table = pd.crosstab(restricted["prior_work_bucket"], restricted["is_new_style"])  # bucket x outcome counts
    chi2, p_value, dof, _ = chi2_contingency(table)  # expected counts are not needed by the caller
    return {"chi2": chi2, "p_value": p_value, "dof": dof, "contingency_table": table}


def _single_genre_bucketed(transitions: pd.DataFrame, min_previous_works: int, exclude_genres: tuple = ("1",)) -> pd.DataFrame:
    """Modeling population with bucket labels, keeping only single-genre transitions and real genres."""
    restricted = _bucket_by_prior_works(transitions, min_previous_works)  # modeling population with bucket labels
    restricted = restricted.dropna(subset=["genre"])  # multi-genre works have no single genre cell, dropped so cells stay mutually exclusive
    return restricted[~restricted["genre"].isin(exclude_genres)]  # drop the junk "1" genre tag flagged earlier


def new_style_rate_by_genre_and_bucket(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """New-style introduction rate per genre (rows) and prior-work bucket (columns)."""
    restricted = _single_genre_bucketed(transitions, min_previous_works)  # single-genre modeling population with bucket labels
    rates = restricted.pivot_table(index="genre", columns="prior_work_bucket", values="is_new_style", aggfunc="mean", observed=True)  # mean of the 0/1 flag is the rate
    return rates.loc[rates.mean(axis=1).sort_values(ascending=False).index]  # genres ordered by overall average rate, highest first


def genre_bucket_sample_sizes(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """Transition count per genre (rows) and prior-work bucket (columns), to read the rates against."""
    restricted = _single_genre_bucketed(transitions, min_previous_works)  # single-genre modeling population with bucket labels
    return pd.crosstab(restricted["genre"], restricted["prior_work_bucket"])  # raw counts per cell


def genre_chi_square_by_bucket(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """Chi-square test of independence between genre and is_new_style, run separately inside each prior-work bucket."""
    restricted = _single_genre_bucketed(transitions, min_previous_works)  # single-genre modeling population with bucket labels
    rows = {}  # one result row per bucket
    for bucket, group in restricted.groupby("prior_work_bucket", observed=True):
        table = pd.crosstab(group["genre"], group["is_new_style"])  # genre x outcome counts inside this bucket
        chi2, p_value, dof, expected = chi2_contingency(table)  # expected counts are kept to check the test's size assumption
        n = table.to_numpy().sum()  # transitions in this bucket
        rows[bucket] = {
            "n_transitions": n,
            "chi2": chi2,
            "dof": dof,
            "p_value": p_value,
            "cramers_v": (chi2 / (n * (min(table.shape) - 1))) ** 0.5,  # effect size, comparable across buckets unlike chi2
            "min_expected_count": expected.min(),  # smallest expected cell, should be well above 5
        }
    return pd.DataFrame(rows).T  # one row per bucket, in ascending bucket order


def genre_rank_consistency(rate_table: pd.DataFrame) -> dict:
    """Genre ranking (1 = highest rate) per bucket, plus Spearman correlation of rankings between buckets."""
    ranks = rate_table.rank(ascending=False).astype(int)  # rank genres within each bucket, 1 is the highest new-style rate
    ranks["rank_range"] = ranks.max(axis=1) - ranks.min(axis=1)  # how far each genre's position moves across buckets
    return {
        "ranks": ranks,
        "bucket_spearman": rate_table.corr(method="spearman"),  # rank agreement between every pair of buckets
    }


def _chi_square_with_effect(table: pd.DataFrame) -> dict:
    """Chi-square test of independence plus Cramer's V and the smallest expected cell, for any contingency table."""
    chi2, p_value, dof, expected = chi2_contingency(table)  # expected counts are kept to check the test's size assumption
    n = table.to_numpy().sum()  # transitions behind the table
    return {
        "chi2": chi2,
        "dof": dof,
        "p_value": p_value,
        "cramers_v": (chi2 / (n * (min(table.shape) - 1))) ** 0.5,  # effect size, comparable across tables unlike chi2
        "min_expected_count": expected.min(),  # smallest expected cell, should be well above 5
    }


YEAR_GAP_BUCKET_EDGES = [-1, 0, 1, 3, 5, 10, float("inf")]  # right-inclusive bucket edges for year_gap
YEAR_GAP_BUCKET_LABELS = ["0", "1", "2-3", "4-5", "6-10", "11+"]  # same-year, then growing gaps


def _modeling_population(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """Transitions with enough prior history to be modeled, same filter as the earlier hypotheses."""
    return transitions[transitions["prior_work_count"] >= min_previous_works]


def plot_year_gap_distribution(transitions: pd.DataFrame, min_previous_works: int, sample_size: int = 5000, seed: int = 0) -> dict:
    """Histogram of year_gap plus skewness, kurtosis and a normality test, to choose between parametric and rank tests."""
    gaps = _modeling_population(transitions, min_previous_works)["year_gap"]  # modeling population only
    fig, ax = plt.subplots(figsize=(8, 4))  # one wide panel
    ax.hist(gaps, bins=range(0, int(gaps.quantile(0.99)) + 2), color=SERIES_COLOR)  # clip the axis at the 99th percentile so the long tail does not flatten the bulk
    ax.set_yscale("log")  # log counts, since the bulk is far taller than the tail
    ax.set_xlabel("Years since previous work")  # x axis label
    ax.set_ylabel("Transitions (log scale)")  # y axis label
    ax.set_title("Year gap between consecutive works")  # chart title
    plt.show()  # draw the chart in the notebook

    sample = gaps.sample(min(sample_size, len(gaps)), random_state=seed)  # fixed-size sample, a normality test on millions of rows rejects for any tiny deviation
    stat, p_value = normaltest(sample)  # D'Agostino-Pearson test, null is a normal distribution
    return {
        "n": len(gaps),
        "mean": gaps.mean(),
        "median": gaps.median(),
        "skewness": skew(gaps),  # 0 for a symmetric distribution
        "excess_kurtosis": kurtosis(gaps),  # 0 for a normal distribution
        "normaltest_stat": stat,
        "normaltest_p": p_value,
        "sample_size": len(sample),
    }


def new_style_rate_by_year_gap_bucket(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """New-style introduction rate and sample size per year-gap bucket, descriptive companion to the rank test."""
    restricted = _modeling_population(transitions, min_previous_works).copy()  # copy so the caller's frame is untouched
    restricted["year_gap_bucket"] = pd.cut(restricted["year_gap"], bins=YEAR_GAP_BUCKET_EDGES, labels=YEAR_GAP_BUCKET_LABELS)  # ordered buckets
    grouped = restricted.groupby("year_gap_bucket", observed=True)["is_new_style"]  # one group per bucket
    return grouped.agg(n_transitions="count", new_style_count="sum", new_style_rate="mean")  # one row per bucket


def _mann_whitney_gap(group: pd.DataFrame) -> dict:
    """Mann-Whitney U on year_gap between new-style and no-new-style transitions, with rank-biserial effect size."""
    new = group.loc[group["is_new_style"] == 1, "year_gap"]  # gaps before a transition that introduced a new style
    old = group.loc[group["is_new_style"] == 0, "year_gap"]  # gaps before a transition that did not
    u_stat, p_value = mannwhitneyu(new, old, alternative="two-sided")  # rank test, no normality assumption
    return {
        "n_new_style": len(new),
        "n_no_new_style": len(old),
        "median_gap_new_style": new.median(),
        "median_gap_no_new_style": old.median(),
        "u_stat": u_stat,
        "p_value": p_value,
        "rank_biserial": 2 * u_stat / (len(new) * len(old)) - 1,  # effect size in [-1, 1], positive means new-style transitions follow longer gaps
    }


def year_gap_mann_whitney(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """Mann-Whitney U test of year_gap by is_new_style, pooled and inside each prior-work bucket."""
    restricted = _bucket_by_prior_works(transitions, min_previous_works)  # modeling population with bucket labels
    rows = {"all": _mann_whitney_gap(restricted)}  # pooled test first
    for bucket, group in restricted.groupby("prior_work_bucket", observed=True):
        rows[bucket] = _mann_whitney_gap(group)  # same test with prior-work count held roughly fixed
    return pd.DataFrame(rows).T  # one row per population, pooled first


PRIOR_STYLE_BUCKET_EDGES = [-1, 0, 1, 2, 4, 8, float("inf")]  # right-inclusive bucket edges for prior_style_count
PRIOR_STYLE_BUCKET_LABELS = ["0", "1", "2", "3-4", "5-8", "9+"]  # no style tags yet, then growing vocabularies


def _bucket_by_prior_styles(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """Restrict to the modeling population and add prior_work_bucket and prior_style_bucket columns."""
    restricted = _bucket_by_prior_works(transitions, min_previous_works)  # modeling population with prior-work buckets
    restricted["prior_style_bucket"] = pd.cut(
        restricted["prior_style_count"], bins=PRIOR_STYLE_BUCKET_EDGES, labels=PRIOR_STYLE_BUCKET_LABELS
    )  # ordered categorical, so groupby and crosstab keep buckets in ascending order
    return restricted


def new_style_rate_by_prior_style_bucket(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """New-style introduction rate and sample size per prior-style-count bucket."""
    restricted = _bucket_by_prior_styles(transitions, min_previous_works)  # modeling population with both bucket labels
    grouped = restricted.groupby("prior_style_bucket", observed=True)["is_new_style"]  # one group per bucket
    return grouped.agg(n_transitions="count", new_style_count="sum", new_style_rate="mean")  # one row per bucket


def prior_style_bucket_chi_square(transitions: pd.DataFrame, min_previous_works: int) -> dict:
    """Chi-square test of independence between is_new_style and the prior-style bucket."""
    restricted = _bucket_by_prior_styles(transitions, min_previous_works)  # modeling population with both bucket labels
    table = pd.crosstab(restricted["prior_style_bucket"], restricted["is_new_style"])  # bucket x outcome counts
    return {**_chi_square_with_effect(table), "contingency_table": table}


def prior_style_chi_square_by_work_bucket(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """Chi-square of is_new_style vs prior-style bucket inside each prior-work bucket, since style count grows with work count."""
    restricted = _bucket_by_prior_styles(transitions, min_previous_works)  # modeling population with both bucket labels
    rows = {}  # one result row per prior-work bucket
    for bucket, group in restricted.groupby("prior_work_bucket", observed=True):
        table = pd.crosstab(group["prior_style_bucket"], group["is_new_style"])  # style bucket x outcome inside this work bucket
        rows[bucket] = {"n_transitions": table.to_numpy().sum(), **_chi_square_with_effect(table)}
    return pd.DataFrame(rows).T  # one row per prior-work bucket


def new_style_rate_by_prior_style_and_work_bucket(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """New-style rate per prior-style bucket (rows) and prior-work bucket (columns), to read the controlled trend directly."""
    restricted = _bucket_by_prior_styles(transitions, min_previous_works)  # modeling population with both bucket labels
    return restricted.pivot_table(index="prior_style_bucket", columns="prior_work_bucket", values="is_new_style", aggfunc="mean", observed=True)  # mean of the 0/1 flag is the rate


def new_style_rate_by_year_missing(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """New-style rate and sample size for artists with and without undated works."""
    restricted = _modeling_population(transitions, min_previous_works)  # modeling population only
    grouped = restricted.groupby("year_missing")["is_new_style"]  # one group per flag value
    return grouped.agg(n_transitions="count", new_style_count="sum", new_style_rate="mean")  # False row then True row


def year_missing_chi_square(transitions: pd.DataFrame, min_previous_works: int) -> dict:
    """Chi-square test of independence between is_new_style and year_missing (2 x 2)."""
    restricted = _modeling_population(transitions, min_previous_works)  # modeling population only
    table = pd.crosstab(restricted["year_missing"], restricted["is_new_style"])  # flag x outcome counts
    return {**_chi_square_with_effect(table), "contingency_table": table}


def new_style_rate_by_year_missing_and_work_bucket(transitions: pd.DataFrame, min_previous_works: int) -> pd.DataFrame:
    """New-style rate by year_missing (rows) inside each prior-work bucket (columns), with the gap per bucket."""
    restricted = _bucket_by_prior_works(transitions, min_previous_works)  # modeling population with bucket labels
    rates = restricted.pivot_table(index="year_missing", columns="prior_work_bucket", values="is_new_style", aggfunc="mean", observed=True)  # mean of the 0/1 flag is the rate
    rates.loc["gap (True - False)"] = rates.loc[True] - rates.loc[False]  # per-bucket difference, shows whether the pooled gap survives holding history length fixed
    return rates


TRANSFORM_COLUMNS = ["prior_work_count", "year_gap", "prior_style_count"]  # right-skewed count columns of the transitions table
CORRELATION_COLUMNS = TRANSFORM_COLUMNS + ["is_new_style"]  # numeric columns analysed above, plus the 0/1 target


def transformation_check(transitions: pd.DataFrame, min_previous_works: int, columns: list = TRANSFORM_COLUMNS, sample_size: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Skewness and normality test of each count column, raw and after log1p, on the modeling population.

    Skewness and excess kurtosis use every row. The D'Agostino-Pearson test
    uses one fixed-seed random sample of at most `sample_size` rows, the same
    rows for the raw and the log1p version, because a normality test on
    millions of rows rejects for any tiny deviation. Returns one row per
    column and version, with n, the sample size, skewness, excess kurtosis,
    the test statistic and its p-value.
    """
    population = _modeling_population(transitions, min_previous_works)  # same filter as the hypotheses above
    rows = []  # one row per column and version
    for column in columns:  # each count column in turn
        values = population[column].astype(float)  # counts as floats
        sample = values.sample(min(sample_size, len(values)), random_state=seed)  # one fixed-seed sample per column
        for version, data, drawn in (("raw", values, sample), ("log1p", np.log1p(values), np.log1p(sample))):  # log1p keeps the zeros defined
            stat, p_value = normaltest(drawn)  # null is a normal distribution
            rows.append({"variable": column, "version": version, "n": len(data), "sample_size": len(drawn), "skewness": skew(data), "excess_kurtosis": kurtosis(data), "normaltest_stat": stat, "normaltest_p": p_value})  # skew and kurtosis on all rows, test on the sample
    return pd.DataFrame(rows)  # long table, raw row first for each column


def spearman_table(transitions: pd.DataFrame, min_previous_works: int, columns: list = CORRELATION_COLUMNS) -> pd.DataFrame:
    """Spearman correlation, two-sided p-value and n for every pair of numeric columns of the modeling population.

    Rank correlation is used because the count columns are heavy-tailed and a
    log transform leaves it unchanged (it only depends on ranks). Every pair
    is a separate test, so the table also carries the Bonferroni threshold
    (0.05 divided by the number of pairs) and whether each p-value is below it.
    """
    population = _modeling_population(transitions, min_previous_works)[columns].astype(float)  # numeric columns only, same filter as above
    pairs = list(itertools.combinations(columns, 2))  # every pair once
    threshold = 0.05 / len(pairs)  # Bonferroni threshold for this family of tests
    rows = []  # one row per pair
    for first, second in pairs:  # each pair in turn
        r, p_value = spearmanr(population[first], population[second])  # rank correlation and two-sided p-value
        rows.append({"variable_a": first, "variable_b": second, "n": len(population), "spearman_r": r, "p_value": p_value, "bonferroni_alpha": threshold, "below_bonferroni": bool(p_value < threshold)})  # one test per pair
    return pd.DataFrame(rows)  # one row per pair


def inspect_near_duplicate_titles(works_df: pd.DataFrame, artist: str, sample_size: int = 40) -> pd.DataFrame:
    """Rank one artist's titles by how close they are to another title of theirs.

    Uses fuzzy string similarity (rapidfuzz token_sort_ratio), not exact
    match, so near-duplicates that differ by punctuation, bracket suffixes
    ("(Deluxe)", "(Remastered)"), or minor spelling/format differences
    still surface. Returns the sample_size titles with the highest
    similarity to any other title by the same artist, each paired with
    that closest match, for manual inspection.
    """
    artist_works = works_df[works_df["artist"] == artist][["work_id", "title", "year", "work_source"]].reset_index(drop=True)  # this artist's works only, fresh row index

    titles = artist_works["title"].tolist()  # plain list of titles, in the same order as artist_works' rows
    similarity_matrix = process.cdist(titles, titles, scorer=fuzz.token_sort_ratio)  # every title compared against every other title, 0-100 scores

    for i in range(len(titles)):  # loop over each title's own row in the matrix
        similarity_matrix[i, i] = 0  # zero out the self-match, a title is always 100% similar to itself

    artist_works["best_match_score"] = similarity_matrix.max(axis=1)  # each title's highest similarity to any other title by this artist
    artist_works["best_match_title"] = [titles[i] for i in similarity_matrix.argmax(axis=1)]  # which other title produced that highest score

    near_duplicates = artist_works.sort_values("best_match_score", ascending=False)  # titles most likely to be near-duplicates first
    return near_duplicates.head(sample_size)  # top sample_size rows for manual inspection


def sample_pairs_by_similarity_band(
    works_df: pd.DataFrame,
    bands: tuple = ((85, 90), (90, 95), (95, 99)),
    pairs_per_band: int = 20,
    random_state: int = 1,
    window_size: int = 10,
    guard_number_or_roman: bool = False,
) -> dict:
    """Sample candidate duplicate title pairs within the same artist_id, by similarity band.

    Diagnostic only, used to pick a fuzzy-dedup score threshold before
    writing an actual filter. Exact matches (score 100) are excluded,
    those are already handled by the existing exact-match dedup. When
    `guard_number_or_roman` is True, a pair is also skipped if its titles'
    only differing word tokens are numbers or roman numerals (the
    numbered-series false positives, e.g. "Vol. 2" vs "Vol. 3").

    Memory- and time-safe by construction without capping any artist:
    titles are compared within a single artist only (never across
    works_df as a whole), and within that artist each title is only
    compared against its next `window_size` neighbors in alphabetical
    order, not every other title (a true duplicate's titles sort right
    next to each other, so this blocking window still finds them while
    turning the per-artist cost from O(n^2) into O(n * window_size)).
    Matching pairs are kept via reservoir sampling as the loop goes, not
    collected into one unbounded list first, so memory stays bounded to
    roughly `pairs_per_band * len(bands)` records regardless of dataset size.

    Returns a dict of band label -> DataFrame of sampled pairs, each row
    carrying both titles, years, work_sources and work_ids for manual
    inspection.
    """
    rng = random.Random(random_state)  # local RNG, so this function doesn't touch the global random state

    work_counts = works_df["artist_id"].value_counts()  # works per artist_id, needed to find artists that can form a pair
    multi_work_artist_ids = work_counts[work_counts >= 2].index  # every artist with at least 2 works, no upper cap now
    print(f"Comparing titles within {len(multi_work_artist_ids)} artists ({int(work_counts.loc[multi_work_artist_ids].sum())} works), window size {window_size}")  # visibility into the full scope of the scan

    candidates = works_df[works_df["artist_id"].isin(multi_work_artist_ids)]  # restrict to artists with 2+ works, before grouping

    band_labels = [f"{low}-{high}" for low, high in bands]  # one label per band, reused below
    lowest_band_floor = min(low for low, _ in bands)  # the lowest score any requested band cares about
    reservoirs = {label: [] for label in band_labels}  # band label -> running reservoir sample, capped at pairs_per_band
    seen_counts = {label: 0 for label in band_labels}  # band label -> total qualifying pairs seen so far (reservoir sampling needs this)

    for artist_id, group in candidates.groupby("artist_id"):  # one group per artist, comparisons never run across more than one artist's titles
        titles = group["title"].tolist()  # that artist's titles, in their original order
        rows = group[["work_id", "year", "work_source"]].to_dict("records")  # same original order, one dict per title

        sort_order = sorted(range(len(titles)), key=lambda idx: titles[idx].lower())  # row positions sorted alphabetically, so near-duplicate titles land next to each other
        n = len(sort_order)  # this artist's group size, used to bound the comparison window

        for pos in range(n):  # walk the alphabetically sorted titles once
            i = sort_order[pos]  # original row index of the title at this sorted position
            for pos2 in range(pos + 1, min(pos + 1 + window_size, n)):  # only the next window_size titles in sorted order, not every other title in the group
                j = sort_order[pos2]  # original row index of the nearby title being compared

                score = fuzz.token_sort_ratio(titles[i], titles[j])  # similarity between this one pair only
                if score >= 100 or score < lowest_band_floor:  # drop exact matches (handled elsewhere) and anything below every requested band
                    continue  # nothing to do with this pair

                label = next((lbl for lbl, (low, high) in zip(band_labels, bands) if low <= score < high), None)  # which band this pair falls into
                if label is None:  # falls in a gap between requested bands (e.g. between bands if they don't tile exactly)
                    continue  # skip it, nothing to sample into

                if guard_number_or_roman and differs_only_by_number_or_roman(titles[i], titles[j]):  # numbered-series false positive, only enabled when the guard is requested
                    continue  # skip it, same reasoning as the band-gap case above

                record = {
                    "artist_id": artist_id,  # which artist this pair belongs to
                    "score": float(score),  # similarity score for this pair
                    "title_a": titles[i], "title_b": titles[j],  # the two titles being compared
                    "year_a": rows[i]["year"], "year_b": rows[j]["year"],  # their years, for the same-title/different-year ambiguity check
                    "work_source_a": rows[i]["work_source"], "work_source_b": rows[j]["work_source"],  # master vs synthetic for each
                    "work_id_a": rows[i]["work_id"], "work_id_b": rows[j]["work_id"],  # ids, to look the works up again if needed
                }

                seen_counts[label] += 1  # one more qualifying pair seen for this band, including this one
                reservoir = reservoirs[label]  # this band's running sample
                if len(reservoir) < pairs_per_band:  # reservoir not full yet, just append
                    reservoir.append(record)
                else:  # reservoir full, replace an existing entry with the classic decreasing-probability rule
                    replace_at = rng.randint(0, seen_counts[label] - 1)  # uniform over every pair seen so far for this band
                    if replace_at < pairs_per_band:  # only replace if the random draw lands inside the reservoir
                        reservoir[replace_at] = record

    return {label: pd.DataFrame(records) for label, records in reservoirs.items()}  # one DataFrame per band, ready to print and eyeball


LAST_RELIABLE_LABEL_YEAR = 2022 # last snapshot year whose next-year label is reliable (Y+1 is at most 2023, later years are incompletely loaded)
GAP_BUCKET_EDGES = [0, 1, 2, 4, 9, float("inf")] # right-inclusive bucket edges for years_since_last
GAP_BUCKET_LABELS = ["1", "2", "3-4", "5-9", "10+"] # bucket names for years since the last Work
HISTORY_BUCKET_EDGES = [0, 2, 5, 10, 20, float("inf")] # right-inclusive bucket edges for history_count
HISTORY_BUCKET_LABELS = ["1-2", "3-5", "6-10", "11-20", "21+"] # bucket names for catalogue length


def load_when_gap_rows(snapshots_path: str, last_year: int = LAST_RELIABLE_LABEL_YEAR) -> pd.DataFrame:
    """One row per artist-year for the WHEN timing test, with gap and history buckets."""
    columns = ["artist_id", "year", "works_so_far_this_year", "history_count", "years_since_last", "next_1y"] # only the columns the test needs
    snapshots = pd.read_parquet(snapshots_path, columns=columns) # cached snapshots table, not rebuilt here
    rows = snapshots[snapshots["works_so_far_this_year"] == 1] # first snapshot of each artist-year
    rows = rows[(rows["history_count"] >= 1) & (rows["years_since_last"] >= 1)] # artists with a past Work and a real gap
    rows = rows[rows["next_1y"].notna() & (rows["year"] <= last_year)].copy() # resolved label and reliable years only
    rows["gap_bucket"] = pd.cut(rows["years_since_last"], bins=GAP_BUCKET_EDGES, labels=GAP_BUCKET_LABELS) # ordered gap buckets
    rows["history_bucket"] = pd.cut(rows["history_count"], bins=HISTORY_BUCKET_EDGES, labels=HISTORY_BUCKET_LABELS) # ordered catalogue-length buckets
    return rows # one row per artist-year, ready for the tests


def next_year_rate_by_gap_bucket(rows: pd.DataFrame) -> pd.DataFrame:
    """Count and next-year release rate per years-since-last bucket."""
    grouped = rows.groupby("gap_bucket", observed=True)["next_1y"] # next_1y grouped by gap bucket
    return grouped.agg(n_rows="count", released_next_year="sum", rate="mean") # one row per bucket


def gap_bucket_chi_square(rows: pd.DataFrame) -> dict:
    """Chi-square test of independence between next_1y and the gap bucket."""
    table = pd.crosstab(rows["gap_bucket"], rows["next_1y"]) # bucket x outcome counts
    chi2, p_value, dof, expected = chi2_contingency(table) # chi-square independence test
    return {
        "chi2": chi2, # test statistic
        "p_value": p_value, # p-value, overstated because rows of one artist are not independent
        "dof": dof, # degrees of freedom
        "cramers_v": (chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))) ** 0.5, # effect size, comparable across tables
        "min_expected_count": float(expected.min()), # size assumption check
        "n_rows": int(table.to_numpy().sum()), # rows behind the test
    }


def gap_chi_square_by_history_bucket(rows: pd.DataFrame) -> pd.DataFrame:
    """Repeat the gap test inside each history_count bucket and list the rate per gap bucket."""
    records = [] # one record per history bucket
    for label in HISTORY_BUCKET_LABELS: # keeps the bucket order
        subset = rows[rows["history_bucket"] == label] # rows of this catalogue length only
        result = gap_bucket_chi_square(subset) # same test with history held fixed
        rates = subset.groupby("gap_bucket", observed=False)["next_1y"].mean() # rate per gap bucket, NaN if a cell is empty
        record = {"history_bucket": label, "n_rows": result["n_rows"], "chi2": result["chi2"], "p_value": result["p_value"], "cramers_v": result["cramers_v"]} # test summary
        record.update({f"rate_gap_{gap}": rates[gap] for gap in GAP_BUCKET_LABELS}) # rate per gap bucket as columns
        records.append(record) # collect the row
    return pd.DataFrame(records).set_index("history_bucket") # one row per history bucket


def plot_next_year_rate_by_gap(rate_table: pd.DataFrame) -> None:
    """Bar chart of the pooled next-year release rate per gap bucket."""
    fig, ax = plt.subplots(figsize=(8, 5)) # single chart
    bars = ax.bar(rate_table.index.astype(str), rate_table["rate"] * 100, color=SERIES_COLOR) # rate in percent per bucket
    ax.bar_label(bars, fmt="%.1f%%") # value label on each bar
    ax.set_xlabel("Years since the last release") # x-axis label
    ax.set_ylabel("Artist-years with a release next year (%)") # y-axis label
    ax.set_title("Chance of releasing next year by years since the last release") # chart title
    ax.spines["top"].set_visible(False) # cleaner look
    ax.spines["right"].set_visible(False) # cleaner look
    plt.show() # render the figure
