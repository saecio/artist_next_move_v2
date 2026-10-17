"""Style series duplicate audit, counts only: style by year counts on the original works and on the works collapsed to the earliest of each artist and title."""

import pandas as pd  # tables
from scipy.stats import spearmanr  # rank correlation between the two growth rankings

from config import CONFIG  # project settings read from config.yaml
from duplicate_sensitivity import collapse_works  # definition A collapse, reused without changes
from style_checks import smoothed_growth, three_year_base, top_k_overlap  # growth and top-k overlap helpers
from style_forecast import DEV_FIRST_TARGET, DEV_LAST_TARGET, MIN_BASE, TOP_K, style_table  # development years, eligibility base, ranking size, series builder
from style_series import WORKS_PATH, keep_dated_styled, style_year_counts  # series builders from the first phase

WORKS_COLUMNS = ["work_id", "artist_id", "title", "year", "styles"]  # the only columns the audit needs
HOLDOUT_START = CONFIG["years"]["holdout_start"]  # years.holdout_start; first year that is only counted, never modelled
STYLE_TEST_LAST = 2022  # last style test target year (docs/decisions.md, entry "Style trend forecast, design fixed before any model result exists")
SHARE_LIMIT = 0.05  # removed share above which a style-year is flagged
TOP_STYLES = 20  # number of styles listed by removed share
PARTITIONS = {  # partition name -> (first year, last year), None means open
    f"<{DEV_LAST_TARGET - 1}": (None, DEV_LAST_TARGET - 2),  # years before the last two development years
    f"{DEV_LAST_TARGET - 1}-{DEV_LAST_TARGET}": (DEV_LAST_TARGET - 1, DEV_LAST_TARGET),  # last two development years
    f"{HOLDOUT_START}-{STYLE_TEST_LAST}": (HOLDOUT_START, STYLE_TEST_LAST),  # style test years
    f">{STYLE_TEST_LAST}": (STYLE_TEST_LAST + 1, None),  # years after the style test
}


def load_audit_works() -> pd.DataFrame:
    """Read only the columns of the cleaned Works table that the audit uses."""
    return pd.read_parquet(WORKS_PATH, columns=WORKS_COLUMNS)  # one row per Work


def kept_versions(works: pd.DataFrame) -> tuple:
    """Kept (dated and styled) Works of the original table and of the collapsed table, plus the collapse counts."""
    collapsed, info = collapse_works(works)  # definition A clusters, earliest work kept
    kept_original, _ = keep_dated_styled(works)  # series input of the original works
    kept_collapsed, _ = keep_dated_styled(collapsed)  # series input of the collapsed works
    return kept_original, kept_collapsed, info  # both versions and the collapse counts


def count_comparison(kept_original: pd.DataFrame, kept_collapsed: pd.DataFrame) -> pd.DataFrame:
    """Per style and year: original count, collapsed count and difference (original minus collapsed)."""
    original = style_year_counts(kept_original).rename(columns={"n_works": "original"})  # long table, original
    collapsed = style_year_counts(kept_collapsed).rename(columns={"n_works": "collapsed"})  # long table, collapsed
    table = original.merge(collapsed, on=["style", "year"], how="outer").fillna(0)  # a style-year absent from one side counts 0
    table[["original", "collapsed"]] = table[["original", "collapsed"]].astype(int)  # whole counts
    table["difference"] = table["original"] - table["collapsed"]  # works removed by the collapse
    return table.sort_values(["style", "year"], ignore_index=True)  # stable order


def partition_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Total style counts and removed share per year partition (a Work in several styles counts once per style)."""
    rows = []  # one row per partition
    for name, (first, last) in PARTITIONS.items():  # each partition
        keep = pd.Series(True, index=counts.index)  # start with every row
        if first is not None:  # lower bound
            keep &= counts["year"] >= first  # from the first year
        if last is not None:  # upper bound
            keep &= counts["year"] <= last  # up to the last year
        part = counts[keep]  # rows of the partition
        rows.append({"section": "partition", "key": name, "original": int(part["original"].sum()), "collapsed": int(part["collapsed"].sum()), "removed_share": part["difference"].sum() / part["original"].sum()})  # totals and share
    return pd.DataFrame(rows)  # partition table


def top_removed_styles(counts: pd.DataFrame, last_year: int, n: int = TOP_STYLES) -> pd.DataFrame:
    """The n styles with the highest removed share over the years up to last_year (share of the summed counts)."""
    part = counts[counts["year"] <= last_year].groupby("style")[["original", "collapsed", "difference"]].sum()  # summed counts per style
    part = part[part["original"] > 0]  # a share needs a positive denominator
    part["removed_share"] = part["difference"] / part["original"]  # removed share per style
    top = part.sort_values("removed_share", ascending=False).head(n).reset_index()  # highest shares first
    return top.rename(columns={"style": "key"}).assign(section=f"top_removed_share_to_{last_year}")[["section", "key", "original", "collapsed", "removed_share"]]  # summary layout


def styles_over_share(counts: pd.DataFrame, last_year: int = None, limit: float = SHARE_LIMIT) -> int:
    """Number of styles with a removed share above `limit` in at least one year (up to last_year when given)."""
    part = counts if last_year is None else counts[counts["year"] <= last_year]  # restrict the years when asked
    part = part[part["original"] > 0]  # a share needs a positive denominator
    return int(part.loc[part["difference"] / part["original"] > limit, "style"].nunique())  # distinct styles above the limit


def growth_agreement(kept_original: pd.DataFrame, kept_collapsed: pd.DataFrame, first: int = DEV_FIRST_TARGET, last: int = DEV_LAST_TARGET) -> pd.DataFrame:
    """Per development year: Spearman and top-10 overlap between the original and collapsed growth rankings.

    Both tables are cut at `last`, so no later year is read. Growth is count over the mean of the three
    earlier years minus one. The styles are those eligible on the original table (3-year base at least MIN_BASE).
    """
    table_original = style_table(kept_original, last)  # original style by year table, years up to last only
    table_collapsed = style_table(kept_collapsed, last).reindex(index=table_original.index, fill_value=0)  # same styles and years
    rows = []  # one row per year
    for year in range(first, last + 1):  # development target years
        eligible = three_year_base(table_original, year) >= MIN_BASE  # eligibility from the original table
        pair = pd.DataFrame({"o": smoothed_growth(table_original, year)[eligible], "c": smoothed_growth(table_collapsed, year)[eligible]}).dropna()  # growth under both versions, defined in both
        overlap = top_k_overlap(table_original, table_collapsed, year, k=TOP_K, min_base=MIN_BASE)  # top-k overlap on the eligible set
        rows.append({"section": "growth_agreement", "key": year, "eligible": int(eligible.sum()), "spearman": spearmanr(pair["o"], pair["c"])[0], "top10_overlap": overlap["overlap"]})  # one year
    return pd.DataFrame(rows)  # per-year agreement
