"""Style by year series of Works counts, the input of the style growth forecast.

A Work counts once in each of its styles, so styles overlap and the shares
across styles do not sum to 1. Nothing is fitted or saved here.

Usage:
    from style_series import load_works, keep_dated_styled, style_year_counts, style_year_table
    works = load_works()
    kept, dropped = keep_dated_styled(works)
    table = style_year_table(style_year_counts(kept))
"""

from pathlib import Path  # path handling for the Works file

import pandas as pd  # tables

from snapshots import CONFIG, ROOT, _split_styles  # same list-to-set conversion used by the other models

WORKS_PATH = ROOT / CONFIG["paths"]["works"]  # paths.works, cleaned Works table used by the WHEN and WHAT models


def load_works(path: Path = WORKS_PATH) -> pd.DataFrame:
    """Read the cleaned Works table."""
    return pd.read_parquet(path)  # one row per Work


def keep_dated_styled(works: pd.DataFrame) -> tuple:
    """Keep Works with a year and at least one style tag, and count the dropped ones by reason.

    Returns the kept Works (year as int, styles as a set) and a dict with the
    number of Works dropped for each reason. A Work missing both is counted
    under `no_year` only, so the reasons do not overlap.
    """
    has_year = works["year"].notna()  # Works with a usable year
    style_sets = works["styles"].map(_split_styles)  # one set of tags per Work, empty when none
    has_style = style_sets.map(len) > 0  # Works with at least one style tag
    dropped = {
        "no_year": int((~has_year).sum()),  # missing year, whatever the styles
        "no_style_only": int((has_year & ~has_style).sum()),  # has a year but no style tag
    }
    kept = works[has_year & has_style].copy()  # Works that enter the series
    kept["year"] = kept["year"].astype(int)  # calendar year as integer
    kept["styles"] = style_sets[has_year & has_style]  # replace the raw cell by the set of tags
    return kept, dropped  # kept Works and the drop counts


def style_year_counts(kept: pd.DataFrame) -> pd.DataFrame:
    """Count Works per style and year, one count per Work in each of its styles."""
    pairs = kept[["year", "styles"]].explode("styles")  # one row per Work and style, a set has no repeated tag
    return pairs.groupby(["styles", "year"]).size().rename("n_works").reset_index().rename(columns={"styles": "style"})  # long table style, year, n_works


def style_year_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long counts into a style by year table, years without Works filled with 0."""
    table = counts.pivot(index="style", columns="year", values="n_works").fillna(0).astype(int)  # rows styles, columns years
    return table.reindex(columns=range(table.columns.min(), table.columns.max() + 1), fill_value=0)  # every year in range present
