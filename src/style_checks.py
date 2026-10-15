"""Checks on the style by year series: Work type split and label concentration.

Read only helpers, nothing is fitted or saved. They reuse `style_series.py`
for the series and `works.py` for the synthetic id offset.

Usage:
    from style_series import load_works, keep_dated_styled
    from style_checks import universe_tables, smoothed_growth
"""

from pathlib import Path  # path handling for the parquet files

import pandas as pd  # tables

from style_series import style_year_counts, style_year_table  # series builders from the first phase

DATA_DIR = Path(__file__).resolve().parent.parent / "data/processed"  # folder with the parquet files
NOT_ON_LABEL = "Not On Label"  # Discogs placeholder name (variants start with this text)


def universe_tables(kept: pd.DataFrame, years: range) -> dict:
    """Style by year tables for all Works, master Works only and synthetic Works only.

    Every table has the same rows (styles of the full series) and the same
    year columns, so they can be compared cell by cell. Missing cells are 0.
    """
    universes = {"all": kept, "master": kept[kept["work_source"] == "master"], "synthetic": kept[kept["work_source"] == "synthetic"]}  # the three subsets
    full = style_year_table(style_year_counts(kept))  # reference table for the row index
    return {name: style_year_table(style_year_counts(sub)).reindex(index=full.index, columns=years, fill_value=0).fillna(0).astype(int) for name, sub in universes.items()}  # same shape for all three


def three_year_base(table: pd.DataFrame, year: int) -> pd.Series:
    """Mean count of the three years before `year`, per style."""
    return table[[year - 1, year - 2, year - 3]].mean(axis=1)  # simple mean of t-1, t-2, t-3


def smoothed_growth(table: pd.DataFrame, year: int) -> pd.Series:
    """Count in `year` divided by the three-year base, minus 1. Missing when the base is 0."""
    base = three_year_base(table, year)  # denominator per style
    return (table[year] / base.where(base > 0)) - 1  # a zero base gives NaN instead of infinity


def label_flags(label_names: list) -> bool:
    """True when a release has no label at all or only Not On Label entries."""
    return all(name.startswith(NOT_ON_LABEL) for name in label_names)  # all() of an empty list is True, which covers the no-label case
