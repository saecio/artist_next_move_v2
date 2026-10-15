"""Checks on the style by year series: tables per Work source and the three-year growth.

Read only helpers, nothing is fitted or saved. They reuse `style_series.py`
for the series.

Usage:
    from style_series import load_works, keep_dated_styled
    from style_checks import universe_tables, smoothed_growth
"""

import pandas as pd  # tables

from style_series import style_year_counts, style_year_table  # series builders from the first phase


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
