"""Checks on the style by year series: Work type split and label concentration.

Read only helpers, nothing is fitted or saved. They reuse `style_series.py`
for the series and `works.py` for the synthetic id offset.

What this module reproduces in docs/decisions.md, entry "Style trend forecast,
design fixed before any model result exists" (15-10-2026):
- the label figures of the context signals: distinct labels and top-3 label
  share per style, and the share of Works whose release has no label or only
  "Not On Label" names, with and without those names (`work_release_ids`,
  `load_release_info`, `label_flags`, `label_concentration`);
- the comparison of the growth of the top styles between all Works, master
  Works only and synthetic Works only (`universe_tables`, `three_year_base`,
  `smoothed_growth`, `top_k_overlap`).
The synthetic share of a style (signal c of that entry) is not computed here.
The repo holds no saved output of these figures, so the entry is their only record.

Usage:
    from style_series import load_works, keep_dated_styled
    from style_checks import universe_tables, smoothed_growth
"""

from pathlib import Path  # path handling for the parquet files

import pandas as pd  # tables
import pyarrow.dataset as ds  # filtered read of the large releases file

from style_series import style_year_counts, style_year_table  # series builders from the first phase
from works import SYNTHETIC_WORK_ID_OFFSET  # synthetic work_id = release_id + this offset

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


def top_k_overlap(table_all: pd.DataFrame, table_other: pd.DataFrame, year: int, k: int = 10, min_base: float = 100) -> dict:
    """Top k styles by growth under two universes, on the set eligible under all Works."""
    eligible = three_year_base(table_all, year) >= min_base  # eligibility always from the all-Works table
    growth_all = smoothed_growth(table_all, year)[eligible]  # growth under all Works
    growth_other = smoothed_growth(table_other, year)[eligible]  # growth under the other universe, same styles
    top_all = growth_all.sort_values(ascending=False).head(k)  # top k under all Works
    top_other = growth_other.dropna().sort_values(ascending=False).head(k)  # top k under the other universe, undefined growth skipped
    return {"eligible": int(eligible.sum()), "undefined_other": int(growth_other.isna().sum()), "top_all": top_all, "top_other": top_other, "overlap": len(set(top_all.index) & set(top_other.index))}  # tables and overlap size


def work_release_ids(works: pd.DataFrame) -> pd.Series:
    """Release id of the release that represents each Work, indexed by work_id.

    Synthetic Works carry their own release id (work_id minus the offset).
    Master Works use the main release of the Master from masters.parquet.
    """
    masters = pd.read_parquet(DATA_DIR / "masters.parquet", columns=["master_id", "main_release_id"]).set_index("master_id")["main_release_id"]  # Master -> main release
    is_master = works["work_source"] == "master"  # Works that come from a Master
    ids = pd.Series((works["work_id"] - SYNTHETIC_WORK_ID_OFFSET).to_numpy(), index=works["work_id"].to_numpy())  # synthetic rule for every Work first
    ids[works.loc[is_master, "work_id"].to_numpy()] = works.loc[is_master, "work_id"].map(masters).to_numpy()  # overwrite the Master Works with the main release id
    return ids  # NaN where a Master has no main release


def load_release_info(release_ids) -> pd.DataFrame:
    """Read label names and release date for the given release ids only."""
    dataset = ds.dataset(DATA_DIR / "releases.parquet")  # lazy handle, nothing is loaded yet
    table = dataset.to_table(columns=["release_id", "master_id", "released", "labels"], filter=ds.field("release_id").isin(list(release_ids)))  # only the wanted rows and columns
    info = table.to_pandas()  # to pandas
    info["label_names"] = info["labels"].map(lambda entries: sorted({e["name"] for e in entries}) if entries is not None else [])  # distinct label names per release, empty when none
    return info.drop(columns="labels").drop_duplicates("release_id").set_index("release_id")  # one row per release


def label_flags(label_names: list) -> bool:
    """True when a release has no label at all or only Not On Label entries."""
    return all(name.startswith(NOT_ON_LABEL) for name in label_names)  # all() of an empty list is True, which covers the no-label case


def label_concentration(works: pd.DataFrame, k: int = 3) -> dict:
    """Label figures for one set of Works (needs a `label_names` column, one list per Work).

    Each Work counts once per label. The top-k share is the sum of the k
    largest label counts divided by the number of Works. `real_` figures
    leave the Not On Label names out of the label counts.
    """
    names = works["label_names"].explode().dropna()  # one row per Work and label, Works without labels vanish
    real = names[~names.str.startswith(NOT_ON_LABEL)]  # same without the placeholder names
    n = len(works)  # denominator
    return {
        "n_works": n,  # Works in the set
        "n_labels": names.nunique(),  # distinct label names, placeholder included
        "top3_share": names.value_counts().head(k).sum() / n,  # top k labels, placeholder included
        "n_real_labels": real.nunique(),  # distinct labels without the placeholder
        "top3_real_share": real.value_counts().head(k).sum() / n,  # top k real labels
        "missing_share": works["label_names"].map(label_flags).mean(),  # no label or only Not On Label
    }
