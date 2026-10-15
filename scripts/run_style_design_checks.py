"""Recompute the label and master versus synthetic figures of the style design entry of 15-10-2026 in docs/decisions.md.

Calls the functions of src/style_checks.py: universe_tables, smoothed_growth and top_k_overlap for the master versus synthetic
comparison of the 2023 growth of the top styles, and work_release_ids, load_release_info and label_concentration for the label
figures of the context signals (a) and (b). Large tables are read by column (Works: four columns; masters: two columns, inside
work_release_ids; releases: four columns for the wanted release ids only, inside load_release_info). Nothing is fitted and
there is no randomness, so the project seed is not used.

Parameters taken from the entry: base year 2023, top 10 styles, eligibility base of at least 100, top-3 label share, percentile
among the eligible styles. Parameters the entry does not describe, chosen here and printed with the results: the label figures use
the Works of each style dated in the base year 2023, the percentile is the share of eligible styles with a value less than or equal
to the style's value, and a Work whose release is not found counts as a Work with no label.

The outputs are created once and never overwritten (exclusive creation, the script refuses to run if the folder has files).

Usage:
    venv/bin/python3 -m py_compile scripts/run_style_design_checks.py && venv/bin/python3 scripts/run_style_design_checks.py
"""

import resource  # peak memory of the process
import sys  # module search path and exit
import time  # runtime
from pathlib import Path  # project root

import pandas as pd  # tables

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from config import CONFIG  # paths and year bounds from config.yaml
from style_checks import label_concentration, load_release_info, smoothed_growth, three_year_base, top_k_overlap, universe_tables, work_release_ids  # functions that produced the figures
from style_series import keep_dated_styled  # keeps dated Works with at least one style

BASE_YEAR = 2023  # base year of the context signals and of the 2023 growth, as written in the entry
TOP_K = 10  # size of the top list, as written in the entry
MIN_BASE = 100  # eligibility base, as written in the entry
ENTRY_PERCENTILE_STYLES = ["Russian Pop", "Future House", "Latin", "Ethno-pop"]  # styles the entry places at the 91st to 100th percentile of top-3 label share
ENTRY_OTHER_STYLES = ["Noise Rock", "Alt-Pop", "Plunderphonics"]  # (Plunderphonics is named in the projection part of the entry) styles the entry names without a percentile figure
OUT_DIR = ROOT / CONFIG["paths"]["results"] / "style_design_checks"  # new folder of this script
WORKS_PATH = ROOT / CONFIG["paths"]["works"]  # cleaned Works table
FIRST_YEAR = CONFIG["years"]["style_first_column"]  # first calendar column of the style tables
LAST_YEAR = CONFIG["years"]["partial"]  # last calendar column of the style tables

if OUT_DIR.exists() and any(OUT_DIR.iterdir()):  # never overwrite
    sys.exit(f"{OUT_DIR} already has files, nothing was run")  # refuse to start, the outputs are created once

start = time.time()  # start of the run

# Works by column only, then the dated Works with a style tag, as in the series of the entry.
works = pd.read_parquet(WORKS_PATH, columns=["work_id", "work_source", "year", "styles"])  # four columns of the cleaned Works table
kept, dropped = keep_dated_styled(works)  # dated and styled Works, drop counts by reason
del works  # free the raw table
print(f"Works kept in the series: {len(kept):,}; dropped: {dropped}", flush=True)  # the entry gives 510,364 undated and 851,725 untagged

# Master versus synthetic comparison of the 2023 growth of the top styles.
tables = universe_tables(kept, range(FIRST_YEAR, LAST_YEAR + 1))  # style by year tables for all, master and synthetic Works
overlap = {name: top_k_overlap(tables["all"], tables[name], BASE_YEAR, k=TOP_K, min_base=MIN_BASE) for name in ("master", "synthetic")}  # top 10 under each universe on the all-Works eligible set
top = overlap["master"]["top_all"]  # top 10 styles by 2023 growth under all Works
growth = pd.DataFrame({name: smoothed_growth(tables[name], BASE_YEAR) for name in ("all", "master", "synthetic")})  # 2023 growth per style under each universe
top_table = growth.loc[top.index].assign(rank_all=range(1, len(top) + 1))  # growth of the top 10 under the three universes
positive = {name: int((top_table[name] > 0).sum()) for name in ("master", "synthetic")}  # top styles whose growth is positive in each universe
print(f"top {TOP_K} styles by {BASE_YEAR} growth (all Works), eligible styles: {overlap['master']['eligible']}", flush=True)  # eligible count of the growth ranking

# Label figures for the Works dated in the base year of every eligible style.
elig_target = three_year_base(tables["all"], BASE_YEAR + 1) >= MIN_BASE  # eligible for the 2024 projection: mean of 2021 to 2023 at least 100
elig_growth = three_year_base(tables["all"], BASE_YEAR) >= MIN_BASE  # eligible for the 2023 growth: mean of 2020 to 2022 at least 100
styles = sorted(set(elig_target[elig_target].index) | set(elig_growth[elig_growth].index) | set(ENTRY_PERCENTILE_STYLES + ENTRY_OTHER_STYLES) & set(tables["all"].index))  # eligible under either rule, plus the styles named in the entry
base_works = kept[kept["year"] == BASE_YEAR][["work_id", "work_source", "styles"]].copy()  # Works dated in the base year
release_ids = work_release_ids(base_works)  # release that represents each Work, masters.parquet read by two columns
wanted = release_ids.dropna().astype("int64").unique().tolist()  # release ids to look up
info = load_release_info(wanted)  # label names of the wanted releases only, four columns, filtered read
labels = release_ids.map(info["label_names"])  # label names per work_id, NaN where the release was not found
unlinked = int(labels.isna().sum())  # Works whose release was not found
base_works["label_names"] = [x if isinstance(x, list) else [] for x in base_works["work_id"].map(labels)]  # a Work without a found release counts as a Work with no label
print(f"Works dated {BASE_YEAR}: {len(base_works):,}; release not found: {unlinked:,}", flush=True)  # coverage of the release link
long = base_works.explode("styles")  # one row per Work and style
rows = []  # one row of label figures per style
for style, group in long[long["styles"].isin(styles)].groupby("styles"):  # each style in turn
    rows.append({"style": style, **label_concentration(group, k=3)})  # label figures of the style, top-3 share as in the entry
labels_table = pd.DataFrame(rows).set_index("style")  # one row per style
labels_table["eligible_2024_base"] = elig_target.reindex(labels_table.index).fillna(False)  # eligible for the projection
labels_table["eligible_2023_growth"] = elig_growth.reindex(labels_table.index).fillna(False)  # eligible for the 2023 growth
for flag, column in (("eligible_2024_base", "pct_top3_share"), ("eligible_2023_growth", "pct_top3_share_alt")):  # the two eligibility rules
    pool = labels_table.loc[labels_table[flag], "top3_share"]  # top-3 share of the eligible styles
    labels_table[column] = labels_table["top3_share"].map(lambda v: 100 * (pool <= v).mean())  # share of eligible styles at or below the value, in percent
labels_table["pct_missing_share"] = labels_table["missing_share"].map(lambda v: 100 * (labels_table.loc[labels_table["eligible_2024_base"], "missing_share"] <= v).mean())  # same percentile for the no-label share

# Comparison with the entry.
comparison = []  # figure in the entry, recomputed value, match
for style in ENTRY_PERCENTILE_STYLES:  # the four styles with a figure in the entry
    if style not in labels_table.index:  # the tag does not exist or has no Work in the base year
        comparison.append({"figure": f"{style}: top-3 label share percentile", "entry": "91st to 100th", "recomputed": "style not found", "match": "no"})  # reported, not hidden
        continue  # next style
    pct, alt = labels_table.loc[style, "pct_top3_share"], labels_table.loc[style, "pct_top3_share_alt"]  # percentile under both eligibility rules
    ok = "yes" if 91 <= round(pct) <= 100 else "no"  # integer percentile inside the range of the entry
    comparison.append({"figure": f"{style}: top-3 label share percentile", "entry": "91st to 100th", "recomputed": f"{pct:.1f} (alt eligibility {alt:.1f}); top-3 share {labels_table.loc[style, 'top3_share']:.3f}", "match": ok})  # one row per style
for style in ["Noise Rock", "Alt-Pop"]:  # styles the entry calls not concentrated by label
    if style in labels_table.index:  # the style exists
        comparison.append({"figure": f"{style}: top-3 label share percentile", "entry": "not concentrated by label (no figure)", "recomputed": f"{labels_table.loc[style, 'pct_top3_share']:.1f} (alt eligibility {labels_table.loc[style, 'pct_top3_share_alt']:.1f})", "match": "n/a, no figure in the entry"})  # descriptive
for style in ENTRY_PERCENTILE_STYLES + ENTRY_OTHER_STYLES:  # all styles named in the entry
    if style in growth.index:  # the style exists in the table
        g = growth.loc[style]  # growth under the three universes
        comparison.append({"figure": f"{style}: {BASE_YEAR} growth in top {TOP_K}?", "entry": "named as a top or spike style", "recomputed": f"all {g['all']:.2f}, master {g['master']:.2f}, synthetic {g['synthetic']:.2f}; in top {TOP_K} (all Works): {style in top.index}", "match": "n/a, no figure in the entry"})  # descriptive
comparison.append({"figure": f"{BASE_YEAR} growth of the top {TOP_K} styles present in master Works", "entry": "present (no figure)", "recomputed": f"{positive['master']} of {TOP_K} with growth above 0; overlap of the top {TOP_K}: {overlap['master']['overlap']}; undefined growth: {overlap['master']['undefined_other']}", "match": "yes" if positive["master"] == TOP_K else f"no ({positive['master']} of {TOP_K})"})  # present means growth above 0
comparison.append({"figure": f"{BASE_YEAR} growth of the top {TOP_K} styles present in synthetic Works", "entry": "present (no figure)", "recomputed": f"{positive['synthetic']} of {TOP_K} with growth above 0; overlap of the top {TOP_K}: {overlap['synthetic']['overlap']}; undefined growth: {overlap['synthetic']['undefined_other']}", "match": "yes" if positive["synthetic"] == TOP_K else f"no ({positive['synthetic']} of {TOP_K})"})  # present means growth above 0
comparison = pd.DataFrame(comparison)  # table to print and save

pd.set_option("display.width", 250)  # readable printed tables
pd.set_option("display.max_colwidth", 120)  # long cells stay on one line
print("\nParameters not described in the entry (chosen here):", flush=True)  # say what was chosen
print(f"- label figures use the Works of each style dated in {BASE_YEAR}; percentile = share of eligible styles with a value at or below the style's, eligibility of the 2024 projection (mean of 2021 to 2023 at least {MIN_BASE}), alternative rule in brackets (mean of 2020 to 2022)", flush=True)  # assumptions
print("- 'present in a universe' = 2023 growth above 0 in that universe; a Work whose release was not found counts as a Work with no label; top-3 share uses the module default k=3", flush=True)  # assumptions
print("\nComparison with docs/decisions.md, line 111:", flush=True)  # header of the table
print(comparison.to_string(index=False), flush=True)  # figure, recomputed value, match

OUT_DIR.mkdir(parents=True, exist_ok=True)  # created only after the run succeeded
comparison.to_csv(OUT_DIR / "comparison.csv", mode="x", index=False)  # exclusive creation
top_table.to_csv(OUT_DIR / "top_growth_2023.csv", mode="x")  # top 10 by 2023 growth under the three universes
labels_table.round(6).to_csv(OUT_DIR / "label_figures_2023.csv", mode="x")  # label figures and percentiles per style
peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # peak resident memory, bytes on macOS
print(f"\nwritten to {OUT_DIR}; runtime {time.time() - start:.0f}s; peak memory {peak_gb:.2f} GB", flush=True)  # runtime and peak memory
