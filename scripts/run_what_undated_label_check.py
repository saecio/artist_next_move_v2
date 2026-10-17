"""Measure how much the WHAT label depends on undated Works, on the seeded 30,000-artist sample of what_undated_check.

Label A is the real label (styles of undated Works count as already seen). Label B is the same rule with the styles of
undated Works NOT counted as seen. B is built by mirroring lines 127-130 of src/what_target.py (the label rule of
build_artist_years) on the sample without its undated Works. A rebuilt by the same mirror is asserted equal to the real label.

Usage:
    venv/bin/python3 scripts/run_what_undated_label_check.py
"""

import sys  # module search path and exit
import time  # runtime report
from pathlib import Path  # project root

import numpy as np  # NaN-aware comparison
import pandas as pd  # tables

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from config import CONFIG  # project settings from config.yaml
from what_target import _artist_sample, build_artist_years  # same sample draw and the real label builder

N_ARTISTS = 30_000  # sample size of what_undated_check (default of its n_artists)
SEED = CONFIG["project"]["SEED"]  # seed of what_undated_check (default of its seed, project.SEED)
MIN_HISTORY = CONFIG["what"]["min_previous_works"]  # history rule of build_what_table
HOLDOUT_START = CONFIG["years"]["holdout_start"]  # first holdout year
HOLDOUT_END = CONFIG["years"]["holdout_end"]  # last holdout year, labels must resolve by then
OUT_DIR = ROOT / "results/what_undated_check"  # output folder
OUT_CSV = OUT_DIR / "undated_label_counts.csv"  # the one output file

if OUT_CSV.exists():  # outputs are written once
    sys.exit(f"{OUT_CSV.name} already exists, nothing was run")  # stop before any work

start = time.time()  # clock of the whole run
works = pd.read_parquet(ROOT / CONFIG["paths"]["works"])  # cleaned Works table
sample = _artist_sample(works, N_ARTISTS, SEED)  # all Works of the seeded artist sample, as in what_undated_check
full = build_artist_years(sample)  # artist-years with the real label A (undated styles in the seen set)
dated = build_artist_years(sample[sample["year"].notna()])  # same artist-years, undated Works removed (no undated baseline)
keys = ["artist_id", "year"]  # row identity
assert full[keys].equals(dated[keys]), "the two builds do not have the same rows"  # undated Works define no row

def mirror(frame: pd.DataFrame) -> np.ndarray:
    """Label rule of what_target.py lines 127-130: 1 if the next year has a style outside styles_seen, NaN if the next year has none."""
    out = np.full(len(frame), np.nan)  # NaN where the next year has no styles (line 127 condition)
    for i, (seen, nxt) in enumerate(zip(frame["styles_seen"], frame["next_year_styles"])):  # one row at a time
        if nxt:  # label only when the next year has styles (line 129)
            out[i] = float(len(nxt - seen) > 0)  # 1 if a style is not yet seen (line 130)
    return out  # label array in row order

label_a = mirror(full)  # label A rebuilt by the mirror, seen set includes undated styles
label_b = mirror(dated)  # label B, seen set built from dated Works only
assert np.array_equal(label_a, full["new_style_year"].to_numpy(), equal_nan=True), "mirror of label A differs from the real label"  # stop if the mirror is not exact
assert np.array_equal(np.isnan(label_a), np.isnan(label_b)), "A and B are labelled on different rows"  # labelled rows do not depend on undated Works

rows = full[keys + ["next_active_year", "history_count"]].copy()  # row identity, resolution year and history
rows["a"], rows["b"] = label_a, label_b  # both labels
rows = rows[(rows["history_count"] >= MIN_HISTORY) & rows["a"].notna()]  # rows of build_what_table
rows["differ"] = rows["a"] != rows["b"]  # label changes
rows["a0b1"] = (rows["a"] == 0) & (rows["b"] == 1)  # undated styles lower the label
rows["a1b0"] = (rows["a"] == 1) & (rows["b"] == 0)  # opposite direction, not expected

dev = rows[rows["year"] < HOLDOUT_START]  # development rows, no resolution restriction (ambiguity: the folds also require next_active_year < test start)
hold = rows[(rows["year"] >= HOLDOUT_START) & (rows["next_active_year"] <= HOLDOUT_END)]  # holdout label rule, years 2018 to 2022 in practice

def summarize(name: str, year, part: pd.DataFrame) -> dict:
    """Counts and rates of one group: rows, positive rate under A and B, differences and their direction."""
    return {"scope": name, "year": year, "rows": len(part), "pos_rate_a": part["a"].mean(), "pos_rate_b": part["b"].mean(),  # size and rates
            "differ": int(part["differ"].sum()), "differ_share": part["differ"].mean(),  # rows where A and B differ
            "a0_b1": int(part["a0b1"].sum()), "a1_b0": int(part["a1b0"].sum())}  # direction of the differences

table = [summarize("development", "all", dev), summarize("holdout", "all", hold)]  # the two scopes
for year in range(2013, 2023):  # origin years 2013 to 2022
    part = dev[dev["year"] == year] if year < HOLDOUT_START else hold[hold["year"] == year]  # development rule before 2018, holdout rule from 2018
    table.append(summarize("development" if year < HOLDOUT_START else "holdout", year, part))  # one row per origin year
result = pd.DataFrame(table)  # all counts in one table

OUT_DIR.mkdir(parents=True, exist_ok=True)  # create the folder if missing
result.to_csv(OUT_CSV, mode="x", index=False)  # exclusive creation, fails if the file exists
pd.set_option("display.width", 250)  # readable printed table
print(result.round(5).to_string(index=False), flush=True)  # printed table
elapsed = time.time() - start  # seconds of the whole run
print(f"artists in sample {sample['artist_id'].nunique()}, rows in the WHAT population {len(rows)}, elapsed {elapsed:.1f} s, {elapsed / (N_ARTISTS / 1000):.2f} s per 1,000 artists", flush=True)  # runtime report
