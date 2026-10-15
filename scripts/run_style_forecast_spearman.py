"""Spearman interval on the development years: bootstrap over styles within each year (console only).

Recomputes the walk-forward predictions with the same function as the development script,
uses the locked Random Forest, checks the other methods against the saved per-year hits, and reports the mean Spearman per method with its interval.

Usage:
    venv/bin/python3 scripts/run_style_forecast_spearman.py
"""

import sys  # module search path
from pathlib import Path  # project root

import pandas as pd  # tables

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from style_forecast import BASELINES, DEV_LAST_TARGET, LOCKED_RF_PARAMS, MODELS, build_panel, hits_table, spearman_boot, walk_forward  # forecast functions
from style_series import keep_dated_styled, load_works  # series from the first phase

kept, _ = keep_dated_styled(load_works())  # Works with a year and a style
kept = kept[kept["year"] <= DEV_LAST_TARGET]  # nothing after the last development year is read
results = walk_forward(build_panel(kept, DEV_LAST_TARGET), rf_params=LOCKED_RF_PARAMS)  # same walk-forward predictions as the development script, locked forest

saved = pd.read_csv(ROOT / "results/style_forecast/dev_hits.csv", index_col="target_year")  # hits written by the development script
fresh = hits_table(results, 10)[BASELINES + MODELS].astype(float)  # hits@10 recomputed here
same = all((fresh[m].to_numpy() == saved["hits10_" + m].to_numpy()).all() for m in BASELINES + ["bayes_ridge", "xgboost"])  # every untuned cell equal, the forest is the locked one
print("recomputed hits@10 of the baselines and the untuned models equal to the saved development hits:", same)

print("\n== Spearman on the styles scored by every method, development years 1993 to 2017")
print("mean over 25 years and 95% percentile interval (2,000 draws over styles within year, seed 1)")
print(spearman_boot(results).round(3).to_string())
