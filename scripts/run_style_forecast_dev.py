"""Run the style forecast baselines and models once on the development target years 1993 to 2017.

Prints every table and writes one small CSV of per-year hits (exclusive mode, fails if it exists).
No test year is read or scored, and no model is saved.

Usage:
    venv/bin/python3 scripts/run_style_forecast_dev.py
"""

import sys  # module search path
from pathlib import Path  # project root

import pandas as pd  # tables

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from style_forecast import BASELINES, DEV_LAST_TARGET, MODELS, TRANCHES, boot_ci, build_panel, count_metrics, count_table, hits_table, pass_rule, spearman_table, truncation_test, walk_forward  # forecast functions
from style_series import keep_dated_styled, load_works  # series from the first phase

OUT_CSV = ROOT / "results/style_forecast/dev_hits.csv"  # the only file written
pd.set_option("display.width", 250)  # wide console tables
pd.set_option("display.max_columns", 30)  # show every column

kept, _ = keep_dated_styled(load_works())  # Works with a year and a style
kept = kept[kept["year"] <= DEV_LAST_TARGET]  # nothing after the last development year is ever used
panel = build_panel(kept, DEV_LAST_TARGET)  # style by origin rows
results = walk_forward(panel)  # one fit per model and target year

print("== eligible styles and training rows per target year")
sizes = results.groupby("target").size().rename("eligible").to_frame()  # eligible styles per target year
sizes["train_rows"] = [len(panel[(panel["origin"] <= t - 2) & panel["label"].notna()]) for t in sizes.index]  # rows with a label known at the origin
print(sizes.T.to_string())

methods = ["random"] + BASELINES + MODELS  # every method in report order
hits = {k: hits_table(results, k) for k in (10, 5)}  # hits@10 and hits@5 per year
for k, table in hits.items():  # print both tables
    print(f"\n== hits@{k} per target year")
    print(table[["eligible"] + methods].round(2).to_string())

print("\n== mean hits by tranche and pooled (95% interval by bootstrap over years)")
for k, table in hits.items():  # both cut-offs
    rows = {}  # method -> tranche means
    for m in methods:  # each method
        row = {f"{a}-{b}": table.loc[a:b, m].astype(float).mean() for a, b in TRANCHES}  # five tranche means
        mean, low, high = boot_ci(table[m].to_numpy(dtype=float))  # pooled mean and interval
        row.update({"pooled": mean, "ci_low": low, "ci_high": high})  # pooled columns
        rows[m] = row  # keep
    print(f"-- hits@{k}")
    print(pd.DataFrame(rows).T.round(2).to_string())

spearman = spearman_table(results)  # rank correlation per year
print("\n== mean Spearman on the styles scored by every method (25 years)")
print(spearman[BASELINES + MODELS].mean().round(3).to_string())
print("styles in common per year, min and median:", int(spearman["common"].min()), int(spearman["common"].median()))

rule = pass_rule(hits[10], spearman)  # five conditions
print(f"\n== pass rule: candidate {rule['candidate']}, best baseline {rule['baseline']}")
for name, (ok, value) in rule["conditions"].items():  # each condition with its number
    print(f"{ok!s:5} | {name} | {value}")
print("paired difference vs best baseline (mean, low, high):", [round(x, 3) for x in rule["diff_base"]])
print("paired difference vs random (mean, low, high):", [round(x, 3) for x in rule["diff_rand"]])
print("recommended ranking:", rule["recommended"])

print("\n== count metrics (target years 1994 to 2017, nominal coverage 0.80)")
counts = count_table(results)  # expected counts and ranges
print(count_metrics(counts, results).round(3).to_string(index=False))

print("\n== truncation test (all Works dated after t removed, features of origin t rebuilt)")
print(truncation_test(kept, panel, [1950, 1980, 1995, 2005, 2016]).to_string(index=False))

per_year = pd.concat({"hits10": hits[10][methods], "hits5": hits[5][methods]}, axis=1)  # per-year hits for every method
per_year.columns = [f"{a}_{b}" for a, b in per_year.columns]  # flat column names
per_year.insert(0, "eligible", hits[10]["eligible"])  # eligible styles per year
with open(OUT_CSV, "x") as handle:  # exclusive mode, stops if the file exists
    per_year.to_csv(handle, index_label="target_year")  # write the one CSV
print("\nwritten:", OUT_CSV)
