"""Run the Random Forest style forecast phase 4 on the development target years 1993 to 2017 only.

Three parts, console output only, no file written and the trial log is only read:
adoption evaluation from the log, train versus validation diagnostic for stage 0 and the candidate,
and the pass rule of the design entry applied again with the locked Random Forest.

Usage:
    venv/bin/python3 scripts/run_rf_style_adoption.py > results/trials/rf_style_forecast_adoption_$(date +%Y%m%d_%H%M%S).out 2>&1
"""

import sys  # module search path
from pathlib import Path  # project root

import numpy as np  # arrays
import pandas as pd  # tables
from scipy.stats import spearmanr  # rank correlation

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from style_forecast import BASELINES, DEV_LAST_TARGET, MODELS, TRANCHES, boot_ci, build_panel, hits_table, pass_rule, spearman_table, walk_forward_tuned  # forecast functions
from style_series import keep_dated_styled, load_works  # series from the first phase
from tuning import rf_adoption, rf_overfit_summary, rf_overfit_table  # phase 4 helpers

LOG_PATH = ROOT / "results/trials/rf_style_forecast.jsonl"  # trial log, read only
DEV_CSV = ROOT / "results/style_forecast/dev_hits.csv"  # development hits from the first run, read only
pd.set_option("display.width", 250)  # wide console tables
pd.set_option("display.max_columns", 30)  # show every column

print("== step 1: adoption evaluation, from the log (no refit)")
ad = rf_adoption(LOG_PATH)  # candidate against stage 0
print("log rows used for the candidate: best non-stage0 trial by mean Spearman")
print("candidate:", ad["candidate"]["trial"], ad["candidate"]["params"], "mean Spearman", round(ad["candidate"]["mean_spearman"], 6), "mean hits@10", ad["candidate"]["mean_hits10"])
print("stage0:", ad["stage0"]["params"], "mean Spearman", round(ad["stage0"]["mean_spearman"], 6), "mean hits@10", ad["stage0"]["mean_hits10"])
for name, (ok, value) in ad["conditions"].items():  # each condition with its number
    print(f"{ok!s:5} | {name} | {value}")
print("candidate adopted:", ad["adopted"])
print("locked Random Forest:", ad["locked"])

print("\n== step 2: train versus validation diagnostic (Spearman, fold's own training rows against target year)")
kept, _ = keep_dated_styled(load_works())  # Works with a year and a style
kept = kept[kept["year"] <= DEV_LAST_TARGET]  # nothing after the last development year is ever used
panel = build_panel(kept, DEV_LAST_TARGET)  # style by origin rows
for label, row in [("stage0", ad["stage0"]), (ad["candidate"]["trial"], ad["candidate"])]:  # both configurations, always
    table = rf_overfit_table(panel, row["params"])  # per-year train, validation and gap
    same = np.allclose(table["validation"].to_numpy(), row["spearman"])  # validation must equal the log value
    print(f"-- {label}, validation equals the log row in all 25 years: {same}")
    print(rf_overfit_summary(table).round(4).to_string())

print("\n== step 3: pass rule with the locked Random Forest")
results = walk_forward_tuned(panel, ad["locked"])  # all methods, forest with the locked settings
dev = pd.read_csv(DEV_CSV, index_col="target_year")  # development hits of the first run
hits = {k: hits_table(results, k) for k in (10, 5)}  # hits@10 and hits@5 per year
ok_untuned = all(  # untuned models and baselines against the first run
    (hits[k][m].astype(int).to_numpy() == dev[f"hits{k}_{m}"].astype(int).to_numpy()).all()
    for k in (10, 5) for m in BASELINES + ["bayes_ridge", "xgboost"]
)
row = ad["candidate"] if ad["adopted"] else ad["stage0"]  # log row of the locked forest
ok_hits = all((hits[k]["random_forest"].astype(int).to_numpy() == np.array(row[f"hits{k}"])).all() for k in (10, 5))  # forest hits against its log row
log_sp = np.array([spearmanr(g["random_forest"], g["label"])[0] for _, g in results.groupby("target")])  # forest Spearman over all eligible styles
ok_sp = np.allclose(log_sp, row["spearman"])  # forest Spearman against its log row
print("untuned models and baselines reproduce dev_hits.csv:", ok_untuned)
print("Random Forest reproduces the hits@10 and hits@5 of its log row:", ok_hits)
print("Random Forest reproduces the Spearman of its log row:", ok_sp)
if not (ok_untuned and ok_hits and ok_sp):  # stop and report if anything differs
    raise SystemExit("check failed, stopping before the pass rule")

methods = ["random"] + BASELINES + MODELS  # every method in report order
print("\n== mean hits per method (95% interval by bootstrap over years)")
for k, table in hits.items():  # both cut-offs
    summary = {m: dict(zip(["mean", "ci_low", "ci_high"], boot_ci(table[m].to_numpy(dtype=float)))) for m in methods}  # pooled mean and interval
    print(f"-- hits@{k}")
    print(pd.DataFrame(summary).T.round(2).to_string())
print("\n== mean hits by tranche")
for k, table in hits.items():  # both cut-offs
    rows = {m: {f"{a}-{b}": table.loc[a:b, m].astype(float).mean() for a, b in TRANCHES} for m in methods}  # five tranche means
    print(f"-- hits@{k}")
    print(pd.DataFrame(rows).T.round(2).to_string())
spearman = spearman_table(results)  # rank correlation per year, common styles
print("\n== mean Spearman on the styles scored by every method (25 years)")
print(spearman[BASELINES + MODELS].mean().round(3).to_string())

rule = pass_rule(hits[10], spearman)  # five conditions
print(f"\n== pass rule: candidate {rule['candidate']}, best baseline {rule['baseline']}")
for name, (ok, value) in rule["conditions"].items():  # each condition with its number
    print(f"{ok!s:5} | {name} | {value}")
print("paired difference vs best baseline (mean, low, high):", [round(x, 3) for x in rule["diff_base"]])
print("paired difference vs random (mean, low, high):", [round(x, 3) for x in rule["diff_rand"]])
print("recommended ranking:", rule["recommended"])
