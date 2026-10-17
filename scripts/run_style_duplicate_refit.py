"""Style forecast duplicate audit, step B: development-only refit on the original and the collapsed series.

Rule fixed in docs/decisions.md, entry "Style forecast: duplicate audit design". Only target years up to the last
development year are fitted and scored, and the Works table is filtered to those years while it is read. The control
against results/style_forecast/dev_hits.csv runs first and no file is written if it differs.

Usage:
    venv/bin/python3 scripts/run_style_duplicate_refit.py
"""

import sys  # module search path and exit
import time  # seconds per version
from pathlib import Path  # project root

import pandas as pd  # tables

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from style_duplicate_refit import SETTINGS, check_counts_against_step_a, control_against_dev_hits, dev_works, kept_versions, method_table, per_year_table, run_version, status_table  # step B pieces
from style_forecast import pass_rule  # five acceptance conditions

OUT_DIR = ROOT / "results/style_duplicate_audit"  # step A folder, already exists
PER_YEAR_PATH = OUT_DIR / "dev_refit.csv"  # per-year hits and Spearman
VERDICT_PATH = OUT_DIR / "dev_refit_verdict.csv"  # per-method table and status of the conditions

for path in (PER_YEAR_PATH, VERDICT_PATH):  # never overwrite
    if path.exists():
        sys.exit(f"{path.relative_to(ROOT)} already exists, nothing was run")
pd.set_option("display.width", 250)  # readable printed tables
pd.set_option("display.max_rows", 200)  # show every development year

kept_original, kept_collapsed, info = kept_versions(dev_works())  # series input of both versions, rows dated up to 2017 only
compared, differ = check_counts_against_step_a(kept_original, kept_collapsed, OUT_DIR / "style_year_counts.csv")  # early filter must not change the collapse
print(f"== counts check against step A (years up to the last development year): {compared} style-years compared, {differ} differ ==", flush=True)
if differ:  # stop before any fit
    sys.exit("the filtered collapse differs from the step A counts, nothing was fitted")

print("== STEP 1: control on the original table (default settings) against results/style_forecast/dev_hits.csv, 4 decimals ==")
start = time.time()  # clock of the original version
original = run_version(kept_original)  # default and locked settings
print(f"original fits {time.time() - start:.0f} s", flush=True)
control = control_against_dev_hits(original["default"][0], ROOT / "results/style_forecast/dev_hits.csv")  # per-column differences
print(control.round(6).to_string())
if control["years_different"].sum():  # stop before the collapsed fits and any file
    sys.exit("the recomputed hits@10 differ from dev_hits.csv at 4 decimals, nothing was written")
print("control passed", flush=True)

print("== STEP 2: collapsed fits ==")
start = time.time()  # clock of the collapsed version
collapsed = run_version(kept_collapsed)  # same settings and seeds
print(f"collapsed fits {time.time() - start:.0f} s", flush=True)

per_year, tables = [], []  # per-year long tables, method and status tables
for setting in SETTINGS:  # default forest, then locked forest
    (hits_o, sp_o), (hits_c, sp_c) = original[setting], collapsed[setting]  # tables of both versions
    rule_o, rule_c = pass_rule(hits_o, sp_o), pass_rule(hits_c, sp_c)  # five conditions on each version
    status = status_table(setting, rule_o, rule_c, hits_o, hits_c)  # conditions, candidate, ranking
    methods = method_table(setting, hits_o, sp_o, hits_c, sp_c, status)  # one row per method
    per_year += [per_year_table(setting, "original", hits_o, sp_o), per_year_table(setting, "collapsed", hits_c, sp_c)]  # per-year rows
    tables += [methods, status]  # verdict file rows
    print(f"\n== setting {setting}: per-year hits@10, original then collapsed ==")
    print(pd.concat({"original": hits_o, "collapsed": hits_c}, axis=1).round(2).to_string())
    print(f"\n== setting {setting}: per-year Spearman, original then collapsed ==")
    print(pd.concat({"original": sp_o, "collapsed": sp_c}, axis=1).round(3).to_string())
    print(f"\n== setting {setting}: per method (means over the development years, 95% year-bootstrap intervals) ==")
    print(methods.drop(columns=["section", "setting"]).round(4).to_string(index=False))
    print(f"\n== setting {setting}: five conditions, candidate, baseline and ranking on both versions ==")
    print(status.drop(columns=["section", "setting"]).to_string(index=False))

print("\n== STEP 3: verdict per method (both readings, none chosen) ==")
verdicts = pd.concat([t for t in tables if t["section"].iloc[0] == "method"], ignore_index=True)  # method rows of both settings
print(verdicts[["setting", "key", "primary_outside_ci", "setting_status_changed", "primary_material", "secondary_ci_excludes_zero"]].rename(columns={"key": "method"}).to_string(index=False))

with open(PER_YEAR_PATH, "x") as handle:  # exclusive mode
    pd.concat(per_year, ignore_index=True).to_csv(handle, index=False)  # per-year table
with open(VERDICT_PATH, "x") as handle:  # exclusive mode
    pd.concat(tables, ignore_index=True).to_csv(handle, index=False)  # method and status rows
print("\nwritten", PER_YEAR_PATH.relative_to(ROOT), VERDICT_PATH.relative_to(ROOT))
