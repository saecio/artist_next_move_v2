"""Style series duplicate audit, steps A (counts only): no model is fitted and no growth is computed for 2018 or later.

Writes results/style_duplicate_audit/style_year_counts.csv and summary.csv, both in exclusive mode.

Usage:
    venv/bin/python3 scripts/run_style_duplicate_audit.py
"""

import sys  # module search path and exit
from pathlib import Path  # project root

import pandas as pd  # tables

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from style_duplicate_audit import DEV_LAST_TARGET, count_comparison, growth_agreement, kept_versions, load_audit_works, partition_summary, styles_over_share, top_removed_styles  # audit pieces

OUT_DIR = ROOT / "results/style_duplicate_audit"  # new result folder
COUNTS_PATH = OUT_DIR / "style_year_counts.csv"  # per style and year counts
SUMMARY_PATH = OUT_DIR / "summary.csv"  # summaries

for path in (COUNTS_PATH, SUMMARY_PATH):  # never overwrite
    if path.exists():
        sys.exit(f"{path.relative_to(ROOT)} already exists, nothing was run")
pd.set_option("display.width", 250)  # readable printed tables
pd.set_option("display.max_rows", 100)  # show the 25 development years

kept_original, kept_collapsed, info = kept_versions(load_audit_works())  # both versions of the series input
print("== collapse counts ==")
print(pd.Series(info).to_string())
print(f"kept Works (dated and styled): original {len(kept_original)}, collapsed {len(kept_collapsed)}", flush=True)

counts = count_comparison(kept_original, kept_collapsed)  # per style and year
print("== counts: rows", len(counts), "styles", counts["style"].nunique(), "years", counts["year"].min(), "to", counts["year"].max())

partitions = partition_summary(counts)  # totals per partition
print("== total style counts and removed share per partition ==")
print(partitions.round(5).to_string(index=False))

top = top_removed_styles(counts, DEV_LAST_TARGET)  # 20 highest removed shares up to the last development year
print(f"== 20 styles with the highest removed share, years up to {DEV_LAST_TARGET} (share of summed counts) ==")
print(top.round(5).to_string(index=False))

over = pd.DataFrame([
    {"section": "styles_over_5pct", "key": "any_year", "count": styles_over_share(counts)},  # all years
    {"section": "styles_over_5pct", "key": f"to_{DEV_LAST_TARGET}", "count": styles_over_share(counts, DEV_LAST_TARGET)},  # development years only
])
print("== styles with a removed share above 5% in at least one year ==")
print(over.to_string(index=False))

agreement = growth_agreement(kept_original, kept_collapsed)  # development years only
print(f"== growth ranking agreement, original versus collapsed, years up to {DEV_LAST_TARGET} ==")
print(agreement.drop(columns="section").round(4).to_string(index=False))

summary = pd.concat([partitions, top, over, agreement], ignore_index=True)  # one long table
OUT_DIR.mkdir(parents=True, exist_ok=True)  # create the folder on first use
with open(COUNTS_PATH, "x") as handle:  # exclusive mode
    counts.to_csv(handle, index=False)  # per style and year
with open(SUMMARY_PATH, "x") as handle:  # exclusive mode
    summary.to_csv(handle, index=False)  # summaries
print("written", COUNTS_PATH.relative_to(ROOT), len(counts), "rows;", SUMMARY_PATH.relative_to(ROOT), len(summary), "rows")
