"""Run the Random Forest style forecast manual trials (phase 3) and append every trial to results/trials/.

Scored on the 25 development target years 1993 to 2017 only, no test year is read or scored.
The log is append-only and shared with phases 1 and 2. Trials already in it are skipped, so a rerun resumes.

Usage:
    venv/bin/python3 scripts/run_rf_style_manual.py > results/trials/rf_style_forecast_manual_$(date +%Y%m%d_%H%M%S).out 2>&1
"""

import sys  # module search path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from style_forecast import DEV_LAST_TARGET, build_panel  # development limit and panel builder
from style_series import keep_dated_styled, load_works  # series from the first phase
from tuning import read_log, rf_base_params, run_rf_manual_search  # tuning helpers

LOG_PATH = ROOT / "results/trials/rf_style_forecast.jsonl"  # append-only trial log shared with phases 1 and 2

print(f"log rows at start: {len(read_log(LOG_PATH))}", flush=True)  # what a rerun will skip
print("base configuration (best non-stage0 trial):", *rf_base_params(LOG_PATH), flush=True)  # id and settings every manual trial changes one of
kept, _ = keep_dated_styled(load_works())  # Works with a year and a style
kept = kept[kept["year"] <= DEV_LAST_TARGET]  # nothing after the last development year is ever used
panel = build_panel(kept, DEV_LAST_TARGET)  # style by origin rows, built once for all trials
run_rf_manual_search(panel, LOG_PATH)  # seven trials, absolute_error last, resumable
print(f"log rows at end: {len(read_log(LOG_PATH))}", flush=True)  # log after this run
