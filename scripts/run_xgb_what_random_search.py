"""Run the WHAT XGBoost random search (phase 1) and append every trial to results/trials/.

The log is append-only and shared by the three phases. Trials already in it are skipped, so a rerun resumes.

Usage:
    venv/bin/python3 scripts/run_xgb_what_random_search.py
"""

import sys  # module search path
from collections import Counter  # rows per phase already in the log
from pathlib import Path  # project root

import pandas as pd  # dataframe library used to read the Works table

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import make_what_folds  # development folds with the label-resolution rule
from tuning import prepare_what_fold_data, read_log, run_random_search  # tuning helpers
from what_target import build_artist_years, build_what_features, build_what_table  # WHAT table and inputs

N_TRIALS = 35  # same number of random combinations as the WHEN search
TARGET = "new_style_year"  # WHAT target
LOG_PATH = ROOT / "results/trials/xgb_what_new_style_year.jsonl"  # shared by the three phases, append-only

logged = read_log(LOG_PATH)  # empty list on the first run
print(f"log rows at start: {len(logged)}, by phase: {dict(Counter(r['phase'] for r in logged))}", flush=True)  # what a rerun will skip
works = pd.read_parquet(ROOT / "data/processed/works_checkpoint.parquet")  # cleaned Works table
table = build_what_table(build_artist_years(works))  # labelled artist-year rows, history >= 3
features = build_what_features(works)  # the 23 columns of build_what_features (vocabulary, novelty, timing and format, genre flags); the 2 table inputs complete the 25 WHAT_FEATURES
folds = make_what_folds(table)  # five development folds, the holdout is never read
fold_data = prepare_what_fold_data(folds, features, TARGET)  # DMatrix per fold, built once for all trials
run_random_search(fold_data, TARGET, N_TRIALS, LOG_PATH)  # stage0 row first, then trials 1..35, resumable
final = read_log(LOG_PATH)  # log after this run
print(f"log rows at end: {len(final)}, by phase: {dict(Counter(r['phase'] for r in final))}", flush=True)
