"""Build the small data files of the static tabs under results/demo/static/ (exclusive creation).

When the first four files already exist, only works_per_year.csv is created and the two source counts are appended to data_figures.csv.

Usage:
    venv/bin/python3 scripts/run_static_assets.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

import pandas as pd  # cleaned Works table

from config import CONFIG  # project settings
from static_assets import build_figures, build_h1, build_h6, build_source_counts, build_summary, build_works_per_year, check_controls, check_new_controls, output_paths, static_config, write_new  # asset logic

sc = static_config()  # static_tabs settings
paths = output_paths(sc)  # the five output files
base_keys = ("h1_buckets", "h6_buckets", "data_figures", "hypotheses")  # files of the first build
if paths["works_per_year"].exists():  # never overwrite
    sys.exit(f"refusing to run, this file exists: {paths['works_per_year']}")  # stop with a message
present = [k for k in base_keys if paths[k].exists()]  # first-build files already there
if present and len(present) != len(base_keys):  # a half-built state is not touched
    sys.exit(f"refusing to run, only some files exist: {present}")  # stop with a message
extend = bool(present)  # True when the first four files are already built

works = pd.read_parquet(ROOT / CONFIG["paths"]["works"], columns=["artist_id", "work_id", "year", "work_source", "styles", "genres"])  # cleaned Works, columns the functions need
per_year = build_works_per_year(works)  # dated Works per year
sources = build_source_counts(works, sc["rows"])  # Works per source
if extend:  # first four files stay as they are
    figures = pd.read_csv(paths["data_figures"])  # stored Works and artists
else:  # first build
    h1 = build_h1(works)  # Hypothesis 1 buckets
    h6 = build_h6()  # Hypothesis 6 buckets
    figures = build_figures(works)  # Works and artists
    summary = build_summary(sc, h1, h6)  # hypotheses table
    failed = check_controls(sc, h1, h6, figures)  # printed controls
    if failed:  # a control does not hold: nothing is written
        sys.exit(f"controls failed: {failed}; no file written")  # report instead of forcing
failed = check_new_controls(works, per_year, sources, figures)  # printed controls of the new files
if failed:  # a control does not hold: nothing is written
    sys.exit(f"controls failed: {failed}; no file written")  # report instead of forcing

paths["works_per_year"].parent.mkdir(parents=True, exist_ok=True)  # output folder
if extend:  # append the two source counts, existing rows are untouched
    if not set(sources["metric"]) & set(figures["metric"]):  # not appended yet
        with open(paths["data_figures"], "a", encoding="utf-8", newline="") as f:  # append mode
            sources.to_csv(f, index=False, header=False)  # two rows, no header
        print(f"appended {len(sources)} rows to {paths['data_figures'].relative_to(ROOT)}")  # confirmation
    outputs = [("works_per_year", per_year)]  # only the new file
else:  # first build
    outputs = [("h1_buckets", h1), ("h6_buckets", h6), ("data_figures", pd.concat([figures, sources], ignore_index=True)), ("hypotheses", summary), ("works_per_year", per_year)]  # five files
for key, table in outputs:  # files to create
    write_new(paths[key], table)  # exclusive creation
    print(f"written {paths[key].relative_to(ROOT)}")  # confirmation
