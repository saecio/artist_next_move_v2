# Project Pipeline Overview

Short, scannable status of the pipeline as it stands. Reasoning and counts live elsewhere: [decisions.md](decisions.md) (chronological log), [work_definition.md](work_definition.md) (what a Work is and how it is built), [abandoned_approaches.md](abandoned_approaches.md) (failed methods). Last updated: 2026-10-10 (after the Master-only revert).

## Objective

Predict an artist's next move from temporal snapshots, out-of-time evaluation only (never a random split).

- **WHEN**: will a new Work arrive within 1 and 2 years of the snapshot?
- **WHAT**: will the next Work introduce a style the artist never used before? (can be dropped if it fails the kill gate)

Models: Logistic Regression, KNN, Random Forest, XGBoost, always compared against historical baselines. Demo: Streamlit "Time Machine".

## Pipeline

1. **Raw sources** (`data/raw/`, not versioned, see [README](../README.md)): Discogs dump dated 2026-09-01 (masters and releases XML).
2. **Parsing** (`src/parse_dumps.py`): streams each XML into `data/processed/masters.parquet` (2,589,349 rows) and `releases.parquet` (~19.4M rows).
3. **Cleaning** (`notebooks/01_cleaning_eda.ipynb` + `src/works.py`): masters and standalone releases are exclusion-filtered, combined into one Works table, then deduplicated and filtered. Writes `data/processed/works_checkpoint.parquet`: **5,949,217 rows, all sources (master + synthetic)**, 1,034,756 artists. Steps, order and counts: [work_definition.md](work_definition.md).
4. **Year handling**:
   - `recover_missing_master_years` recovers 38,670 master years from their Releases.
   - Three imputation attempts (id ordering, MusicBrainz title match, MusicBrainz catalog-number join) were abandoned, see [abandoned_approaches.md](abandoned_approaches.md).
   - Master-only filter tried and reversed on 2026-10-10 (see CUT in [decisions.md](decisions.md)).
   - Result: 510,364 works (8.58%) keep `year = NaN`. `year_missing` exists at the artist level in `src/eda.py`; the work-level flag is still pending, to be added in the snapshots block.
5. **EDA** (`notebooks/02_eda_visualization.ipynb`, `src/eda.py`): univariate, bivariate, multivariate and hypothesis tests on `works_checkpoint.parquet`. Findings are logged in [decisions.md](decisions.md).
6. **Baselines** (`src/baselines.py`): historical baselines for WHEN and WHAT, plus the kill gate. A model passes only if the lower bound of its 95% artist-level bootstrap PR-AUC interval is above the upper bound of the baseline's.
7. **Snapshots** (`src/snapshots.py`): `build_snapshots` turns the Works table into one row per dated work (5,438,765 on the full checkpoint, about 12 seconds) with the WHEN and WHAT targets. WHEN looks at the next year with activity, not the next work. Re-validated at full scale on 2026-10-10, see [decisions.md](decisions.md).

## Next

- Snapshots at scale, plus the work-level `year_missing` flag.
- Open items are tracked under PENDING in [decisions.md](decisions.md): final feature set, temporal cutoffs, handling of high-volume artists, per-feature missing-year handling.
- Modeling and baseline comparison, then the Streamlit demo, README and slide deck.

## Calendar

- Oct 6-25: development.
- Oct 26-30: presentation prep.
- Oct 31: presentation.
