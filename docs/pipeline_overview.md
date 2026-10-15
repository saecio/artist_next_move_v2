# Project Pipeline Overview

Short, scannable status of the pipeline as it stands. Reasoning and counts live elsewhere: [decisions.md](decisions.md) (chronological log), [work_definition.md](work_definition.md) (what a Work is and how it is built), [abandoned_approaches.md](abandoned_approaches.md) (failed methods). Last updated: 15-10-2026 (sections "WHAT" and "Style trend forecast" added; the other sections describe the state of 10-10-2026).

## Objective

Predict an artist's next move from temporal snapshots, out-of-time evaluation only (never a random split).

- **WHEN**: will a new Work arrive within 1 and 2 years of the snapshot?
- **WHAT**: will the next Work introduce a style the artist never used before? (can be dropped if it fails the kill gate)

Models compared: Logistic Regression, Naive Bayes, Random Forest, HistGradientBoosting and XGBoost, always against historical baselines. Demo: Streamlit "Time Machine".

## Pipeline

1. **Raw sources** (`data/raw/`, not versioned, see [README](../README.md)): Discogs dump dated 01-09-2026 (masters and releases XML).
2. **Parsing** (`src/parse_dumps.py`): streams each XML into `data/processed/masters.parquet` (2,589,349 rows) and `releases.parquet` (~19.4M rows).
3. **Cleaning** (`notebooks/01_cleaning_eda.ipynb` + `src/works.py`): masters and standalone releases are exclusion-filtered, combined into one Works table, then deduplicated and filtered. Writes `data/processed/works_checkpoint.parquet`: **5,949,217 rows, all sources (master + synthetic)**, 1,034,756 artists. Steps, order and counts: [work_definition.md](work_definition.md).
4. **Year handling**:
   - `recover_missing_master_years` recovers 38,670 master years from their Releases.
   - Three imputation attempts (id ordering, MusicBrainz title match, MusicBrainz catalog-number join) were abandoned, see [abandoned_approaches.md](abandoned_approaches.md).
   - Master-only filter tried and reversed on 10-10-2026 (see CUT in [decisions.md](decisions.md)).
   - Result: 510,364 works (8.58%) keep `year = NaN`. `year_missing` exists at the artist level in `src/eda.py`; the work-level flag is still pending, to be added in the snapshots block.
5. **EDA** (`notebooks/02_eda_visualization.ipynb`, `src/eda.py`): univariate, bivariate, multivariate and hypothesis tests on `works_checkpoint.parquet`. Findings are logged in [decisions.md](decisions.md).
6. **Baselines** (`src/baselines.py`): historical baselines for WHEN and WHAT, plus the kill gate. A model passes only if the lower bound of its 95% artist-level bootstrap PR-AUC interval is above the upper bound of the baseline's.
7. **Snapshots** (`src/snapshots.py`): `build_snapshots` turns the Works table into one row per dated work (5,438,765 on the full checkpoint, about 12 seconds) with the WHEN and WHAT targets. WHEN looks at the next year with activity, not the next work. Re-validated at full scale on 10-10-2026, see [decisions.md](decisions.md).

## WHAT

Facts below come from [decisions.md](decisions.md) and `results/holdout_what/headline.csv`.

- **Unit and target**: one row per artist and calendar year (`src/what_target.py`). `new_style_year` is 1 if the Works of the artist's next active year contain a style absent from every style seen through the end of the current year (styles of undated Works count as seen). Rows are kept when `history_count >= 3` (`config.yaml`, `what.min_previous_works`) and the label exists. Prevalence is 30.9% over 763,371 labelled development rows.
- **Kill gate**: PASS. The model is above baseline (c), the history and volume lookup, in 5 of 5 development folds, with 25 inputs.
- **Model comparison and tuning**: five models on the same folds and inputs (`notebooks/what_models.ipynb`); every model is above baseline (c) in 5 of 5 folds. Staged XGBoost tuning (69 log rows, stage 0 plus 68 trials, `results/trials/xgb_what_new_style_year.jsonl`) ended with stage 0 kept.
- **Holdout**: training rows have `year < 2018` and `next_active_year < 2018` (687,628 rows). Test rows are labelled rows with `2018 <= year <= 2023` and `next_active_year <= 2023`: 143,228 rows from 71,523 artists, effectively 2018 to 2022. Evaluated once. Stage 0: PR-AUC 0.499 [0.493, 0.504] and ROC-AUC 0.678 against a base rate of 0.313; baseline (c): PR-AUC 0.346 and ROC-AUC 0.559.

## Style trend forecast

Facts below come from [decisions.md](decisions.md) and `results/style_forecast/`.

- **Goal and series**: forecast the 10 styles with the highest growth in Works next year, with expected Works and a range. Series of dated, styled Works by style and year (`src/style_series.py`); counts are Works catalogued in Discogs, not sales or listens. Growth is the count of the target year over the mean of the three previous years minus 1; a style is eligible when that mean is at least 100.
- **Protocol**: expanding-window walk-forward, one target year at a time. Development target years are 1993 to 2017; the test years are 2018 to 2022, scored once. 2023 is used only as the base year of the 2024 projection.
- **Models and baselines**: BayesianRidge, XGBoost and Random Forest against random, repeat last year's ranking and 3-year mean growth. The Random Forest was tuned on the development years only (70 log rows in `results/trials/rf_style_forecast.jsonl`) and then locked.
- **Result**: on the development years the pass rule failed two of its five conditions, so the recommended ranking is repeat last year. On the test, top_10_styles mean is 6.00 [5.20, 7.00] for the locked Random Forest, 5.40 [4.00, 6.80] for repeat last year, 4.00 [2.80, 5.20] for mean growth and 0.42 for random (`results/style_forecast/test_20261015_173215.out`).
- **Projection**: 2024 from the base year 2023, illustrative, saved in `results/style_forecast/predictions.csv`.

## Next

- README and the Streamlit app are pending.
- Open items are tracked under PENDING in [decisions.md](decisions.md); known limits are under Limitations in the same file.

## Calendar

- 17-10-2026: build the Streamlit app and prepare the presentation.
- 27-10-2026: code freeze.
- 31-10-2026: presentation.
