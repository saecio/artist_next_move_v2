# Artist Next Move

**Live demo:** https://artist-next-move.streamlit.app

Predicts two things about an artist's next release, using only information that was available at a past date:

- **Timing** (called WHEN in the code, notebooks and docs): will the artist publish a new work in the next year, and in the next two?
- **Style** (called WHAT in the code, notebooks and docs): will the next release bring a style that is new to that artist? New to the artist, not to the world.

Every model is compared with simple historical baselines, and every score comes from out-of-time evaluation: models are trained on the past and tested on years they never saw. The Streamlit demo, the "Time Machine", hides the future, predicts, then reveals what actually happened.

Data: Discogs, 5,949,217 works by 1,034,756 artists. Final test on a sealed holdout (2018 to 2023), evaluated once. Timing was re-scored once more only to add 95% intervals, with identical point estimates.

| Question | Our model | Release counts | By chance |
|---|---|---|---|
| Timing, next year (PR-AUC) | 0.785 | 0.735 | 0.542 |
| Style (PR-AUC) | 0.499 | 0.346 | 0.313 |

Timing over the next two years: PR-AUC 0.835 for the model, 0.789 for release counts and 0.643 by chance.

Among the artists the model rates highest (top tenth), 90 out of 100 publish the next year (54 by chance) and 60 out of 100 bring a style new to them (31 by chance). The figures are in `results/holdout/`, `results/holdout_ci95/`, `results/holdout_what/` and `results/calibration/`. PR-AUC is not an accuracy: do not read it as "right x% of the time".

## Run the demo

The demo needs no raw data. Everything it reads is committed in `results/` and `app/`.

Requirements: Python 3.12 or newer (developed on 3.14).

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Run it from the project root: the theme and fonts in `.streamlit/config.toml` use paths relative to it. The app opens at http://localhost:8501. On Linux, installing XGBoost also downloads a large optional GPU library (about 300 MB) that the project does not use.

The app has eight tabs:

1. Problem and question
2. Data and cleaning
3. Data exploration
4. Method
5. Results
6. Time Machine demo (pick an artist, hide the future, predict, then reveal)
7. Global style forecast (the ten styles expected to grow most)
8. Conclusions

## Terms used in the code and docs

| In the app | In the code and docs | Meaning |
|---|---|---|
| Timing | WHEN, `next_1y`, `next_2y` | A new work within one or two years |
| Style | WHAT, `new_style` | The next release has a style the artist has not used before |
| Global style forecast | Style trend forecast | Which styles will have more works next year, worldwide |
| Release counts | Baselines | Simple rules built from the artist's own release counts |
| Release counts (Style and Timing rules) | Baseline (c) and (b) in notebooks 04 and 05; WHEN baseline in notebook 03 | Style rule is (c): train positive rate of the artist's bucket of earlier releases and bucket of releases this year ((b) uses the earlier-releases bucket only and is not shown in the app). Timing rule is the WHEN baseline: releases per year of activity so far (`historical_rate`) |
| A release | A Work, a work | One row per artist-level creative release: an album, EP or single, not a Discogs release record (see `docs/work_definition.md`) |

## Project layout

| Folder or file | Contents |
|---|---|
| `app/` | Streamlit app (`streamlit_app.py`), demo settings (`demo_config.yaml`), fonts and artist photos with their credits |
| `src/` | All the logic: parsing, cleaning, snapshots, baselines, models, tuning, evaluation and the app's data layers |
| `notebooks/` | Six notebooks that call functions from `src/` and record each question, result, interpretation and decision |
| `scripts/` | One script per reproducible step (tuning runs, final evaluations, audits, demo assets) and the app checks |
| `results/` | Committed outputs: scores, tuning logs, stored models, demo assets |
| `docs/` | Method notes and the decision log (see below) |
| `config.yaml` | Paths, years, the random seed (1) and shared settings |
| `data/` | Not versioned. Raw dumps and processed tables (see below) |

Notebooks:

| Notebook | Topic |
|---|---|
| `01_cleaning_eda.ipynb` | Cleaning the Discogs dumps into the Works table |
| `02_eda_visualization.ipynb` | Exploration and hypothesis tests |
| `03_when_backtesting_design.ipynb` | Timing: snapshots, backtesting design, models, tuning record |
| `04_what_design.ipynb` | Style: target design, population, label rules, baselines and the kill gate |
| `05_what_models.ipynb` | Style: comparison of five models |
| `06_style_forecast.ipynb` | Global style forecast |

## Method in short

- Evaluation is always out of time. There is never a random split.
- Each prediction uses a snapshot: only what was known at that date.
- Baselines come before machine learning. A model passes only if the lower bound of its 95% artist-level bootstrap interval is above the baseline's upper bound.
- The final holdout (snapshot years 2018 to 2023) is evaluated once, after all choices are fixed (Timing was re-scored once more only for the 95% intervals).
- Years 2024 and 2025 are excluded from training, tuning and metrics because they are incomplete.
- Models compared: Logistic Regression, Naive Bayes, Random Forest, HistGradientBoosting and XGBoost. XGBoost is tuned in stages (random search, grid, manual) on the development years only; the others keep their default settings. For the Global style forecast, the Random Forest is the one tuned. It does not pass the pass rule against repeating last year's ranking (6.28 against 6.00 hits in the top 10, a margin of 0.28 where 0.5 is required), so the recommended ranking is repeating last year's list (notebook 06).

## Reproduce everything from the raw data

You only need this to rebuild the tables and results from scratch. All results used by the demo are already committed.

### Data sources

Place the raw files in `data/raw/`.

- Discogs monthly data dump dated 01-09-2026 (`discogs_20260901_masters.xml` and `discogs_20260901_releases.xml`), from https://data.discogs.com/. The site offers them compressed (`.xml.gz`): decompress them first.
- MusicBrainz JSON dump, snapshot 20261007-001001 (`release-group` and `release` entities), from https://musicbrainz.org/doc/MusicBrainz_Database/Download. It is only used by the year-imputation experiments, which were abandoned (see `docs/abandoned_approaches.md`). The main pipeline does not need it.

Licences: the Discogs data dumps are released under CC0 ("No Rights Reserved"), with no attribution requirement. The MusicBrainz core data is also CC0, while its remaining supplementary data is licensed CC BY-NC-SA 3.0, which does not allow commercial use. The main pipeline and the demo do not use MusicBrainz.

### Disk space

| Item | Size |
|---|---|
| Discogs releases XML (decompressed) | about 62 GB |
| Discogs masters XML (decompressed) | about 3 GB |
| MusicBrainz dumps (optional) | about 19 GB |
| `data/processed/` | about 6.5 GB (the releases table is 4.6 GB of it) |
| `results/` | about 90 MB |

Without MusicBrainz, plan for about 72 GB of free disk. Parsing the 62 GB releases file is the longest step: run it once. Building the snapshots table from the Works table takes about 12 seconds.

### Processed files

Run everything from the project root with the `venv` environment active.

| File in `data/processed/` | How it is produced |
|---|---|
| `masters.parquet` | `python -m src.parse_dumps masters data/raw/discogs_20260901_masters.xml data/processed/masters.parquet` |
| `releases.parquet` | `python -m src.parse_dumps releases data/raw/discogs_20260901_releases.xml data/processed/releases.parquet` |
| `masters_clean_checkpoint.parquet`, `works_checkpoint_raw.parquet`, `works_checkpoint_dedup.parquet`, `works_checkpoint_fuzzy_dedup.parquet` | Intermediate stages of `notebooks/01_cleaning_eda.ipynb` (logic in `src/works.py`) |
| `works_checkpoint.parquet` | Final Works table, written by the last cell of `notebooks/01_cleaning_eda.ipynb`. Read by the other notebooks and by `src/snapshots.py` |
| `snapshots.parquet` | Cache of the snapshots, built the first time `notebooks/03_when_backtesting_design.ipynb` runs (`load_or_build_snapshots` in `src/backtesting.py`) |
| `musicbrainz_release_groups.parquet` | `scripts/run_release_group_parse.py` (reads the `release-group` dump file placed in `data/raw/`) |
| `musicbrainz_releases.parquet` | `scripts/run_release_parse.py` (reads the MusicBrainz `release` dump from stdin, usage in the script docstring) |
| `works_missing_year.parquet`, `catno_pilot.parquet`, `catno_backtest.parquet`, `year_match_pilot*.parquet` | Year-imputation experiments, functions in `src/year_imputation.py` |
| `*_sample.parquet` | Small samples written by `notebooks/01_cleaning_eda.ipynb` for inspection |

### Order of the pipeline

1. Parse the two Discogs dumps (first two rows of the table above).
2. Run `notebooks/01_cleaning_eda.ipynb` top to bottom. The MusicBrainz and year-imputation files are optional.
3. Run the notebooks `02` to `06` in order. Notebook `03` creates `snapshots.parquet`.
4. Run the scripts below for the tuning runs, final evaluations and demo assets.

Scripts are run from the project root as `python scripts/<name>.py`. Many of them create their output with exclusive creation and stop if the file already exists, so a result is never overwritten by accident. To rebuild one, move the old file aside first. The final holdout evaluations are one-time by design.

**Timing (WHEN)**

| Step | Scripts |
|---|---|
| XGBoost tuning, phases 1 to 3 | `run_when_xgb_random_search`, `run_when_xgb_grid_search`, `run_when_xgb_manual` |
| Overfit check | `run_when_xgb_overfit_check` |
| Final configuration on the development folds | `run_when_xgb_final`, `run_when_xgb_final_ci95` |
| Sealed holdout (once) | `run_when_holdout`, `run_when_holdout_ci95` |
| Calibration of the final models | `run_calibration_check` |

**Style (WHAT)**

| Step | Scripts |
|---|---|
| XGBoost tuning, phases 1 to 3 | `run_xgb_what_random_search`, `run_xgb_what_grid_search`, `run_xgb_what_manual` |
| Adoption and overfit checks | `run_xgb_what_compare`, `run_xgb_what_overfit_check` |
| Sealed holdout (once) | `run_xgb_what_holdout` |

**Global style forecast (tab 7)**

| Step | Scripts |
|---|---|
| Development years, baselines and models | `run_style_forecast_dev`, `run_style_forecast_spearman` |
| Random Forest tuning, phases 1 to 4 | `run_rf_style_random_search`, `run_rf_style_grid_search`, `run_rf_style_manual`, `run_rf_style_adoption` |
| Final test years and the projection (once) | `run_style_forecast_test` |

**Audits and sensitivity checks**

| Check | Script |
|---|---|
| Label look-ahead in the last development fold | `run_label_lookahead_check`, `run_label_lookahead_ranking_check` |
| Duplicate sensitivity of the Works table | `run_duplicate_sensitivity` |
| Dependence of the Style label on undated works | `run_what_undated_label_check` |
| Style forecast design checks and duplicate audit | `run_style_design_checks`, `run_style_duplicate_audit`, `run_style_duplicate_refit` |

**Demo assets (`results/demo/`)**

Run `run_when_demo_scores`, `run_works_demo` and `run_works_demo_b` first, because the later steps read their files.

| Asset | Scripts |
|---|---|
| Live scoring assets for the two artist pools | `run_live_assets`, `run_live_pool_b_when`, `run_live_pool_b_what`, `run_live_what_extended` |
| Live scoring assets of the style forecast | `run_live_style_assets`, `run_live_style_extra` |
| Artist-years without a dated work | `run_when_year_end`, `run_what_year_end` |
| Reality cells of tab 6 | `run_what_new_styles`, `run_release_pick`, `run_demo_route_table` |
| Static tabs | `run_static_assets`, `run_models_compared_assets` |
| Demo walk-through (prints only) | `run_game_sim` |

`run_xgb_what_live` prints a score for a fixed list of well-known artists and writes nothing.

## Check the app

Six scripts run the app with `streamlit.testing.v1.AppTest` and print their results (they write no files). Run them from the project root with the environment active:

```
python scripts/run_static_apptest.py
python scripts/run_multi_apptest.py
python scripts/run_style_apptest.py
python scripts/run_game_sim.py
python scripts/run_app_live_test.py
python scripts/run_game_apptest.py
```

Together they take about 8 minutes. Only `run_static_apptest.py` needs the processed data: it also compares figures with `data/processed/works_checkpoint.parquet`. The other five run from the committed files alone. The style check sometimes times out at 60 seconds: run it again.

## Additional checks

These checks are not shown in the notebooks. Each script refuses to overwrite an existing output and reads development data only, except the calibration check, which reads the already spent holdout.

- Label look-ahead: counts the fold 5 test rows whose label depends on years of the holdout period (2018 onward) and measures whether dropping them changes the fold 5 PR-AUC. Script `scripts/run_label_lookahead_check.py` (and `scripts/run_label_lookahead_ranking_check.py`), output in `results/label_lookahead/`.
- Artist-disjoint: refits the models on training rows from artists that do not appear in the test fold and scores them on the development folds, so the score no longer benefits from having seen the artist. Script `scripts/run_artist_disjoint_check.py`, output in `results/artist_disjoint/artist_disjoint.csv` and `results/artist_disjoint.out`.
- Paired confidence interval: gives the interval of the PR-AUC difference itself for three comparisons that were judged by overlapping or barely separated intervals. Script `scripts/run_paired_ci_check.py`, output in `results/paired_ci/paired_differences.csv`.
- Calibration: read-only calibration check of the final WHAT and WHEN models on the final holdout, with no refit and no recalibration. Script `scripts/run_calibration_check.py`, output in `results/calibration/`.

## Documentation

| File | Contents |
|---|---|
| `docs/work_definition.md` | What a Work is and how the Works table is built |
| `docs/pipeline_overview.md` | Short status of the pipeline |
| `docs/abandoned_approaches.md` | Methods that failed validation |
| `docs/decisions.md` | Index of the decision log |
| `docs/decisions_core.md`, `docs/decisions_demo.md` | Decisions with the reasoning and numbers behind each one |
| `docs/decisions_pending.md`, `docs/decisions_archive.md` | Open items, and decisions that were reversed |

The decision files are long. Find an entry by its exact title with `grep -n` and read only that range.

## Limits

Counts are works catalogued in Discogs, not sales or listens. Some works have no year. The Style question can only be answered for artists with tagged styles. The full list is in tab 8 of the app and in the Limitations section of `docs/decisions_core.md`.

## Credits and licences

- Data: Discogs monthly data dumps (CC0). See the licence note under "Data sources".
- Artist photos in `app/assets/artists/`: the Taylor Swift photo is by Ronald Woan, CC BY-SA 2.0 (resized), from Wikimedia Commons; the Bad Bunny photo is the author's own work. Details in `app/assets/artists/credits.csv`.
- Fonts in `app/static/fonts/`: Inter and Fraunces, under the SIL Open Font License 1.1 (the licence files are next to the fonts).
