# Artist Next Move

Predicts two things about an artist's next release, using only information available at a past date: WHEN it will happen and WHAT style it will bring. Models are always compared against simple historical baselines, and the demo ("Time Machine", Streamlit) hides the future, predicts, then reveals what actually happened.

Method notes live in `docs/`: `work_definition.md` (what a Work is and how the table is built), `decisions.md` (chronological decision log), `abandoned_approaches.md` (methods that failed validation), `pipeline_overview.md`.

## Data

No data is stored in this repository. The whole `data/` directory is gitignored. Everything is regenerated locally from public sources.

### Sources

- Discogs monthly data dump dated 2026-09-01 (`discogs_20260901_masters.xml`, `discogs_20260901_releases.xml`), from the public Discogs data dumps page.
- MusicBrainz JSON dump, snapshot 20261007-001001 (`release-group` and `release` entities), from the public MusicBrainz data dumps. Only used by the year-imputation experiments, which were abandoned (see `docs/abandoned_approaches.md`). The main pipeline does not need it.

Place the raw files in `data/raw/`. Run everything from the project root with the `venv` environment (`pip install -r requirements.txt`).

### Regenerating the processed files

| File in `data/processed/` | How it is produced |
|---|---|
| `masters.parquet` | `python -m src.parse_dumps masters data/raw/discogs_20260901_masters.xml data/processed/masters.parquet` |
| `releases.parquet` | `python -m src.parse_dumps releases data/raw/discogs_20260901_releases.xml data/processed/releases.parquet` |
| `masters_clean_checkpoint.parquet`, `works_checkpoint_raw.parquet`, `works_checkpoint_dedup.parquet`, `works_checkpoint_fuzzy_dedup.parquet` | Intermediate stages of `notebooks/01_cleaning_eda.ipynb` (logic in `src/works.py`) |
| `works_checkpoint.parquet` | Final Works table, written by the last cell of `notebooks/01_cleaning_eda.ipynb`. Read by `notebooks/02_eda_visualization.ipynb` and `src/snapshots.py` |
| `musicbrainz_release_groups.parquet` | `scripts/run_release_group_parse.py` (reads the `release-group` dump file placed in `data/raw/`) |
| `musicbrainz_releases.parquet` | `scripts/run_release_parse.py` (reads the MusicBrainz `release` dump from stdin, usage in the script docstring) |
| `works_missing_year.parquet`, `catno_pilot.parquet`, `catno_backtest.parquet`, `year_match_pilot*.parquet` | Year-imputation experiments, functions in `src/year_imputation.py` |
| `*_sample.parquet` | Small samples written by `notebooks/01_cleaning_eda.ipynb` for inspection |

Order: parse the two Discogs dumps, then run notebook 01 top to bottom. The imputation files are optional.
