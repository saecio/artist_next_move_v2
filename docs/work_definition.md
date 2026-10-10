# Work definition

Current-state reference for what a Work is and how the Works table is built. `docs/decisions.md` stays the chronological decision log (with the reasoning and counts behind each step), `docs/abandoned_approaches.md` holds the methods that failed validation. This file is the consolidated end-to-end explanation, so the history does not have to be reconstructed from the log.

Final table: `data/processed/works_checkpoint.parquet`, 5,949,217 rows (1,951,057 master, 3,998,160 synthetic), 1,034,756 artists. Built by `notebooks/01_cleaning_eda.ipynb`, with the reusable logic in `src/works.py`.

## 1. What a Work is

A Work is one row per artist-level creative release: the unit of analysis for both questions the project asks.

- WHEN: when will the artist release their next Work.
- WHAT: will the next Work introduce a style the artist has never used before.

A Work comes from one of two sources, kept in the `work_source` column:

- `master`: a Discogs Master (the entity that groups all Releases of the same album or single). 2,589,349 masters in the 2026-09-01 dump before cleaning.
- `synthetic`: a standalone Release with no Master link (`master_id == 0`, 41.6% of releases), promoted to a Work so artists who never got a Master are not lost. Only standalone releases from artists with 3+ standalone releases qualify. This recovers 388,610 artists that would disappear entirely under a Master-only definition.

Both sources go through the same exclusion and cleaning rules, then sit in one table.

## 2. How `work_id` is assigned

- Master: `work_id = master_id` (unique key, verified with 0 duplicates).
- Synthetic: `work_id = release_id + 10_000_000_000` (`SYNTHETIC_WORK_ID_OFFSET` in `src/works.py`, `SYNTHETIC_OFFSET` in notebook 01).

The offset guarantees a synthetic id can never collide with a real `master_id`, since release ids are far below 10 billion. It is also invertible (`work_id - offset = release_id`), which the catalog-number filter uses to look up a synthetic work's label and catalog number. `work_id` is verified unique across the combined table.

## 3. What is excluded and why

Applied to masters and standalone releases alike, before the two sources are combined.

- Placeholder artists: `Various`, `Unknown Artist`, `No Artist`, compared case-insensitively, plus the variants found in the final QA pass (`Unknown`, `N/A`, `VA`). Not a real person or group, so there is no trajectory to model.
- Out-of-scope genres: `Classical` and `Brass & Military` (`DEFAULT_EXCLUDED_GENRES`; mostly performances of old compositions or ensembles, not an artist's own evolving body of work), plus `Non-Music` and `Children's` (not relevant to musical style prediction).
- Placeholder/test record: `master_id` 376240 ("Dummy1").
- `data_quality == "Entirely Incorrect"` (139 masters). `"Needs Vote"` is kept.
- Compilations and unofficial releases, read from a representative release's `format_descriptions`.
- Missing-year sentinels: `year == 0` and `year > 2026` (the dump date, so a future year is a pre-announced release) are converted to missing, not excluded.

## 4. Cleaning filters, in pipeline order

Run on the combined table, in this order. Every row count is in `docs/decisions.md` under DECIDED.

1. Master year recovery (section 5), done first because the dedup tie-breaks below use year.
2. Exact duplicates: same `artist_id`, normalized title and year. Master preferred over synthetic, then smallest `work_id`.
3. Reissue/repress/remaster tagged works that share `artist_id` and normalized title with an earlier-year work by the same artist. The earliest survives.
4. Title patterns that mark a non-original work: Live, Greatest Hits, best of / anthology / box set / discography / b-sides / rarities, Remixes, Tribute, "(Acoustic)", "(Instrumental)", Unplugged, radio edit / radio version / BBC session. Each pattern was checked against a 20-row sample first. Patterns that were too noisy ("collection", "complete", "karaoke", "session", bare "acoustic") were cut.
5. Placeholder-artist QA pass (case-insensitive, catches variants the original exclusion missed).
6. Fuzzy-duplicate titles (`remove_fuzzy_duplicate_titles`): within an artist, alphabetical window of 10, rapidfuzz `token_sort_ratio >= 90`, guarded so pairs that differ only by a number or roman numeral (volume and episode series) are not merged.
7. Catalog-number duplicates (`remove_catalog_number_duplicates`): same `artist_id` and title across different years, collapsed when two entries share a real (label, catalog number) pair. Blank and placeholder catalog numbers never count as evidence.
8. Mass-upload artists (`remove_mass_upload_artists`): artists with 50+ works that are at least 90% synthetic, 50% bare roman-numeral titles and 80% inside one 2-year window, all three at once. Matches exactly one artist (6391827, 1,854 works).

`works_checkpoint.parquet` is written by exactly one cell, after step 8. Any new cleaning step goes before that cell. Intermediate stages have their own filenames (`works_checkpoint_raw`, `_dedup`, `_fuzzy_dedup`) so a partial re-run cannot silently regress the final file.

## 5. Year recovery

`recover_missing_master_years` (`src/works.py`) fills a Master's missing `year` from its own Releases. For each Master with no year, it parses every linked Release with the same rule used everywhere else (`_add_release_year`: first four characters of `released`, `0000` and years past 2026 treated as missing) and takes the earliest valid year. This is real data already in `releases.parquet`, not an inference.

Result: 38,670 of 101,543 master works missing a year (38.1%) were recovered. Recovery is uneven across genres: genres with the worst missing rate recover least (Spearman r=-0.809, p=0.0026, n=11), so the genre gap is not closed. Accepted as a known limitation.

No other imputation is applied. Three further methods (Master id ordering, MusicBrainz title fuzzy-match, MusicBrainz catalog-number join) were tried and abandoned, see `docs/abandoned_approaches.md`.

## 6. Works still without a year

After recovery, 510,364 of 5,949,217 works (8.58%) have `year = NaN`: 61,388 master (3.15%) and 448,976 synthetic (11.23%). They stay in the table. Dropping the 173,938 artists who have any undated work would remove 1,995,677 works (33.6%) and skew the data against high-missing genres (Latin, Reggae, Folk/World/& Country).

Flagging status:

- Artist level: `year_missing` is implemented in `src/eda.py` only (`artist_style_transitions`), True when the artist has at least one undated work. Used for the EDA checks, not stored on the Works table.
- Snapshot level: `build_snapshots` in `src/snapshots.py` reports `artist_undated_work_count` on every snapshot row and treats an artist's undated styles as already seen when computing `new_style`.
- Work level: a `year_missing` column on the Works table itself is still pending, to be added in the snapshots block.

Any feature or label that strictly needs a year excludes or special-cases those rows, not the whole work. The modeling plan is a model that splits on missingness natively (LightGBM or HistGradientBoostingClassifier), with evaluation also reported split by missingness.

## 7. Master-only, tried and reversed

On 2026-10-10 the Works table was filtered to `work_source == "master"` to avoid the missing-year gap. It cut 3,998,160 works (67.2%) and 403,897 of 1,034,756 artists (39.0%) to fix an 8.58% field-level gap, and removing most of an artist's discography destroys the sequences a next-work model learns from. Reversed the same day. Full reasoning and counts: `docs/decisions.md`, CUT, "Work definition is Master-only".
