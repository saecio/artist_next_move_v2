# Abandoned approaches

Methods that were tried and failed validation. Kept here so they are not retried by mistake. Adopted methods live in docs/decisions.md, not here.

## Filling missing year (works_df)

### 1. Chronological id-ordering fill (abandoned 08-10-2026)

- **What was tried:** fill a missing `year` from the neighbouring dated works of the same artist, assuming `master_id` order follows release order.
- **Validation gate:** on consecutive dated pairs within each artist, the share of non-decreasing years had to reach 90%. It reached 62.6%, so the fill was not built.
- **Why it failed:** `master_id` reflects when the entry was created in the catalog, not when the work was released. Back-catalog and reissues are entered long after the original, so id order does not track release chronology.
- **Source of figures:** the 62.6% result and the cause were taken from prior session output, not independently re-verified in this entry.
- **Do not retry:** any ordering-based fill, unless there is an independently reliable chronological key to order by. `master_id` and `release_id` are not that key.

### 2. MusicBrainz release-group fuzzy match (abandoned 08-10-2026)

- **What was tried:** fuzzy-match artist name (rapidfuzz ratio >= 90, after lowercasing, stripping punctuation and a leading "The") and then title within that artist block against `data/processed/musicbrainz_release_groups.parquet` (4,361,492 release groups, 95.6% of the dump had a usable year), and take `first-release-date` as the missing year. Refined pilot added a "/" split-title comparison and the number/roman-numeral guard from src/works.py.
- **Pilot:** 500 rows from works_missing_year.parquet (250 master, 250 synthetic). 34 of 500 matched at title score >= 90 (6.8%).
- **Spot-check of those 34** (catalog numbers, tracklists, and web sources): 29% strictly correct year (10 of 34), 59% if unverifiable "probably right" cases are counted (20 of 34). Same work in only 14 of 34. The 70-90 score band was 14% strict, 29% lenient (7 rows).
- **One-sided "/" split matches:** 0 of 14 were the same release (shared A-side, different B-side or coupling), and 1 of 14 had a verified correct year.
- **Why it failed:**
  - MusicBrainz `first-release-date` is often a reissue or CD-era date, not the original (for example a 1953 LP matched to its 1990 reissue CD).
  - 33 of the 34 matches scored exactly 100, so no score threshold separates correct from incorrect matches.
  - Same-name artist collisions (different acts sharing a name) pass the artist block.
  - Synthetic works are single pressings, which can be much later than the work-level year MusicBrainz holds.
  - Earliest-year tie-breaking among tied candidates was unreliable.
- **Do not retry:**
  - Do not trust `first-release-date` as ground truth without filtering out reissues.
  - Do not reuse the "/" split-title rule unless both sides of the title must match.
  - Do not use title score alone as an acceptance threshold.
- **Artifacts:** `data/processed/year_match_pilot_refined.parquet` (pilot output) and `data/processed/year_match_spotcheck.csv` (row-by-row verdicts and evidence).

### 3. MusicBrainz catalog-number join (abandoned 10-10-2026)

- **What was tried:** join works to `data/processed/musicbrainz_releases.parquet` (3,157,095 release rows from 5,832,004 MusicBrainz releases) on a normalized (label, catalog number) key, and take the matched release's own `date` year. Discogs side: the main release's labels for a master, the release itself for a synthetic work. Keys: `normalize_name(label)` and `normalize_catno(catno)`, both required, "NONE" and "N/A" catnos dropped.
- **Pilot:** 2,000 master and 2,000 synthetic works from `works_missing_year.parquet` (510,364 rows).
  - Usable match (one agreed MusicBrainz year): 20 of 2,000 master (1.0%), 11 of 2,000 synthetic (0.55%). Matched but MusicBrainz had no year: 2.1% and 1.1%. No conflicting years in either.
  - No usable label and catno on the Discogs side: 9.4% master, 18.4% synthetic. Keyed but no MusicBrainz match: 87.5% and 79.9%.
  - Dropping the label from the key (catno plus artist score >= 90) added only 2 master and 5 synthetic works with a year, so label-name mismatch is not the cause of the low yield.
- **Backtest on known-year works** (2,000 per source, same join, predicted year compared with the year Discogs already has): master 436 usable matches, 97.2% exact and 98.9% within 1 year; synthetic 161 usable matches, 98.8% exact and 99.4% within 1 year. Known-year works are the easier population (they match at 22% master and 8% synthetic), so this is an upper bound for the missing-year works.
- **Spot-check of the 31 matches on missing-year works:** judged offline from artist, title, label, catno and year agreement, not verified against the web or tracklists as in entry 2, so the numbers are approximate. Master: 15 of 20 probably right, 5 questionable (25%): one catno collision with a different artist, one conflicting with the earlier verified year (Johnny Ashcroft, matched 1968 vs about 1960-61), three likely reissue dates. Synthetic: 9 of 11 probably right, 2 uncertain (a title mismatch and a possible CD-reissue date).
- **Why it failed:** accurate when it matches, but immaterial. Scaled to the population the yield is about 614 master and 2,470 synthetic works, roughly 3,100 of 510,364 (0.6%), moving the missing-year rate from 8.58% to about 8.5%. The missing-year works are the obscure, undated releases that MusicBrainz does not hold. On the master side, the one population where a small yield would have mattered, about 25% of the matches that exist are questionable.
- **Do not retry:** any other MusicBrainz key (label, catno, artist, title) as a source of years for this population, since the limit is coverage, not the key. A different reference source with real coverage of obscure releases would be a new approach, not a retry.
- **Artifacts:** `data/processed/catno_pilot.parquet` (missing-year pilot) and `data/processed/catno_backtest.parquet` (known-year backtest). Code: `catno_join`, `summarize_catno_join` in `src/year_imputation.py`.

## Outcome: no imputation method adopted (10-10-2026)

Three methods tried (id ordering, MusicBrainz title fuzzy-match, MusicBrainz catalog-number join), all abandoned. Missing years stay as NaN: 510,364 of 5,949,217 works (8.58%), master and synthetic, are kept in the dataset and flagged with `year_missing`; handling is at the feature/snapshot level (see docs/decisions.md). A Master-only filter was tried as a workaround and reversed (see CUT in docs/decisions.md).
