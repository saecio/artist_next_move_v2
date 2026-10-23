"""Diagnostic helpers to test whether master_id order tracks release year."""

import datetime  # current year as the upper bound for valid release years
import json  # parse one NDJSON line at a time
import re  # anchored 4-digit year extraction
import sys  # stdin as an input source for streamed dumps
import time  # elapsed time in the progress log
import unicodedata  # character categories to strip punctuation without losing non-Latin letters
from functools import lru_cache  # each distinct name or title is normalized only once
from collections import Counter  # skip-reason tallies

import numpy as np  # array helpers for the year == 0 to NaN conversion
import pandas as pd  # tabular loading and manipulation
import pyarrow.dataset as ds  # filtered read of the large Discogs releases table
from rapidfuzz import fuzz, process  # fuzzy scoring and candidate search

from config import CONFIG, ROOT  # project root and settings from config.yaml
from works import SYNTHETIC_WORK_ID_OFFSET, differs_only_by_number_or_roman  # work_id offset and number guard already used in works.py

PROCESSED = ROOT / CONFIG["paths"]["processed"]  # paths.processed, folder of the parquet files

# Placeholder credits already excluded from the Works dataset (see docs/decisions.md)
NON_ARTIST_NAMES = {"Various", "Unknown Artist", "No Artist"}

# Earliest year accepted from MusicBrainz, anything older is treated as a placeholder date
MB_MIN_YEAR = 1860
# A year at the start of the date string: matches "1994", "1994-05" and "1994-05-12"
MB_YEAR_RE = re.compile(r"^\d{4}")


def load_masters_for_check(path: str = str(PROCESSED / "masters.parquet")) -> pd.DataFrame:
    """Load raw masters with one row per master and its first credited artist."""
    # Read only the columns this check needs, the file has 2.5M rows
    df = pd.read_parquet(path, columns=["master_id", "artist_ids", "artist_names", "title", "year"])
    # Drop masters with no credited artist, there is nothing to group them by
    df = df[df["artist_ids"].str.len() > 0].copy()
    # Simplification: keep only the first credited artist (collabs are not exploded)
    df["artist_id"] = df["artist_ids"].str[0].astype("int64")
    # Same first-artist rule for the name, used only to drop placeholder credits
    df["artist_name"] = df["artist_names"].str[0]
    # Remove placeholder credits (Various, Unknown Artist, No Artist)
    df = df[~df["artist_name"].isin(NON_ARTIST_NAMES)]
    # year == 0 means missing in the dump, convert to NaN as logged in decisions.md
    df["year"] = df["year"].astype("float64").replace(0, np.nan)
    # Keep a lean frame and a clean index for the downstream steps
    return df[["master_id", "artist_id", "artist_name", "title", "year"]].reset_index(drop=True)


def consecutive_pair_stats(df: pd.DataFrame, min_dated: int = 5) -> tuple[float, pd.DataFrame]:
    """Share of consecutive same-artist pairs whose year does not decrease as master_id grows."""
    # Only masters with a known year can be compared
    dated = df.dropna(subset=["year"])
    # Count dated masters per artist to apply the minimum history rule
    n_dated = dated.groupby("artist_id")["master_id"].transform("size")
    # Keep artists with enough dated masters and order them by id inside each artist
    dated = dated[n_dated >= min_dated].assign(n_dated=n_dated).sort_values(["artist_id", "master_id"])
    # Year change versus the previous master of the same artist (first row of each artist is NaN)
    dated["year_diff"] = dated.groupby("artist_id")["year"].diff()
    # Each remaining row is one consecutive pair
    pairs = dated.dropna(subset=["year_diff"]).copy()
    # A pair is fine when the year stays equal or goes up
    pairs["non_decreasing"] = pairs["year_diff"] >= 0
    # Bucket pairs by the size of the artist's dated history
    pairs["size_bucket"] = pd.cut(
        pairs["n_dated"], bins=[min_dated - 1, 10, 50, np.inf], labels=[f"{min_dated}-10", "11-50", "51+"]
    )
    # Breakdown per bucket: artists, pairs and share of non-decreasing pairs
    by_size = pairs.groupby("size_bucket", observed=True).agg(
        n_artists=("artist_id", "nunique"), n_pairs=("non_decreasing", "size"), share=("non_decreasing", "mean")
    )
    # Global share over all pairs plus the breakdown
    return pairs["non_decreasing"].mean(), by_size


def parse_release_groups(
    path: str = "../data/raw/release-group",
    out_path: str = str(PROCESSED / "musicbrainz_release_groups.parquet"),
    log_every: int = 500_000,
) -> tuple[pd.DataFrame, Counter]:
    """Stream the MusicBrainz release-group NDJSON into a clean title, artist, year table."""
    # Rows kept and the reason of every skipped entity
    rows, skipped = [], Counter()
    # Latest valid year is the current one
    max_year = datetime.date.today().year
    # Start time for the progress log
    start = time.time()
    # Read line by line so the 18GB dump never sits in memory
    with open(path, encoding="utf-8") as f:
        for line in f:  # one JSON entity per line
            # Count every line so kept + skipped always adds up to the total
            skipped["total"] += 1
            # Progress line every log_every lines: lines read, rows kept, rows skipped, elapsed minutes
            if skipped["total"] % log_every == 0:
                print(
                    f"{skipped['total']:,} lines | kept {len(rows):,} | skipped {skipped['total'] - len(rows):,}"
                    f" | {(time.time() - start) / 60:.1f} min",
                    flush=True,
                )
            # A truncated or corrupt line is skipped, not fatal
            try:
                entity = json.loads(line)  # parse the line into a dict
            except json.JSONDecodeError:  # the line is not valid JSON
                skipped["bad_json"] += 1  # count it under its own reason
                continue  # go to the next line
            # No release date at all (None or empty string)
            date = entity.get("first-release-date")
            if not date:  # skip entities without a date
                skipped["null_or_empty_date"] += 1  # count the missing date
                continue  # go to the next line
            # The date must start with 4 digits, anything else is not guessed
            match = MB_YEAR_RE.match(date)
            if match is None:  # the date does not start with a year
                skipped["regex_no_match"] += 1  # count the unreadable date
                continue  # go to the next line
            # Convert only after the regex passed, then check the sane range
            year = int(match.group())
            if not MB_MIN_YEAR <= year <= max_year:  # the year is outside the plausible range
                skipped["out_of_range_year"] += 1  # count the implausible year
                continue  # go to the next line
            # First credited artist only, same simplification as the Discogs side
            credit = entity.get("artist-credit") or []
            title = entity.get("title")  # release group title
            if not credit or not credit[0].get("name") or not title:  # a missing artist or title makes the row unusable
                skipped["no_artist_or_title"] += 1  # count the incomplete entity
                continue  # go to the next line
            # Keep the row in the three-column reference layout
            rows.append((title, credit[0]["name"], year))
    # Build the table with the agreed column names and a compact year type
    df = pd.DataFrame(rows, columns=["mb_title", "mb_artist_name", "mb_year"]).astype({"mb_year": "int16"})
    # Persist the reference table
    df.to_parquet(out_path, index=False)
    # Kept rows plus the skip counts, so the usable share of the dump is visible
    skipped["kept"] = len(df)
    return df, skipped  # table plus the skip counts


def build_missing_year_subset(
    path: str = str(PROCESSED / "works_checkpoint.parquet"),
    out_path: str = str(PROCESSED / "works_missing_year.parquet"),
) -> tuple[pd.DataFrame, int]:
    """Works that still have no year after cleaning, with the columns needed for matching."""
    # Final cleaned works table, read only the columns the matching needs
    works = pd.read_parquet(path, columns=["work_id", "work_source", "artist_id", "artist", "title", "year"])
    # Keep rows whose year is still NaN after recover_missing_master_years and all cleaning filters
    missing = works[works["year"].isna()].reset_index(drop=True)
    # Persist the subset for the fuzzy-match pilot
    missing.to_parquet(out_path, index=False)
    # Subset plus the size of the full table, so the share can be reported
    return missing, len(works)


@lru_cache(maxsize=None)
def normalize_name(text: str) -> str:
    """Lowercase, drop a Discogs '(2)' suffix and a leading 'the', strip punctuation, collapse spaces."""
    # Lowercase and trim the edges
    text = text.lower().strip()
    # Discogs adds '(2)', '(3)'... to tell apart artists with the same name
    text = re.sub(r"\s\(\d+\)$", "", text)
    # Leading article, so 'The Cure' and 'Cure' compare equal
    text = re.sub(r"^the\s+", "", text)
    # Keep letters, numbers, marks and separators of any script, drop punctuation and symbols
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] in "LNMZ")
    # Collapse repeated spaces left by the removals
    return re.sub(r"\s+", " ", text).strip()


def sample_by_source(missing: pd.DataFrame, n_per_source: int = 250, seed: int = 42) -> pd.DataFrame:
    """Same number of rows from each work_source, so match quality can be judged separately."""
    # Even split per source, fixed seed so the pilot is reproducible
    return missing.groupby("work_source").sample(n=n_per_source, random_state=seed).reset_index(drop=True)


def title_variants(raw_title: str, use_split: bool) -> list[str]:
    """Normalized title plus, when it holds a '/', each side on its own (Discogs multi-track singles)."""
    # Full title always compared
    variants = [normalize_name(raw_title)]
    # Each side of a '/' is also a valid way to name the same release
    if use_split and "/" in raw_title:
        variants += [normalize_name(part) for part in raw_title.split("/")]  # also add each side of the slash
    # Drop empty and repeated variants
    return list(dict.fromkeys(v for v in variants if v))


def differs_by_number_marker(title_a: str, title_b: str) -> bool:
    """Reuse the works.py number/roman guard, treating 'volume' and 'vol' as the same word."""
    # works.py tokenizes 'Vol.' and 'Volume' as different words (documented limitation), unify them first
    title_a, title_b = (re.sub(r"\bvolume\b", "vol", t) for t in (title_a, title_b))
    # True when the wording is identical once numbers and roman numerals are stripped
    return differs_only_by_number_or_roman(title_a, title_b)


def fuzzy_match_pilot(
    sample: pd.DataFrame, mb: pd.DataFrame, artist_cutoff: int = 90, use_split: bool = True, use_guard: bool = True
) -> pd.DataFrame:
    """Best MusicBrainz match per row: artist block (score >= cutoff), then best title inside the block."""
    # Normalize every MusicBrainz artist once and group row positions by normalized name
    mb_artist_norm = mb["mb_artist_name"].map(normalize_name)
    block_rows = mb_artist_norm.groupby(mb_artist_norm, sort=False).indices  # row positions of each distinct normalized artist
    # Distinct normalized artists to search against
    artist_pool = list(block_rows)
    # Plain arrays make the per-row lookups fast
    mb_title = mb["mb_title"].to_numpy()
    mb_artist = mb["mb_artist_name"].to_numpy()  # artist names as a plain array
    mb_year = mb["mb_year"].to_numpy()  # release years as a plain array
    out = []  # one result record per sampled row
    for row in sample.itertuples():  # go through the sampled works one by one
        # Result template: no match unless a block and a title are found
        rec = {"work_id": row.work_id, "work_source": row.work_source, "artist": row.artist, "title": row.title,
               "mb_artist": None, "artist_score": np.nan, "mb_title": None, "title_score": np.nan,
               "mb_year": np.nan, "n_tied_years": 0, "n_block": 0}
        artist_norm, own_titles = normalize_name(row.artist), title_variants(row.title, use_split)  # normalized artist and title variants of this work
        # Names that normalize to nothing cannot be matched
        if artist_norm and own_titles:
            # Block: every MusicBrainz artist whose name scores >= cutoff against this one
            hits = process.extract(artist_norm, artist_pool, scorer=fuzz.ratio, score_cutoff=artist_cutoff, limit=None)
            if hits:  # continue only when some artist is close enough
                # Row positions of the whole block and the artist score of each row
                parts = [block_rows[name] for name, _, _ in hits]
                pos = np.concatenate(parts)  # all row positions joined into one array
                a_sc = np.repeat([score for _, score, _ in hits], [len(part) for part in parts])  # artist score repeated for each row of its block
                # Every candidate title expands to its variants, keep where each candidate's variants start
                cand_titles = [title_variants(t, use_split) for t in mb_title[pos]]  # title variants of every candidate row
                starts = np.cumsum([0] + [len(v) for v in cand_titles])[:-1]  # index where each candidate's variants begin
                flat = [v for variants in cand_titles for v in variants]  # all candidate variants in one flat list
                owner = np.repeat(np.arange(len(pos)), [len(v) for v in cand_titles])  # which candidate each variant belongs to
                # Score matrix: own title variants (rows) against every candidate variant (columns)
                sc = process.cdist(own_titles, flat, scorer=fuzz.ratio)
                if use_guard:  # the number guard can be switched off
                    # Number guard on every pair that scores high enough to matter, rejected pairs score 0
                    for i, j in zip(*np.nonzero(sc >= 70)):
                        if differs_by_number_marker(own_titles[i], flat[j]):  # same title apart from a number marker
                            sc[i, j] = 0  # reject the pair by zeroing its score
                # Best variant pair per candidate (candidates with no variants stay at 0)
                t_sc = np.zeros(len(pos))
                has_variants = np.diff(np.append(starts, len(flat))) > 0  # candidates that have at least one variant
                t_sc[has_variants] = np.maximum.reduceat(sc.max(axis=0), starts[has_variants])  # keep the best variant score per candidate
                # Best title first, then best artist score, then earliest year
                best = np.lexsort((mb_year[pos], -a_sc, -t_sc))[0]
                rec.update(mb_artist=mb_artist[pos[best]], artist_score=a_sc[best], mb_title=mb_title[pos[best]],  # fill the template with the winning candidate
                           title_score=t_sc[best], mb_year=mb_year[pos[best]], n_block=len(pos),
                           # Distinct years among rows tied at the best title score (1 = unambiguous)
                           n_tied_years=np.unique(mb_year[pos][t_sc == t_sc.max()]).size)
        out.append(rec)  # store the record for this work
    return pd.DataFrame(out)  # one row per sampled work


def normalize_catno(text: str) -> str:
    """Uppercase and keep only letters and numbers, so 'ABC-123', 'abc 123' and 'ABC123' share one key."""
    # Same character filter as normalize_name (any script), without lowercasing or collapsing spaces
    return "".join(ch for ch in text.upper() if unicodedata.category(ch)[0] in "LN")


def parse_releases(
    path: str = "../data/raw/release.ndjson",
    out_path: str = str(PROCESSED / "musicbrainz_releases.parquet"),
    log_every: int = 100_000,
) -> tuple[pd.DataFrame, Counter]:
    """Stream the MusicBrainz release NDJSON into one row per (release, label, catalog number)."""
    # Rows kept and the counts of every outcome
    rows, stats = [], Counter()
    # Latest valid year is the current one
    max_year = datetime.date.today().year
    # Start time for the progress log
    start = time.time()
    # "-" reads the stream from stdin, so the compressed dump never has to be unpacked to disk
    with (sys.stdin if path == "-" else open(path, encoding="utf-8")) as f:
        for line in f:  # one release per line
            # Count every release line so the outcomes always add up to the total
            stats["releases"] += 1
            # Progress line: releases read, rows kept, elapsed minutes
            if stats["releases"] % log_every == 0:
                print(f"{stats['releases']:,} releases | rows {len(rows):,} | {(time.time() - start) / 60:.1f} min", flush=True)
            # A truncated or corrupt line is skipped, not fatal
            try:
                rel = json.loads(line)  # parse the line into a dict
            except json.JSONDecodeError:  # the line is not valid JSON
                stats["bad_json"] += 1  # count the bad line
                continue  # go to the next line
            # Year of the release's own date (not the release group's first-release-date); NaN when missing or invalid
            year, date = np.nan, rel.get("date")
            if date:  # the release has its own date
                match = MB_YEAR_RE.match(date)  # check that the date starts with a year
                if match is None:  # count the unreadable date
                    stats["date_regex_no_match"] += 1  # the year is plausible, so keep it
                elif MB_MIN_YEAR <= int(match.group()) <= max_year:  # keep the year as a number
                    year = int(match.group())  # the year is outside the plausible range
                else:  # count the implausible year
                    stats["date_out_of_range"] += 1  # the release has no date
            else:  # count the missing date
                stats["no_own_date"] += 1
            # First credited artist only, same simplification as the Discogs side
            credit = rel.get("artist-credit") or []
            artist = credit[0].get("name") if credit else None  # artist name of the first credit, if any
            # One row per label-info entry that carries a catalog number
            kept = 0
            for info in rel.get("label-info") or []:  # go through the label entries of the release
                catno = (info.get("catalog-number") or "").strip()  # catalog number as text without edge spaces
                if not normalize_catno(catno):  # skip entries with no usable catalog number
                    continue  # go to the next entry
                label = ((info.get("label") or {}).get("name") or "").strip()  # label name as text without edge spaces
                rows.append((rel["id"], (rel.get("release-group") or {}).get("id"), rel.get("title"), artist,  # add one row for this label and catalog number
                             label, catno, date or None, year, rel.get("country"), rel.get("status")))
                kept += 1  # count the row kept for this release
            # Releases with no usable catalog number cannot be joined
            stats["with_catno" if kept else "no_catno"] += 1
            # Of the joinable releases, split by whether the release has a usable own year
            if kept:
                stats["catno_and_year" if not np.isnan(year) else "catno_no_year"] += 1
    # Build the table, normalized join keys included
    df = pd.DataFrame(rows, columns=["mb_release_id", "mb_rg_id", "mb_title", "mb_artist_name", "mb_label",
                                     "mb_catno", "mb_date", "mb_year", "mb_country", "mb_status"])
    df["mb_catno_key"] = df["mb_catno"].map(normalize_catno)  # key to join on catalog number
    df["mb_label_key"] = df["mb_label"].map(normalize_name)  # key to join on label name
    # Persist the reference table
    df.to_parquet(out_path, index=False)
    stats["rows"] = len(df)  # total rows kept
    return df, stats  # table plus the outcome counts


# Catalog numbers that mean "nothing entered" once normalized (NONE, N/A)
EMPTY_CATNO_KEYS = {"", "NONE", "NA"}


def discogs_label_catno(works: pd.DataFrame, releases_path: str = str(PROCESSED / "releases.parquet")) -> pd.DataFrame:
    """One row per (work, label, catalog number) from the Discogs release that represents each work."""
    # Synthetic works are one release (work_id minus the offset), masters are looked up by master_id
    is_master = works["work_source"] == "master"
    master_ids = works.loc[is_master, "work_id"].tolist()  # ids of the works that come from a master
    release_ids = (works.loc[~is_master, "work_id"] - SYNTHETIC_WORK_ID_OFFSET).tolist()  # release ids of the works that are single releases
    # One filtered pass over the releases table, only the columns needed
    flt = ds.field("master_id").isin(master_ids) | ds.field("release_id").isin(release_ids)
    rel = ds.dataset(releases_path).to_table(columns=["release_id", "master_id", "is_main_release", "labels"], filter=flt).to_pandas()  # load only the matching releases into pandas
    # Master side: the main release represents the master, lowest release_id when none is flagged
    m = rel[rel["master_id"].isin(master_ids)].sort_values(["master_id", "is_main_release", "release_id"], ascending=[True, False, True])
    m = m.drop_duplicates("master_id").assign(work_id=lambda d: d["master_id"])  # one release per master, named by the work id
    # Synthetic side: the release itself, work_id rebuilt with the offset
    r = rel[rel["release_id"].isin(release_ids)].assign(work_id=lambda d: d["release_id"] + SYNTHETIC_WORK_ID_OFFSET)
    # Flatten every credited (label, catno) pair, normalized into join keys
    rows = []
    for work_id, labels in pd.concat([m, r])[["work_id", "labels"]].itertuples(index=False):  # go through each release and its labels
        for entry in (labels if labels is not None else []):  # releases with no labels are treated as empty
            rows.append((work_id, normalize_name(entry["name"] or ""), normalize_catno(entry["catno"] or "")))  # add the normalized label and catalog number
    pairs = pd.DataFrame(rows, columns=["work_id", "label_key", "catno_key"]).drop_duplicates()  # one row per distinct pair
    # A pair needs both a label and a real catalog number to be evidence
    pairs = pairs[(pairs["label_key"] != "") & ~pairs["catno_key"].isin(EMPTY_CATNO_KEYS)]
    return works[["work_id", "work_source", "artist", "title"]].merge(pairs, on="work_id", how="left")  # attach the pairs to every work, keeping works without pairs


def catno_join(works: pd.DataFrame, mb: pd.DataFrame, releases_path: str = str(PROCESSED / "releases.parquet")) -> pd.DataFrame:
    """Join works to MusicBrainz releases on (label, catalog number); one row per matched pair, plus unmatched works."""
    # Works with their Discogs keys (works with no usable pair keep NaN keys)
    pairs = discogs_label_catno(works, releases_path)
    # Inner-style match on both keys, left join keeps the works that matched nothing
    cols = ["mb_label_key", "mb_catno_key", "mb_release_id", "mb_title", "mb_artist_name", "mb_label", "mb_catno", "mb_year", "mb_date"]
    out = pairs.merge(mb[cols], left_on=["label_key", "catno_key"], right_on=["mb_label_key", "mb_catno_key"], how="left")  # match works to MusicBrainz releases on both keys
    # Fuzzy agreement of artist and title between the two records (0 when nothing matched)
    out["artist_score"] = [fuzz.ratio(normalize_name(a), normalize_name(b)) if isinstance(b, str) else 0 for a, b in zip(out["artist"], out["mb_artist_name"])]
    out["title_score"] = [fuzz.ratio(normalize_name(a), normalize_name(b)) if isinstance(b, str) else 0 for a, b in zip(out["title"], out["mb_title"])]  # how close the two titles are, from 0 to 100
    return out  # one row per matched pair, plus unmatched works


def summarize_catno_join(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per work: outcome, matched-year spread and the best-agreeing MusicBrainz record."""
    # Best agreeing record first, so groupby.first() keeps it for display
    ordered = joined.sort_values(["artist_score", "title_score"], ascending=False)
    g = ordered.groupby("work_id", sort=False)  # group the rows of each work
    out = g.agg(work_source=("work_source", "first"), artist=("artist", "first"), title=("title", "first"),  # one summary row per work
                has_key=("catno_key", lambda s: s.notna().any()), n_mb=("mb_release_id", "count"),
                n_years=("mb_year", "nunique"), year_min=("mb_year", "min"), year_max=("mb_year", "max"),
                mb_artist=("mb_artist_name", "first"), mb_title=("mb_title", "first"), mb_label=("mb_label", "first"),
                mb_catno=("mb_catno", "first"), mb_year=("mb_year", "first"), artist_score=("artist_score", "max"),
                title_score=("title_score", "max")).reset_index()
    # Outcome ladder: no key, no match, match without a year, one agreed year, conflicting years
    out["status"] = np.select([~out["has_key"], out["n_mb"] == 0, out["n_years"] == 0, out["n_years"] == 1],
                              ["no_label_catno", "no_mb_match", "match_no_year", "unique_year"], "multi_year")
    return out  # one row per work with its outcome
