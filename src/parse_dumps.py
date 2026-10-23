"""
Stream Discogs XML dumps into Parquet.

The raw Discogs dumps (Masters ~3 GB, Releases ~60 GB) are too large to load
into memory, open in an editor, or version control. This module reads each
dump exactly once with a streaming XML parser (`xml.etree.ElementTree.iterparse`)
and writes memory-safe, chunked Parquet files. Everything downstream (EDA,
feature building, modelling) reads Parquet, never the XML directly.

Usage (CLI):
    python -m src.parse_dumps masters discogs_20260901_masters.xml data/processed/masters.parquet
    python -m src.parse_dumps releases discogs_20260901_releases.xml data/processed/releases.parquet

Usage (notebook):
    from src.parse_dumps import convert_masters, convert_releases
    summary = convert_masters(xml_path, out_path)
"""

from __future__ import annotations  # allow type hints to be read lazily

import argparse  # read command line arguments
import time  # measure elapsed time
import xml.etree.ElementTree as ET  # streaming XML parser
from pathlib import Path  # file paths
from typing import Callable, Optional  # types for function arguments

import pyarrow as pa  # in-memory columnar tables
import pyarrow.parquet as pq  # Parquet file writer

# Rows are buffered in a plain Python list and flushed to a Parquet row-group
# every CHUNK_SIZE rows, instead of building one giant pyarrow Table in memory.
CHUNK_SIZE = 50_000

# How often (in rows written) to print a progress line during a long conversion.
LOG_EVERY = 200_000  # log progress at this row interval

# Explicit output schema for the Masters table. Declaring it up front (instead
# of letting pyarrow infer it per chunk) avoids type-mismatch errors between
# chunks, e.g. a chunk where every "year" happens to be null would otherwise
# infer as a null-type column and fail to concatenate with other chunks.
MASTER_SCHEMA = pa.schema([  # fixed column names and types for Masters
    ("master_id", pa.int64()),
    ("main_release_id", pa.int64()),
    ("artist_ids", pa.list_(pa.int64())),
    ("artist_names", pa.list_(pa.string())),
    ("title", pa.string()),
    ("year", pa.int32()),
    ("genres", pa.list_(pa.string())),
    ("styles", pa.list_(pa.string())),
    ("data_quality", pa.string()),
])

# Same idea for the Releases table. This is a deliberate subset of the fields
# present in a <release> element: only what we expect to need to validate /
# filter Masters against (main-release flag, format, country, release date),
# plus a second group of fields (see below) added specifically to help
# confirm or reject suspected duplicate Masters (EDA cleaning decision #9:
# 41,273 Masters flagged as possible duplicates by title+artist+year alone).
RELEASE_SCHEMA = pa.schema([  # fixed column names and types for Releases
    ("release_id", pa.int64()),
    ("status", pa.string()),
    ("artist_ids", pa.list_(pa.int64())),
    ("artist_names", pa.list_(pa.string())),
    ("title", pa.string()),
    ("master_id", pa.int64()),
    ("is_main_release", pa.bool_()),
    ("country", pa.string()),
    ("released", pa.string()),
    ("genres", pa.list_(pa.string())),
    ("styles", pa.list_(pa.string())),
    ("formats", pa.list_(pa.string())),
    ("format_descriptions", pa.list_(pa.string())),
    ("data_quality", pa.string()),
    # --- fields below added to compare release *content*, not just metadata ---
    # Full tracklist as a list of structs (one per track), so downstream code
    # can compare the actual songs/durations across releases sharing a
    # master, a much stronger duplicate/reissue signal than title+artist+year.
    ("tracklist", pa.list_(pa.struct([
        ("position", pa.string()),
        ("title", pa.string()),
        ("duration", pa.string()),
    ]))),
    # Track count, derived during parsing for convenience (avoids every
    # downstream consumer re-computing len(tracklist) themselves).
    ("n_tracks", pa.int32()),
    # Barcode and matrix/runout values: about as close to a ground-truth
    # "same physical product" key as Discogs data offers. Kept as separate
    # string lists (not a generic identifiers struct) because these are the
    # only two identifier types we actually need for duplicate resolution.
    ("barcodes", pa.list_(pa.string())),
    ("matrix_numbers", pa.list_(pa.string())),
    # Label name + catalog number per <label> entry. A release moving to a
    # different label/catno for what is otherwise the same recording is a
    # classic reissue signal, distinct from a genuine duplicate/error.
    ("labels", pa.list_(pa.struct([
        ("name", pa.string()),
        ("catno", pa.string()),
    ]))),
    # Per-format quantity and free-text note, aligned by index with
    # `formats` (formats[i] / format_qty[i] / format_text[i] describe the
    # same <format> element). `text` often literally contains "Reissue",
    # "Repress" or "Remastered", which directly distinguishes editions.
    ("format_qty", pa.list_(pa.int32())),
    ("format_text", pa.list_(pa.string())),
])


# Every numeric field this helper feeds (release/master ids, year, n_tracks,
# format_qty, ...) is a small number in practice; real Discogs IDs top out
# around 19M. Bounding to the int32 range here guards every pa.int32()
# column against overflow crashes from corrupt source data, e.g. a real
# case in the wild: release_id=6662697 had a <format qty="..."> attribute
# containing a 55-digit garbage value instead of a real quantity.
_INT32_MIN, _INT32_MAX = -2_147_483_648, 2_147_483_647  # range of values a 32-bit integer can hold


def _int_or_none(text: Optional[str]) -> Optional[int]:  # turn XML text into an integer, if possible
    """Parse an XML text node as an int, or None if missing / not numeric /
    implausibly large (corrupt data).

    Discogs XML represents "no value" as either a missing tag (text is None)
    or, in some fields, the literal string "0"; we only collapse the former
    to None here and keep "0" as the integer 0, since 0 is sometimes a
    meaningful (if suspicious) value in this data (e.g. year="0").
    """
    if text is None:  # no tag means no value
        return None  # report missing
    text = text.strip()  # drop surrounding spaces
    if not text.lstrip("-").isdigit():  # anything that is not a whole number is invalid
        return None  # report missing
    value = int(text)  # convert the clean text to an integer
    # Reject values that can't fit an int32 column (see _INT32_MIN/_MAX
    # comment above) rather than letting pyarrow crash on them mid-write.
    if not (_INT32_MIN <= value <= _INT32_MAX):  # reject numbers too big for the column
        return None  # report missing
    return value  # valid integer


def _artist_ids(elem: ET.Element, path: str = "./artists/artist/id") -> list[int]:  # artist ids of one record
    """Collect all numeric artist IDs under `elem` at `path`, skipping bad ones."""
    return [i for i in (_int_or_none(e.text) for e in elem.findall(path)) if i is not None]  # keep only ids that parsed correctly


def _artist_names(elem: ET.Element, path: str = "./artists/artist/name") -> list[str]:  # artist names of one record
    """Collect all non-empty artist names under `elem` at `path`."""
    return [e.text for e in elem.findall(path) if e.text]  # keep only names that are not empty


def parse_master(elem: ET.Element) -> dict:  # turn one master record into a flat row
    """Flatten a <master> element into one flat row.

    A <master> looks roughly like:
        <master id="113">
          <main_release>116925</main_release>
          <artists><artist><id>3225</id><name>Vince Watson</name></artist></artists>
          <genres><genre>Electronic</genre></genres>
          <styles><style>Techno</style><style>Tech House</style></styles>
          <year>2002</year>
          <title>Moments In Time</title>
          <data_quality>Correct</data_quality>
          <videos>...</videos>  <!-- intentionally not extracted, not needed -->
        </master>
    """
    return {  # one value per Masters column
        "master_id": int(elem.get("id")),
        "main_release_id": _int_or_none(elem.findtext("main_release")),
        "artist_ids": _artist_ids(elem),
        "artist_names": _artist_names(elem),
        "title": elem.findtext("title"),
        "year": _int_or_none(elem.findtext("year")),
        "genres": [g.text for g in elem.findall("./genres/genre") if g.text],
        "styles": [s.text for s in elem.findall("./styles/style") if s.text],
        "data_quality": elem.findtext("data_quality"),
    }


# --- Release-only helpers below, added to support duplicate-Master resolution ---
# (EDA cleaning decision #9). Each pulls out one extra slice of a <release>
# that isn't captured by the original 14 fields.


def _tracklist(elem: ET.Element) -> list[dict]:  # tracks of one release
    """Extract the release's tracklist, in order, as {position, title, duration} dicts.

    Kept deliberately close to the raw listing (rather than collapsed to just a
    track count) because comparing actual track titles/durations across releases
    that share a master is a far stronger signal for confirming or rejecting a
    suspected duplicate than title/artist/year matching alone.
    """
    return [  # one entry per track, in order
        {
            "position": t.findtext("position"),
            "title": t.findtext("title"),
            "duration": t.findtext("duration"),
        }
        for t in elem.findall("./tracklist/track")
    ]


def _identifiers(elem: ET.Element) -> tuple[list[str], list[str]]:  # barcodes and matrix numbers of a release
    """Pull out just the two identifier types useful for "same product" matching.

    Discogs XML's <identifiers> also carries things like Rights Society, Mould
    SID, Label Code, etc. via a free-form `type` attribute; we deliberately
    skip those here: they're not useful for duplicate resolution and would
    just add noise. Only Barcode and Matrix/Runout (physical pressing
    etchings) are kept, since two releases sharing either are very likely the
    same physical edition.
    """
    barcodes: list[str] = []  # barcodes found so far
    matrix_numbers: list[str] = []  # matrix numbers found so far
    for ident in elem.findall("./identifiers/identifier"):  # go through each identifier of the release
        id_type = (ident.get("type") or "").strip().lower()  # identifier kind in lower case, '' if absent
        value = ident.get("value")  # the identifier itself
        if not value:  # skip identifiers with no value
            continue  # move on to the next identifier
        if id_type == "barcode":  # barcodes go in their own list
            barcodes.append(value)  # save the barcode
        elif "matrix" in id_type or "runout" in id_type:  # matrix and runout etchings count as the same kind
            matrix_numbers.append(value)  # save the etching
    return barcodes, matrix_numbers  # return both lists


def _labels(elem: ET.Element) -> list[dict]:  # label name and catalog number of a release
    """Extract {name, catno} per <label> entry.

    A change in label or catalog number for what is otherwise the same
    recording is one of the clearest signals of a reissue rather than a
    genuine duplicate/error, so this is kept per-label rather than joined
    into a single string.
    """
    return [  # one entry per label that has a name
        {"name": l.get("name"), "catno": l.get("catno")}
        for l in elem.findall("./labels/label")
        if l.get("name")
    ]


def _formats(elem: ET.Element) -> tuple[list[str], list[Optional[int]], list[Optional[str]], list[str]]:  # names, quantities, notes and descriptions of the formats
    """Parse every <format> child into four parallel/flattened pieces.

    Returns (names, qty, text, descriptions). `names`, `qty` and `text` are
    parallel lists aligned by index (names[i]/qty[i]/text[i] describe the same
    <format> element); `descriptions` stays flattened across all formats,
    matching the original behaviour. Only <format> elements with a `name`
    attribute are kept, so the three parallel lists stay aligned; this
    mirrors the original `formats` extraction's own filter.

    `qty` (e.g. "2" for a double LP) and `text` (free text, often literally
    "Reissue", "Repress" or "Remastered") are new: they help distinguish a
    reissue/different edition from a genuine duplicate.
    """
    names: list[str] = []  # format names found so far
    qty: list[Optional[int]] = []  # quantity of each format
    text: list[Optional[str]] = []  # free-text note of each format
    descriptions: list[str] = []  # descriptions of all formats together
    for f in elem.findall("./formats/format"):  # go through each format of the release
        name = f.get("name")  # format name
        if not name:  # skip formats without a name, to keep lists aligned
            continue  # move on to the next format
        names.append(name)  # save the name
        qty.append(_int_or_none(f.get("qty")))  # save the quantity, None if invalid
        text.append(f.get("text"))  # save the note
        descriptions.extend(d.text for d in f.findall("./descriptions/description") if d.text)  # add the non-empty descriptions
    return names, qty, text, descriptions  # return the four lists


def parse_release(elem: ET.Element) -> dict:  # turn one release record into a flat row
    """Flatten a <release> element into one flat row (subset needed for validation/filtering,
    plus content/identity fields used to resolve suspected duplicate Masters).

    `master_id` is read from the <master_id is_main_release="true|false">...</master_id>
    child: the element's text is the master it belongs to, and the attribute tells us
    whether this particular release is the "main" (representative) release of that master.
    """
    master = elem.find("master_id")  # master this release belongs to, if any
    format_names, format_qty, format_text, format_descriptions = _formats(elem)  # format names, quantities, notes and descriptions
    barcodes, matrix_numbers = _identifiers(elem)  # barcodes and matrix numbers
    tracklist = _tracklist(elem)  # list of tracks
    return {  # one value per Releases column
        "release_id": int(elem.get("id")),
        "status": elem.get("status"),
        "artist_ids": _artist_ids(elem),
        "artist_names": _artist_names(elem),
        "title": elem.findtext("title"),
        "master_id": _int_or_none(master.text) if master is not None else None,
        "is_main_release": (master.get("is_main_release") == "true") if master is not None else None,
        "country": elem.findtext("country"),
        "released": elem.findtext("released"),
        "genres": [g.text for g in elem.findall("./genres/genre") if g.text],
        "styles": [s.text for s in elem.findall("./styles/style") if s.text],
        "formats": format_names,
        "format_descriptions": format_descriptions,
        "data_quality": elem.findtext("data_quality"),
        # --- added for duplicate-Master resolution (see module/schema comments) ---
        "tracklist": tracklist,
        "n_tracks": len(tracklist),
        "barcodes": barcodes,
        "matrix_numbers": matrix_numbers,
        "labels": _labels(elem),
        "format_qty": format_qty,
        "format_text": format_text,
    }


def stream_to_parquet(  # read an XML dump and write it as Parquet
    xml_path: Path,
    out_path: Path,
    tag: str,
    parse_fn: Callable[[ET.Element], dict],
    schema: pa.Schema,
    limit: Optional[int] = None,
    chunk_size: int = CHUNK_SIZE,
    log_every: int = LOG_EVERY,
) -> dict:
    """
    Stream `tag` elements (e.g. "master"/"release") out of `xml_path` and write
    them to `out_path` as Parquet, `chunk_size` rows at a time.

    Memory stays flat regardless of dump size, via two clears per element:
      - `elem.clear()` drops the element's own children/text once we've parsed it,
        freeing the (potentially large, e.g. <videos>) subtree we didn't extract.
      - `root.clear()` drops the now-empty sibling shells that `iterparse` would
        otherwise keep appending to the root element for the entire parse, a
        well-known stdlib ElementTree gotcha. This is safe here because dump
        records are flat siblings of the root (no nested state to lose).
    """
    xml_path = Path(xml_path)  # accept strings as well as Path objects
    out_path = Path(out_path)  # same for the output path
    out_path.parent.mkdir(parents=True, exist_ok=True)  # create the output folder if missing

    # Ask for "start" events too, purely to grab a reference to the root
    # element on the first callback so we can periodically clear it below.
    context = ET.iterparse(xml_path, events=("start", "end"))  # iterator that reads the file piece by piece
    _, root = next(context)  # first event gives the root element

    rows: list[dict] = []  # rows waiting to be written
    n_written = 0  # rows written so far
    n_errors = 0  # records that failed to parse
    t0 = time.time()  # start time, to measure speed

    # ParquetWriter lets us append row-groups incrementally instead of
    # holding the whole table (millions of rows) in memory before writing.
    writer = pq.ParquetWriter(out_path, schema, compression="snappy")  # writer that appends chunks to the output file
    try:  # make sure the writer is closed at the end
        for event, elem in context:  # go through the XML events one by one
            # We only care about the closing tag of each record element,
            # i.e. once all of its children have been fully parsed.
            if event != "end" or elem.tag != tag:  # ignore anything that is not a finished record
                continue  # skip to the next event

            try:  # parse the record safely
                rows.append(parse_fn(elem))  # add the parsed row to the buffer
            except Exception:  # any parsing problem lands here
                # A handful of malformed records shouldn't abort a multi-GB
                # conversion; count and skip instead.
                n_errors += 1  # count the failed record
            finally:  # always runs, success or not
                # Free this element's subtree and the empty shells piling up
                # under root, regardless of whether parsing succeeded.
                elem.clear()  # free the record's memory
                root.clear()  # free the root's memory too

            # Flush a full chunk to disk as one Parquet row-group.
            if len(rows) >= chunk_size:  # buffer is full, time to write
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))  # write the buffered rows as one chunk
                n_written += len(rows)  # update the written count
                rows.clear()  # empty the buffer

                if n_written % log_every == 0:  # only log at round numbers of rows
                    elapsed = time.time() - t0  # seconds since the start
                    print(  # print one progress line
                        f"  {n_written:>12,} rows written | "
                        f"{elapsed:6.1f}s elapsed | {n_written / elapsed:8.0f} rows/s"
                    )

            # Testing/debugging escape hatch: stop after `limit` rows total.
            if limit is not None and (n_written + len(rows)) >= limit:  # stop early once the row limit is reached
                break  # leave the loop

        # Flush whatever is left in the buffer (a partial, final chunk).
        if rows:  # rows still in the buffer
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))  # write the last partial chunk
            n_written += len(rows)  # update the written count
    finally:  # always runs, even after an error
        writer.close()  # close the file so it is complete

    elapsed = time.time() - t0  # total running time
    summary = {  # result details of this conversion
        "tag": tag,
        "xml_path": str(xml_path),
        "output_path": str(out_path),
        "rows_written": n_written,
        "parse_errors": n_errors,
        "elapsed_seconds": round(elapsed, 1),
    }
    print(f"Done: {n_written:,} rows -> {out_path} ({elapsed:.1f}s, {n_errors} parse errors)")  # tell the user how it went
    return summary  # return the details


def convert_masters(xml_path: Path, out_path: Path, limit: Optional[int] = None) -> dict:  # convert a Masters dump
    """Convert a Masters XML dump to Parquet. Thin wrapper around stream_to_parquet."""
    return stream_to_parquet(xml_path, out_path, "master", parse_master, MASTER_SCHEMA, limit=limit)  # run the generic converter with the Masters settings


def convert_releases(xml_path: Path, out_path: Path, limit: Optional[int] = None) -> dict:  # convert a Releases dump
    """Convert a Releases XML dump to Parquet. Thin wrapper around stream_to_parquet."""
    return stream_to_parquet(xml_path, out_path, "release", parse_release, RELEASE_SCHEMA, limit=limit)  # run the generic converter with the Releases settings


def main() -> None:  # command line entry point
    """Read the command line and convert the chosen dump (masters or releases) to Parquet. Input: kind, xml path, output path and the optional --limit. Output: none, the Parquet file is written."""
    # CLI entry point: `python -m src.parse_dumps <masters|releases> <xml_path> <out_path>`.
    parser = argparse.ArgumentParser(description="Convert a Discogs XML dump into Parquet.")  # parser for the command line
    parser.add_argument("kind", choices=["masters", "releases"])  # which dump to convert
    parser.add_argument("xml_path", type=Path)  # where the XML file is
    parser.add_argument("out_path", type=Path)  # where to write the Parquet file
    parser.add_argument("--limit", type=int, default=None, help="Only convert the first N records (testing).")  # optional cap on rows, for tests
    args = parser.parse_args()  # read the values typed by the user

    convert_fn = convert_masters if args.kind == "masters" else convert_releases  # pick the converter that matches the chosen dump
    convert_fn(args.xml_path, args.out_path, limit=args.limit)  # run the conversion


if __name__ == "__main__":  # only when run as a script
    main()  # start the command line tool
