"""New styles of the WHAT reality per demo artist and year, for the tab 6 panel. Pure logic plus one run function, no model."""

import json  # the style list is stored as JSON text

import pandas as pd  # tables

from app_data import all_artists, load_demo_config  # demo artists of both pools
from config import CONFIG, ROOT  # project settings and root
from snapshots import _split_styles  # same list-to-set conversion as the label rule

LIVE = load_demo_config()["live"]  # file names of the demo app
OUT_PATH = ROOT / LIVE["what_new_styles"]  # output file, never overwritten
LABELS_PATHS = [ROOT / LIVE["folder"] / LIVE["what_labels"], ROOT / load_demo_config()["pool_b"]["paths"]["what_labels"]]  # stored WHAT labels of both pools, the control reference
HOLDOUT_START = CONFIG["years"]["holdout_start"]  # first year the app can play


def artist_year_table(works: pd.DataFrame) -> pd.DataFrame:  # one row per active year of one artist
    """Active years of one artist with next_active_year, the styles seen through the year (undated Works count as seen) and the styles of the next active year."""
    undated = set().union(*(_split_styles(s) for s in works[works["year"].isna()]["styles"]))  # styles of undated Works, counted as seen
    dated = works[works["year"].notna()].assign(styles=lambda t: t["styles"].map(_split_styles))  # dated Works with style sets
    by_year = dated.groupby(dated["year"].astype(int))["styles"].agg(lambda c: set().union(*c))  # styles of each active year
    years = sorted(by_year.index)  # active years, ascending
    rows, seen = [], set(undated)  # rows and the running vocabulary
    for i, y in enumerate(years):  # one pass in time order
        seen |= by_year[y]  # every style of the year is known at its end
        nxt = years[i + 1] if i + 1 < len(years) else None  # next active year
        new = sorted((by_year[nxt] - seen)) if nxt is not None else []  # styles of the next active year not seen so far
        rows.append({"year": y, "next_active_year": nxt, "new_styles": new})  # label rule at artist-year level
    return pd.DataFrame(rows)  # one row per active year


def build_new_styles(works: pd.DataFrame, ids: list, first_year: int, last_year: int) -> pd.DataFrame:  # the output table
    """Rows (artist_id, year, next_active_year, new_styles) for every demo artist and every year from first_year to last_year after its first active year; a year without a dated Work takes the row of the last active year before it."""
    out = []  # all rows
    for aid in ids:  # each demo artist
        t = artist_year_table(works[works["artist_id"] == aid])  # active years of the artist
        if t.empty:  # no dated Work at all
            continue  # nothing to build
        active = t.set_index("year")  # lookup by active year
        for y in range(first_year, last_year + 1):  # every year the app can show
            prior = [a for a in active.index if a <= y]  # active years up to y
            if not prior:  # no history yet
                continue  # no row
            r = active.loc[max(prior)]  # the year itself, or the last active year for a gap year
            out.append({"artist_id": aid, "year": y, "next_active_year": r["next_active_year"], "new_styles": json.dumps(r["new_styles"], ensure_ascii=False)})  # one row
    t = pd.DataFrame(out)  # table
    t["next_active_year"] = t["next_active_year"].astype("Int64")  # whole years, missing for the last active year
    return t  # rows


def control(table: pd.DataFrame, labels: pd.DataFrame) -> dict:  # stored label against the new styles
    """For every row with a stored label, label == (new_styles is not empty); also compares next_active_year. Returns the counts."""
    m = labels[labels["label"].notna()].merge(table, on=["artist_id", "year"], how="left", suffixes=("_l", ""))  # stored labels next to the rows
    has_new = m["new_styles"].map(lambda s: len(json.loads(s)) > 0 if isinstance(s, str) else None)  # new styles present, None without a row
    return {"rows": len(m), "missing": int(has_new.isna().sum()), "label_mismatch": int((has_new.dropna().astype(int) != m.loc[has_new.notna(), "label"].astype(int)).sum()), "year_mismatch": int((m["next_active_year_l"].astype("Int64") != m["next_active_year"]).sum())}  # counts


def stored_labels() -> pd.DataFrame:  # stored labels of both pools
    """The stored WHAT labels of both pools in one table."""
    return pd.concat([pd.read_csv(p) for p in LABELS_PATHS], ignore_index=True)  # pool A and pool B


def control_file() -> dict:  # control of the written file
    """Control of the existing results/demo/what_new_styles.csv against the stored labels of both pools; read only."""
    t = pd.read_csv(OUT_PATH)  # the written file
    return control(t, stored_labels())  # counts


def run() -> None:  # build, control and write
    """Build the file for every demo artist, run the control and write with exclusive creation; nothing is written when the control fails."""
    if OUT_PATH.exists():  # never overwrite, only repeat the control
        res = control_file()  # control of the existing file
        print(f"{OUT_PATH.relative_to(ROOT)} already exists, nothing was written. control: {res['rows']} stored labels, {res['missing']} without a row, label mismatches {res['label_mismatch']}, next_active_year mismatches {res['year_mismatch']}")  # control only
        raise SystemExit(0 if not (res["missing"] or res["label_mismatch"] or res["year_mismatch"]) else 1)  # exit code follows the control
    ids = [int(a["discogs_id"]) for a in all_artists(load_demo_config())]  # the demo artists of both pools
    works = pd.read_parquet(ROOT / CONFIG["paths"]["works"], columns=["artist_id", "year", "styles"], filters=[("artist_id", "in", [float(i) for i in ids])])  # full Works data of the demo artists
    works["artist_id"] = works["artist_id"].astype("int64")  # whole-number ids
    last_year = int(works["year"].max())  # last year of the data
    table = build_new_styles(works, ids, HOLDOUT_START, last_year)  # rows
    res = control(table, stored_labels())  # control against the stored labels of both pools
    print(f"artists {len(ids)}, rows {len(table)}, artists with rows {table['artist_id'].nunique()}, years {HOLDOUT_START} to {last_year}")  # size
    print(f"control: {res['rows']} stored labels, {res['missing']} without a row, label mismatches {res['label_mismatch']}, next_active_year mismatches {res['year_mismatch']}")  # control result
    if res["missing"] or res["label_mismatch"] or res["year_mismatch"]:  # any difference
        raise SystemExit("the control failed, nothing was written")  # stop before writing
    with open(OUT_PATH, "x", encoding="utf-8", newline="") as f:  # exclusive creation
        table.to_csv(f, index=False)  # write the file
    print(f"path {OUT_PATH.relative_to(ROOT)}, {OUT_PATH.stat().st_size / 1024:.1f} KB")  # file size
