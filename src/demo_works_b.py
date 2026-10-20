"""Works file of the pool B artists: same rules and columns as the pool A file, built with the logic of demo_works, with controls."""

import hashlib  # SHA-256 of the pool A file
import re  # playable years read from the decisions entry
import time  # runtime of the step

import pandas as pd  # tables

from app_data import load_demo_config  # demo settings
from config import CONFIG, ROOT  # project settings and root
from demo_works import COLUMNS, FIRST_SHOWN, HOLDOUT_START, OUT_PATH as POOL_A_PATH, build_full, cut_years  # existing pool A logic, reused as it is
from when_demo import demo_artist_ids  # pool A ids

CFG = load_demo_config()  # settings, including artists_b and pool_b
POOL_B = CFG["pool_b"]  # pool B section
OUT_PATH = ROOT / POOL_B["paths"]["works"]  # new file, never overwritten
DECISIONS = ROOT / "docs/decisions_demo.md"  # entry with the playable years


def sha256(path) -> str:  # hash of a file
    """SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()  # hex digest


def independent_counts(ids: list) -> pd.DataFrame:  # expected rows from the raw table
    """Expected rows per artist and year, from works_checkpoint.parquet with plain pandas: all dated Works from the holdout start, one row per year before it."""
    raw = pd.read_parquet(ROOT / CONFIG["paths"]["works"], columns=["artist_id", "year"])  # raw table, two columns
    raw = raw[raw["artist_id"].isin([float(i) for i in ids]) & raw["year"].notna()]  # these artists, dated Works
    n = raw.groupby(["artist_id", "year"]).size().rename("n").reset_index()  # Works per artist-year
    n["artist_id"], n["year"] = n["artist_id"].astype("int64"), n["year"].astype("int64")  # whole numbers
    n = n[n["year"] >= FIRST_SHOWN].copy()  # nothing before the first shown year
    n["expected"] = n["n"].where(n["year"] >= HOLDOUT_START, 1)  # one representative Work before the holdout start
    return n[["artist_id", "year", "expected"]]  # expected rows


def playable_from_decisions(ids: list) -> dict:  # playable years of the registered list
    """Playable years per id from the entry 'Demo pool B: final list' of docs/decisions_demo.md."""
    text = DECISIONS.read_text(encoding="utf-8")  # decisions file
    entry = text.split("## Demo pool B: final list (19-10-2026)", 1)[1].split("\n## ", 1)[0]  # the entry
    found = {int(i): int(n) for i, n in re.findall(r"\((\d+), [^,()]+(?: / [^,()]+)?, (\d+) playable years\)", entry)}  # id -> playable years
    return {i: found[i] for i in ids}  # only the requested ids, fails when one is missing


def controls(out: pd.DataFrame, ids: list, ids_a: list, hash_before: str) -> pd.DataFrame:  # all controls
    """Run the five controls and return the table; the caller stops when one fails."""
    rows = []  # control rows
    regenerated = cut_years(build_full(ids_a)).to_csv(index=False)  # pool A rows rebuilt in memory
    on_disk = POOL_A_PATH.read_bytes().decode("utf-8").replace("\r\n", "\n")  # pool A file text
    rows.append({"control": "1a pool A rebuilt in memory equals the file exactly", "detail": f"{len(regenerated)} characters", "passed": regenerated == on_disk})  # exact text equality
    rows.append({"control": "1b pool A file hash before the run", "detail": hash_before[:16], "passed": True})  # reference hash, compared again after the run
    exp = independent_counts(ids)  # independent expected rows
    got = out.groupby(["artist_id", "year"]).size().rename("got").reset_index()  # rows of the new file
    m = exp.merge(got, on=["artist_id", "year"], how="outer")  # both sides
    bad = int(((m["expected"] != m["got"]) | m["expected"].isna() | m["got"].isna()).sum())  # mismatching artist-years
    rows.append({"control": "2 Works per artist and year equal the independent count", "detail": f"{len(m)} artist-years, {bad} differ", "passed": bad == 0})  # count control
    missing, leaked = sorted(set(ids) - set(out["artist_id"])), sorted(set(ids_a) & set(out["artist_id"]))  # ids absent, pool A ids present
    rows.append({"control": "3 every id appears, no pool A id appears", "detail": f"{out['artist_id'].nunique()} of {len(ids)} ids, missing {missing}, pool A ids {leaked}", "passed": not missing and not leaked})  # id control
    lo, hi = POOL_B["coverage_years"]  # coverage window
    window = out[(out["year"] >= lo) & (out["year"] <= hi)].groupby("artist_id")["year"].nunique().reindex(ids).fillna(0).astype(int)  # distinct years in the window
    play = out[out["year"] >= lo].groupby("artist_id")["year"].nunique().reindex(ids).fillna(0).astype(int)  # playable years in the file
    reg = playable_from_decisions(ids)  # registered playable years
    diff = {i: (int(play[i]), reg[i]) for i in ids if int(play[i]) != reg[i]}  # differences with the registered column
    short = [i for i in ids if window[i] < POOL_B["min_distinct_years"]]  # ids below the rule
    rows.append({"control": f"4 at least {POOL_B['min_distinct_years']} distinct years in {lo} to {hi}, playable years equal the entry", "detail": f"below the rule {short}, playable differences {diff}", "passed": not short and not diff})  # coverage control
    ref = pd.read_csv(POOL_A_PATH, nrows=5)  # pool A columns and dtypes
    same_cols = list(out.columns) == list(ref.columns) == COLUMNS  # same names and order
    same_types = out.head(5).reset_index(drop=True).dtypes.astype(str).to_dict() == ref.dtypes.astype(str).to_dict() if same_cols else False  # same dtypes
    full = pd.read_csv(POOL_A_PATH)  # whole pool A file for the dtype check on the new file as read back
    back = pd.read_csv(pd.io.common.StringIO(out.to_csv(index=False)))  # new file as it will be read
    rows.append({"control": "5 columns and dtypes equal the pool A file", "detail": f"{list(back.columns) == list(full.columns)}, {back.dtypes.astype(str).to_dict() == full.dtypes.astype(str).to_dict()}", "passed": same_cols and back.dtypes.astype(str).to_dict() == full.dtypes.astype(str).to_dict()})  # column control
    return pd.DataFrame(rows)  # control table


def run() -> None:  # whole step
    """Build, check and write the pool B Works file; refuses to run when it exists and writes nothing when a control fails."""
    start = time.time()  # clock of the step
    if OUT_PATH.exists():  # never overwrite
        raise SystemExit(f"{OUT_PATH.relative_to(ROOT)} already exists, nothing was run")  # stop early
    ids = [int(a["discogs_id"]) for a in CFG["artists_b"]]  # the 21 pool B ids
    ids_a = demo_artist_ids()  # the 40 pool A ids
    hash_before = sha256(POOL_A_PATH)  # pool A file hash before the run
    out = cut_years(build_full(ids))  # same logic and rules as pool A
    table = controls(out, ids, ids_a, hash_before)  # all controls
    pd.set_option("display.width", 250, "display.max_colwidth", 120)  # readable table
    print(table.to_string(index=False), flush=True)  # control output
    if not table["passed"].all():  # any failure stops the step
        raise SystemExit("a control failed, nothing was written")  # write nothing
    with open(OUT_PATH, "x", encoding="utf-8", newline="") as f:  # exclusive creation
        out.to_csv(f, index=False)  # write the file
    after = sha256(POOL_A_PATH)  # pool A file hash after the run
    print(f"\ncontrol 1c pool A file hash after the run equals the one before: {after == hash_before} ({after[:16]})")  # hash control
    if after != hash_before:  # cannot happen unless the file changed
        raise SystemExit("the pool A file changed")  # report
    per = out.groupby("artist_id").agg(rows=("work_id", "size"), first_year=("year", "min"), last_year=("year", "max"), rerecordings=("is_rerecording", "sum"))  # per artist
    names = {int(a["discogs_id"]): a["name"] for a in CFG["artists_b"]}  # display names
    per.insert(0, "name", [names[i] for i in per.index])  # names first
    print("\n" + per.reindex(ids).to_string())  # per-artist table in list order
    print(f"\npath {OUT_PATH.relative_to(ROOT)}, {OUT_PATH.stat().st_size / 1024:.1f} KB, rows {len(out)}, rows 2010 to {HOLDOUT_START - 1} {int((out['year'] < HOLDOUT_START).sum())}, rows {HOLDOUT_START} on {int((out['year'] >= HOLDOUT_START).sum())}")  # path, size, rows
    print(f"is_rerecording rows {int(out['is_rerecording'].sum())}, runtime {time.time() - start:.1f} s")  # re-recordings and runtime
