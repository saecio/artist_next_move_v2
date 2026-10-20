"""Release title shown in the reality cells of tab 6: one Work per artist-year, album-like first (rule B of docs/decisions.md)."""

import pandas as pd  # tables

from app_data import load_demo_config  # demo settings and the pool B file paths
from config import CONFIG, ROOT  # project settings and root

OUT_PATH = ROOT / CONFIG["paths"]["release_pick"]  # output file, never overwritten
SNAPSHOTS_PATH = ROOT / CONFIG["paths"]["snapshots"]  # cached snapshots, source of is_album_like
FIRST_YEAR = CONFIG["years"]["holdout_start"]  # the file covers the holdout start and every later year
COLUMNS = ["artist_id", "year", "work_id", "title", "is_album_like", "picked_by"]  # output order


def load_works() -> pd.DataFrame:  # Works of both pools with the album marker
    """Works of both demo pools from the holdout start on, with is_album_like read from the snapshots (one row per Work)."""
    paths = [ROOT / CONFIG["paths"]["works_demo"], ROOT / load_demo_config()["pool_b"]["paths"]["works"]]  # pool A file, pool B file
    works = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)  # both pools, one table
    works = works[works["year"] >= FIRST_YEAR].copy()  # holdout start on, earlier years hold only the representative Work
    keep = [("work_id", "in", works["work_id"].tolist())]  # filter pushed to the parquet reader
    snaps = pd.read_parquet(SNAPSHOTS_PATH, columns=["work_id", "is_album_like"], filters=keep)  # album marker of the snapshots
    return works.merge(snaps, on="work_id", how="left", validate="one_to_one")  # every Work has exactly one snapshot


def build_picks(works: pd.DataFrame) -> pd.DataFrame:  # one chosen Work per artist-year
    """Rule B: among the non-re-recording Works of an artist-year, the first album-like Work in the seeded within-year order (works_so_far_this_year); with no album-like Work, the representative Work, which is the last in that order (the last remaining one when the representative is a re-recording). Artist-years with only re-recordings get no row."""
    ok = works[~works["is_rerecording"]].sort_values(["artist_id", "year", "works_so_far_this_year"])  # re-recordings skipped, within-year order
    keys = ["artist_id", "year"]  # artist-year key
    albums = ok[ok["is_album_like"]].groupby(keys, as_index=False).head(1).assign(picked_by="album")  # first album-like Work of each artist-year
    rest = ok.merge(albums[keys], on=keys, how="left", indicator=True).query("_merge == 'left_only'")  # artist-years without an album-like Work
    fallback = rest.groupby(keys, as_index=False).tail(1).assign(picked_by="fallback")  # last Work in the order
    return pd.concat([albums, fallback])[COLUMNS].sort_values(keys).reset_index(drop=True)  # fixed columns and order


def run_controls(works: pd.DataFrame, picks: pd.DataFrame) -> bool:  # print the controls and return True when all pass
    """Controls: every Work has an album marker, one pick per artist-year, every pick is a non-re-recording Work of its artist-year, coverage of the route table and of the tab 6 years."""
    keys = ["artist_id", "year"]  # artist-year key
    years = works.groupby(keys).agg(n=("work_id", "size"), rec=("is_rerecording", "sum"))  # Works and re-recordings per artist-year
    only_rec = years[years["n"] == years["rec"]]  # artist-years with only re-recordings
    route = pd.read_csv(ROOT / "results/demo/route_table.csv")  # route table, its artist-years need a title when the next year is a Yes
    nxt = route.assign(year=route["year"] + 1)[keys]  # the year each route row refers to (Y + 1)
    covered = nxt.merge(works[keys].drop_duplicates(), on=keys)  # next years that have a Work in the file
    missing = covered.merge(picks[keys], on=keys, how="left", indicator=True).query("_merge == 'left_only'")  # covered next years without a pick
    ok = {  # control name -> passed
        "every Work has an album marker": bool(works["is_album_like"].notna().all()),  # no snapshot missing
        "one pick per artist-year": not picks.duplicated(keys).any(),  # keys unique
        "picks are Works of the file, not re-recordings": bool(picks["work_id"].isin(works.loc[~works["is_rerecording"], "work_id"]).all()),  # membership
        "pick count = artist-years minus re-recording-only ones": len(picks) == len(years) - len(only_rec),  # coverage of the file
        "every route next year that has a Work in the file has a pick": missing.empty,  # route coverage (no artist-year holds only re-recordings)
    }  # end of the controls
    for name, passed in ok.items():  # one line each
        print(("PASS " if passed else "FAIL ") + name)  # control result
    print(f"artist-years {len(years)}, only re-recordings {len(only_rec)}, picks {len(picks)}, years {int(picks['year'].min())} to {int(picks['year'].max())}")  # sizes
    return all(ok.values())  # overall result


def run() -> None:  # build, check, write
    """Build the pick file, run the controls, write it exclusively and print the album and fallback counts."""
    if OUT_PATH.exists():  # never overwrite
        raise SystemExit(f"{OUT_PATH.relative_to(ROOT)} already exists, nothing was run")  # stop early
    works = load_works()  # Works of both pools with the album marker
    picks = build_picks(works)  # rule B
    if not run_controls(works, picks):  # any failure stops the step
        raise SystemExit("a control failed, nothing was written")  # write nothing
    with open(OUT_PATH, "x", encoding="utf-8", newline="") as f:  # exclusive creation
        picks.to_csv(f, index=False)  # write the file
    print(picks["picked_by"].value_counts().to_string())  # album versus fallback
    many = works.groupby(["artist_id", "year"]).filter(lambda g: (~g["is_rerecording"]).sum() > 1)[["artist_id", "year"]].drop_duplicates()  # artist-years with several releases
    print(f"artist-years with several non-re-recording Works: {len(many)}, by pick: {picks.merge(many, on=['artist_id', 'year'])['picked_by'].value_counts().to_dict()}")  # picks among them
    print(f"path {OUT_PATH.relative_to(ROOT)}, rows {len(picks)}")  # output
