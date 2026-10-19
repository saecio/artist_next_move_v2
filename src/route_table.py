"""Read-only table of artist-years for choosing a disclosed demo route: WHEN and WHAT verdicts next to their realities, both pools."""

import pandas as pd  # tables

from app_data import all_artists, load_demo_config  # demo settings and the mixed artist list
from app_live import artist_timeline, load_tables  # live scores and stored labels, as tab 6 reads them
from config import CONFIG, ROOT  # project settings and root
from demo_game import is_counted, scored_years, verdict  # same rules as tab 6

OUT_PATH = ROOT / "results/demo/route_table.csv"  # new file, never overwritten
DECIMALS = 4  # decimals of the scores in the file


def pool_of(cfg: dict) -> dict:  # pool key per artist id
    """Pool key of every artist id, from the lists named in pool_b.pool_lists."""
    return {int(a["discogs_id"]): key for key, lst in cfg["pool_b"]["pool_lists"].items() for a in cfg[lst]}  # id -> pool


def yn(label) -> str | None:  # label as yes or no
    """'yes' or 'no' for a stored label, None when it is empty."""
    return None if pd.isna(label) else ("yes" if label == 1 else "no")  # stored label text


def build_table() -> pd.DataFrame:  # one row per artist and playable year
    """Every playable year from the holdout start for the artists of both pools, with WHEN and WHAT verdicts and realities."""
    cfg = load_demo_config()  # settings
    threshold = cfg["game"]["threshold"]  # WHEN threshold of tab 6
    cutoff = load_tables()["what_top10"]["score_cutoff"]  # WHAT cutoff of tab 6 (top 10% rule), not 0.5
    pools = pool_of(cfg)  # pool per id
    rows = []  # table rows
    for a in all_artists(cfg):  # both pools, sorted by name
        aid = int(a["discogs_id"])  # artist id
        tl = artist_timeline(aid)  # timeline, representative Work per year
        for year in scored_years(tl, CONFIG["years"]["holdout_start"]):  # playable years
            r = tl[tl["year"] == year].iloc[0]  # row of the year
            when = verdict(r["next_1y"], threshold)  # WHEN verdict
            wl = yn(r["label_next_1y"])  # WHEN stored label
            counted = bool(is_counted(r))  # counted in the scoreboard
            has_what = pd.notna(r["stage0_score"])  # WHAT score exists
            what = verdict(r["stage0_score"], cutoff) if has_what else None  # WHAT verdict
            tl_what = yn(r["what_label"]) if has_what else None  # WHAT stored label
            rows.append({  # one row per artist-year
                "artist": a["name"], "artist_id": aid, "pool": pools[aid], "year": year,  # identity
                "when_score": round(float(r["next_1y"]), DECIMALS), "when_verdict": when,  # WHEN prediction
                "when_reality": wl, "when_counted": counted,  # WHEN reality and whether it counts
                "when_right": (when == wl) if (wl is not None and counted) else None,  # right only when counted
                "what_available": has_what, "what_score": round(float(r["stage0_score"]), DECIMALS) if has_what else None,  # WHAT availability and score
                "what_verdict": what, "what_reality": tl_what,  # WHAT prediction and reality
                "what_label_after_holdout": bool(r["what_label_after_holdout"]) if has_what and pd.notna(r["what_label_after_holdout"]) else None,  # WHAT flag
                "what_right": (what == tl_what) if (what is not None and tl_what is not None) else None,  # right when a label exists
            })  # one row per artist-year
    return pd.DataFrame(rows)  # the table


def summary(t: pd.DataFrame) -> None:  # print the requested counts and lists
    """Print the counts and lists of the route table; results only."""
    pd.set_option("display.width", 250, "display.max_rows", 500)  # readable output
    def rate(s):  # right of known
        k = s.dropna(); return f"{int(k.sum())} right of {len(k)} with a reality"  # text
    wn, an = t[t["when_verdict"] == "no"], t[t["what_verdict"] == "no"]  # No verdict rows
    ay = t[t["what_verdict"] == "yes"]  # WHAT Yes rows
    print(f"rows {len(t)}, pools {t['pool'].value_counts().to_dict()}, WHAT available {int(t['what_available'].sum())}")  # size
    print(f"\nWHEN No verdicts {len(wn)}: {rate(wn['when_right'])} (counted rows only)")  # WHEN No
    print(f"WHAT No verdicts {len(an)}: {rate(an['what_right'])}")  # WHAT No
    cols = ["artist", "pool", "year", "when_score", "when_verdict", "when_reality", "when_counted", "when_right", "what_score", "what_verdict", "what_reality", "what_right"]  # printed columns
    both = t[(t["when_verdict"] == "no") & (t["what_verdict"] == "no")]  # No and No
    print(f"\nWHEN No and WHAT No in the same year: {len(both)}")  # count
    print(both[cols].to_string(index=False))  # rows
    anyno = t[(t["when_verdict"] == "no") | (t["what_verdict"] == "no")]  # at least one No
    print(f"\nartist-years where at least one of WHEN or WHAT says No: {len(anyno)}")  # count
    print(anyno[cols].to_string(index=False))  # rows
    per = anyno.assign(n_no=(anyno["when_verdict"] == "no").astype(int) + (anyno["what_verdict"] == "no").astype(int)).groupby("year").agg(artist_years_with_no=("artist", "size"), no_verdicts=("n_no", "sum")).sort_values(["no_verdicts", "artist_years_with_no"], ascending=False)  # No verdicts per year
    print("\nyears with the most No verdicts across artists (all years from 2018):")  # title
    print(per.to_string())  # table
    print(f"\nWHAT Yes (Likely) verdicts {len(ay)}: {rate(ay['what_right'])}; per pool:")  # WHAT Yes
    for p, g in ay.groupby("pool"):  # each pool
        print(f"  pool {p}: {len(g)} Yes verdicts, {rate(g['what_right'])}")  # per pool
    print(ay[cols].to_string(index=False))  # every WHAT Yes row
    ny = t[(t["when_verdict"] == "no") & (t["what_verdict"] == "yes")]  # WHEN No and WHAT Yes
    yy = t[(t["when_verdict"] == "yes") & (t["what_verdict"] == "yes")]  # WHEN Yes and WHAT Yes
    print(f"\nWHEN No and WHAT Yes together: {len(ny)}")  # count
    print(ny[cols].to_string(index=False))  # rows
    print(f"\nWHEN Yes and WHAT Yes together: {len(yy)}: {rate(yy['what_right'])} on WHAT")  # count
    print(yy[cols].to_string(index=False))  # rows


def run() -> None:  # whole step
    """Build the table, refuse to overwrite, write with exclusive creation and print the summary."""
    if OUT_PATH.exists():  # never overwrite
        raise SystemExit(f"{OUT_PATH.relative_to(ROOT)} already exists, nothing was run")  # stop early
    t = build_table()  # the table
    with open(OUT_PATH, "x", encoding="utf-8", newline="") as f:  # exclusive creation
        t.to_csv(f, index=False)  # write
    print(f"{OUT_PATH.relative_to(ROOT)}  {OUT_PATH.stat().st_size / 1024:.1f} KB  {len(t)} rows")  # file
    summary(t)  # counts and lists
