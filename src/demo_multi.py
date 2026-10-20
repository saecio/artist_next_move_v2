"""Mode 2 of tab 6: several artists in one year as a read-only table. Pure logic, no Streamlit code."""

import json  # style lists stored as JSON text

import pandas as pd  # tables

from app_data import all_artists, artists_by_genre  # mixed artist list and genre filter
from app_live import load_tables, when_rows  # cached tables and WHEN rows of an artist-year, as tab 6 reads them
from demo_game import outside_text_key, reality_status, reliability, scored_years, verdict  # same rules as mode 1
from poc_view import badge_kind, what_parts  # same marks and WHAT parts as the proof of concept


def route_ids(cfg: dict) -> list:  # ids of the prepared selection
    """Discogs ids of route.artists in config order; every id must be an artist of the config."""
    known = {int(a["discogs_id"]) for a in all_artists(cfg)}  # ids of both pools
    ids = [int(i) for i in cfg["route"]["artists"]]  # prepared ids
    missing = [i for i in ids if i not in known]  # ids that are not in the config
    assert not missing, f"route.artists has ids that are not demo artists: {missing}"  # fail early
    return ids  # config order


def picker_options(cfg: dict, genre: str | None, selected: list) -> list:  # ids offered by the multiselect
    """Ids of the artists of the genre plus the ones already selected, so a genre change never drops a selection; sorted by name."""
    pool = {int(a["discogs_id"]): a for a in all_artists(cfg)}  # id -> artist
    keep = {int(a["discogs_id"]) for a in artists_by_genre(cfg, genre)} | {int(i) for i in selected}  # genre plus selected
    return sorted(keep, key=lambda i: pool[i]["name"].lower())  # sorted by name


def pick_title(picks: pd.DataFrame, artist_id: int, year: int) -> str | None:  # release title of an artist-year
    """Title chosen by rule B for the artist-year (results/demo/release_pick.csv), None when the file has no title for it."""
    m = picks[(picks["artist_id"] == artist_id) & (picks["year"] == year)]  # the pick of the artist-year, at most one row
    return None if m.empty or pd.isna(m.iloc[0]["title"]) else str(m.iloc[0]["title"])  # title text or none


def reality_titles(picks: pd.DataFrame, artist_id: int, year: int, label1: str | None, label2: str | None) -> tuple:  # titles of the two reality cells
    """(next-year title, within-2-years title): the next-year cell uses year + 1; the 2-year cell uses the year of the release, year + 1 when the next-year label is Yes, else year + 2. None where the reality is not Yes or the file has no title."""
    t1 = pick_title(picks, artist_id, year + 1) if label1 == "yes" else None  # next-year release
    t2 = pick_title(picks, artist_id, year + 1 if label1 == "yes" else year + 2) if label2 == "yes" else None  # release inside the 2 years
    return t1, t2  # both titles


def pick_styles(picks: pd.DataFrame, works: pd.DataFrame, artist_id: int, year: int) -> list:  # styles of the picked release of an artist-year
    """Style names of the Work chosen by rule B for the artist-year, [] when there is no pick or the Work has no styles."""
    m = picks[(picks["artist_id"] == artist_id) & (picks["year"] == year)]  # pick of the artist-year
    if m.empty:  # no pick
        return []  # no styles
    w = works[works["work_id"] == m.iloc[0]["work_id"]]  # the picked Work
    return json.loads(w.iloc[0]["styles"]) if len(w) else []  # its styles, stored as JSON text


def new_styles_of(nw: pd.DataFrame, artist_id: int, year: int) -> tuple:  # next active year and new styles of an artist-year
    """(next active year or None, list of new styles) from results/demo/what_new_styles.csv; (None, []) when the file has no row."""
    m = nw[(nw["artist_id"] == artist_id) & (nw["year"] == year)]  # row of the artist-year
    if m.empty or pd.isna(m.iloc[0]["next_active_year"]):  # no row, or no next active year
        return None, []  # nothing to show
    return int(m.iloc[0]["next_active_year"]), list(m.iloc[0]["new_styles"])  # year and styles


def playable_years(artist_id: int, timeline: pd.DataFrame, first_year: int) -> list:  # years of mode 1
    """Playable years of mode 1: the years with a scored Work plus the year-end years (no Work, stored year-end score) from first_year on, ascending."""
    ye = load_tables()["when_year_end"]  # stored year-end scores
    gap = ye[(ye["artist_id"] == artist_id) & ye["is_year_end"] & (ye["snapshot_year"] >= first_year)]["snapshot_year"]  # year-end years of the artist
    return sorted(set(scored_years(timeline, first_year)) | {int(y) for y in gap})  # Work years and year-end years


def year_end_what(artist_id: int, year: int, cutoff: float) -> dict:  # WHAT fields of a year-end row
    """WHAT fields of a year-end row from results/demo/what_scores_year_end.csv: with a score and a resolved label the verdict and label are shown, never counted and never marked with an asterisk (what_counted and what_star False); without a stored row what_na is 'needs_three' and the grey dash shows; a stored score with no resolved label shows the verdict with a grey dot and the Style reality line says there is no later release yet."""
    t = load_tables()["what_year_end"]  # stored year-end WHAT scores
    m = t[(t["artist_id"] == artist_id) & (t["year"] == year)]  # row of the artist-year
    off = {"what_available": False, "what_verdict": None, "what_label": None, "what_counted": False, "what_star": False, "what_score": None, "what_from": year}  # no WHAT shown
    if m.empty:  # fewer than 3 earlier Works
        return {**off, "what_na": "needs_three"}  # no score
    r = m.iloc[0]  # the stored row
    what, wlab, _ = what_parts(r, {"score_cutoff": cutoff})  # verdict and label text (None without a resolved label), same rule as the Work rows
    return {"what_available": True, "what_verdict": what, "what_label": wlab, "what_counted": False, "what_star": False, "what_score": r["stage0_score"], "what_na": None, "what_from": int(r["last_active_year"])}  # the label copies the last active year, so it is never counted; label may be None


def play_year(r: pd.Series, answer: str | None) -> dict:  # record one played year of mode 1
    """One history entry from a multi_rows row: user answer, model verdict, next-year reality and whether it counts (the when_counted flag, same rule as demo_game.is_counted)."""
    return {"year": int(r["year"]), "user": answer, "model": r["when_verdict"], "label": text_or_none(r["when_label"]), "counted": bool(r["when_counted"])}  # history entry


def multi_rows(cfg: dict, artist_ids: list, year: int, threshold: float, cutoff: float, cap: int, stage: str = "reveal") -> pd.DataFrame:  # one row per artist with a row
    """Rows of the table for one year, at most `cap` artists. An artist with a Work that year shows its representative Work row, one without shows its year-end row, one with neither is left out (see hidden_counts). From the predict stage on the rows are sorted by the next-year score descending (ties by name); in the placeholder stage they keep the selection order, so the order does not give the scores away."""
    names = {int(a["discogs_id"]): a["name"] for a in all_artists(cfg)}  # id -> name
    rows = []  # table rows
    forecast = year >= int(load_tables()["works"]["year"].max())  # the last data year: its next year has not happened
    for aid in [int(i) for i in artist_ids][:cap]:  # the cap applies to the number of artists
        w = when_rows(aid, year)  # WHEN rows of the artist-year
        if w is None:  # no row to show
            continue  # left out of the table
        h1, h2 = w["next_1y"], w["next_2y"]  # both horizons
        v1, v2 = verdict(h1["score"], threshold), verdict(h2["score"], threshold)  # model answers
        t1, t2 = reality_titles(load_tables()["release_pick"], aid, year, h1["label"], h2["label"])  # release titles of the two reality cells
        out = {"artist": names[aid], "artist_id": aid, "year": year, "has_row": True, "is_year_end": w["kind"] == "year_end", "is_forecast": forecast, "when_title": t1, "when2_title": t2}  # identity, forecast flag and reality titles
        out.update(when_score=h1["score"], when_verdict=v1, when_label=h1["label"], when_counted=h1["counted"], when_star=h1["star"], when2_score=h2["score"], when2_verdict=v2, when2_label=h2["label"], when2_counted=h2["counted"], when2_star=h2["star"], show2=v1 == "no" or v2 == "no")  # both horizons and the cascade rule
        if w["kind"] == "work":  # a Work year: WHAT
            what, wlab, _ = what_parts(w["row"], {"score_cutoff": cutoff})  # WHAT verdict and label
            out.update(what_available=what is not None, what_verdict=what, what_label=wlab, what_score=w["row"]["stage0_score"])  # WHAT fields, the score is only shown in a tooltip
            out.update(what_na=None, what_from=year)  # the default tooltip applies, new styles of this year
        else:  # a year-end row: WHAT from the stored year-end score, shown but never counted
            out.update(year_end_what(aid, year, cutoff))  # WHAT fields of the year-end row
        ny, new = new_styles_of(load_tables()["what_new_styles"], aid, out["what_from"])  # next active year and new styles of the artist-year (of its last active year for a year-end row)
        status = "none" if w["kind"] == "year_end" else reality_status(out["what_label"], ny)  # WHAT counts by the year of the next release, never for a year-end row
        out.update(what_counted=status == "counted", what_star=status == "open")  # counted, or the asterisk when that release is in the year not over
        out.update(what_year=ny, what_new=new, what_release_title=None if ny is None else pick_title(load_tables()["release_pick"], aid, ny), what_styles=[] if ny is None else pick_styles(load_tables()["release_pick"], load_tables()["works"], aid, ny))  # release the WHAT label refers to: year, new styles, rule B title and styles
        rows.append(out)  # one row per artist
    t = pd.DataFrame(rows)  # the table
    if t.empty or stage == "placeholder":  # nothing to show, or the order must not show the scores
        return t  # selection order
    return t.assign(_s=t["when_score"], _n=t["artist"].str.lower()).sort_values(["_s", "_n"], ascending=[False, True]).drop(columns=["_s", "_n"]).reset_index(drop=True)  # score descending, then name

def hidden_counts(artist_ids: list, year: int, cap: int) -> dict:  # artists left out of the table, by case
    """Counts of the selected artists (within the cap) with no row in the year: 'none' have no dated Work before the year, 'yet' have earlier Works but no row (a year with no Work and no year-end row, the last data year)."""
    tables = load_tables()  # cached tables
    works = tables["works"]  # Works of both pools, dated from the first shown year
    scored = pd.concat([tables["when_stored"], tables["when_year_end"]])  # every stored WHEN row, Work rows and year-end rows, from 2018
    counts = {"none": 0, "yet": 0}  # one counter per case
    for aid in [int(i) for i in artist_ids][:cap]:  # the cap applies to the number of artists
        if when_rows(aid, year) is not None:  # the artist has a row
            continue  # not hidden
        earlier = ((works["artist_id"] == aid) & (works["year"] < year)).any() or ((scored["artist_id"] == aid) & (scored["snapshot_year"] < year)).any()  # a Work or a stored WHEN row before the year (an artist with history has a year-end row in every gap year from 2018)
        counts["yet" if earlier else "none"] += 1  # case of the hidden artist
    return counts  # both counts

def hidden_lines(cfg: dict, artist_ids: list, year: int) -> list:  # lines under the table
    """The hidden-artists lines from the yaml texts: 'no earlier releases before {year}' for artists with no history, 'no release yet in {year}' (last data year) for artists with history but no row; singular for 1, empty when nothing is hidden."""
    tx = cfg["route"]["texts"]  # route texts
    c = hidden_counts(artist_ids, year, cfg["route"]["max_artists"])  # counts by case
    pick = lambda n, base: tx[f"{base}_one" if n == 1 else f"{base}_many"].format(n=n, year=year)  # wording variant
    return ([pick(c["none"], "no_history")] if c["none"] else []) + ([pick(c["yet"], "no_release")] if c["yet"] else [])  # none-case line first

def text_or_none(x):  # a missing label as None
    """The label text, or None when it is missing (a DataFrame turns None into NaN)."""
    return None if pd.isna(x) else x  # None for NaN

def row_marks(row: pd.Series) -> list:  # marks of one row
    """(color, mark key) of the shown results in display order: next year, within 2 years when that pill is shown, then WHAT; grey when there is no reality, green or red, with an asterisk when the reality is in the year not over."""
    out = [badge_kind(True, text_or_none(row["when_label"]), not row["when_star"], row["when_verdict"])]  # next-year mark
    if row["show2"]:  # the second pill is shown
        out.append(badge_kind(True, text_or_none(row["when2_label"]), not row["when2_star"], row["when2_verdict"]))  # 2-year mark
    out.append(("grey", "mark_dot") if not row["what_available"] else badge_kind(True, text_or_none(row["what_label"]), not row["what_star"], row["what_verdict"]))  # WHAT mark
    return out  # marks in display order

def multi_summary(rows: pd.DataFrame) -> dict:  # model right in k of N
    """Counts over rows with a counted reality: next year needs a counted label, within 2 years only counts where that pill is shown, WHAT needs a counted label (a release in a complete year, never a year-end row)."""
    ok = rows[rows["has_row"]] if len(rows) else rows  # rows shown
    w = ok[ok["when_label"].notna() & ok["when_counted"]] if len(ok) else ok  # counted next-year rows
    w2 = ok[ok["show2"] & ok["when2_label"].notna() & ok["when2_counted"]] if len(ok) else ok  # counted 2-year rows where the pill is shown
    a = ok[ok["what_available"] & ok["what_label"].notna() & ok["what_counted"]] if len(ok) else ok  # counted WHAT rows
    return {"when_right": int((w["when_verdict"] == w["when_label"]).sum()) if len(w) else 0, "when_n": len(w), "when2_right": int((w2["when2_verdict"] == w2["when2_label"]).sum()) if len(w2) else 0, "when2_n": len(w2), "what_right": int((a["what_verdict"] == a["what_label"]).sum()) if len(a) else 0, "what_n": len(a)}  # k and N

def reliability_lines(top: dict, below: dict, g: dict) -> list:  # the two reliability lines
    """The Yes and No reliability lines of mode 1, built from the holdout figures."""
    scale = g["reliability_scale"]  # the rates are out of this many artists
    x, y = reliability(top, scale)  # rounded rates out of the scale
    return [g["what_reliability"].format(x=x, y=y, scale=scale), g[outside_text_key(below["z"], y, g["similar_gap"])].format(z=below["z"], y=y, scale=scale)]  # Yes line, No line


STAGES = ["placeholder", "predict", "reveal"]  # display stages in order


def multi_signature(year: int, ids: list) -> tuple:  # what the two-step state belongs to
    """Year and selected ids; the state resets when either changes."""
    return (int(year), tuple(int(i) for i in ids))  # hashable and comparable


def advance_stage(state: dict | None, signature: tuple, action: str | None) -> dict:  # the two-step state machine
    """New state: reset to the placeholder stage when the signature changed, then move one step on 'predict' (from placeholder) or 'reveal' (from predict)."""
    if state is None or state["signature"] != signature:  # first run or a change of year or artists
        state = {"signature": signature, "stage": "placeholder"}  # reset
    stage = state["stage"]  # current stage
    if action == "predict" and stage == "placeholder":  # step 1
        stage = "predict"  # show the model verdicts
    elif action == "reveal" and stage == "predict":  # step 2, only after predict
        stage = "reveal"  # show the realities
    return {"signature": signature, "stage": stage}  # new state


def pick_route(cfg: dict, year: int, threshold: float) -> list:  # prepared selection of a year by a fixed rule
    """Ids for `year` by the rule of the 2019 route (route.mix_yes Yes verdicts, the rest No, at most route.max_artists): among the artists with a Work that year, Yes verdicts of the next-year score first, then No verdicts, each group in ascending Discogs id; when the No group is short the Yes group fills the places."""
    rt = cfg["route"]  # route settings
    ids = sorted(int(a["discogs_id"]) for a in all_artists(cfg))  # both pools, ascending id
    rows = {i: when_rows(i, year) for i in ids}  # WHEN rows of the year
    v = {i: verdict(w["next_1y"]["score"], threshold) for i, w in rows.items() if w is not None and w["kind"] == "work"}  # verdict of the artists with a Work that year
    yes, no = [i for i in ids if v.get(i) == "yes"], [i for i in ids if v.get(i) == "no"]  # verdict groups
    n_no = min(len(no), rt["max_artists"] - rt["mix_yes"])  # No places, all the No artists when fewer
    return yes[:rt["max_artists"] - n_no] + no[:n_no]  # Yes group, then No group


def route_for_year(cfg: dict, year: int, current: list) -> list:  # selection after a change of year
    """The prepared list of the year when route.artists_by_year has one; the default list when the current selection is the prepared list of another year; else the current selection unchanged."""
    by_year = {int(y): [int(i) for i in ids] for y, ids in cfg["route"]["artists_by_year"].items()}  # prepared lists by year
    if year in by_year:  # a prepared list for this year
        return by_year[year]  # use it
    if [int(i) for i in current] in by_year.values():  # coming from a prepared year
        return route_ids(cfg)  # back to the default list
    return [int(i) for i in current]  # manual selection stays
