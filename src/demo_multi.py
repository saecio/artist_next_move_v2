"""Mode 2 of tab 6: several artists in one year as a read-only table. Pure logic, no Streamlit code."""

import pandas as pd  # tables

from app_data import all_artists, artists_by_genre  # mixed artist list and genre filter
from app_live import artist_timeline  # live scores and stored labels, as tab 6 reads them
from demo_game import is_counted, outside_text_key, reliability, verdict  # same rules as mode 1
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


def multi_rows(cfg: dict, artist_ids: list, year: int, threshold: float, cutoff: float, cap: int, stage: str = "reveal") -> pd.DataFrame:  # one row per artist
    """Rows of the table for one year, at most `cap` artists. From the predict stage on they are sorted by WHEN score descending (ties by name, artists without a score last); in the placeholder stage they keep the selection order, so the order does not give the scores away."""
    names = {int(a["discogs_id"]): a["name"] for a in all_artists(cfg)}  # id -> name
    rows = []  # table rows
    for aid in [int(i) for i in artist_ids][:cap]:  # the cap applies to the number of artists
        tl = artist_timeline(aid)  # timeline, representative Work per year
        r = tl[tl["year"] == year]  # row of the year, empty when the year is outside the timeline
        r = r.iloc[0] if len(r) else None  # row or none
        has = r is not None and pd.notna(r["next_1y"])  # a WHEN score exists for this artist-year
        out = {"artist": names[aid], "artist_id": aid, "has_row": bool(has)}  # identity
        if has:  # score, verdicts and realities
            lab = r["label_next_1y"]  # stored WHEN label
            what, wlab, safe = what_parts(r, {"score_cutoff": cutoff})  # WHAT verdict, label and flag
            out.update(when_score=float(r["next_1y"]), when_verdict=verdict(r["next_1y"], threshold), when_label=None if pd.isna(lab) else ("yes" if lab == 1 else "no"), when_counted=bool(is_counted(r)), what_available=what is not None, what_verdict=what, what_label=wlab, what_safe=bool(safe))  # all fields
        rows.append(out)  # one row per artist
    t = pd.DataFrame(rows)  # the table
    if t.empty or stage == "placeholder":  # nothing selected, or the order must not show the scores
        return t  # selection order
    return t.assign(_s=t.get("when_score").fillna(-1.0), _n=t["artist"].str.lower()).sort_values(["_s", "_n"], ascending=[False, True]).drop(columns=["_s", "_n"]).reset_index(drop=True)  # score descending, then name


def text_or_none(x):  # a missing label as None
    """The label text, or None when it is missing (a DataFrame turns None into NaN)."""
    return None if pd.isna(x) else x  # None for NaN


def row_marks(row: pd.Series) -> dict:  # marks of one row
    """(color, mark key) of the WHEN and WHAT results of a row; grey when there is no reality, green or red with an asterisk when it is not counted."""
    when = badge_kind(True, text_or_none(row["when_label"]), row["when_counted"], row["when_verdict"])  # WHEN mark
    what = ("grey", "mark_dot") if not row["what_available"] else badge_kind(True, text_or_none(row["what_label"]), row["what_safe"], row["what_verdict"])  # WHAT mark
    return {"when": when, "what": what}  # both marks


def multi_summary(rows: pd.DataFrame) -> dict:  # model right in k of N
    """Counts over rows with a counted reality: WHEN needs a counted label, WHAT needs a label that is not flagged."""
    ok = rows[rows["has_row"]] if len(rows) else rows  # rows with a score
    w = ok[ok["when_label"].notna() & ok["when_counted"]] if len(ok) else ok  # counted WHEN rows
    a = ok[ok["what_available"] & ok["what_label"].notna() & ok["what_safe"]] if len(ok) else ok  # counted WHAT rows
    return {"when_right": int((w["when_verdict"] == w["when_label"]).sum()) if len(w) else 0, "when_n": len(w), "what_right": int((a["what_verdict"] == a["what_label"]).sum()) if len(a) else 0, "what_n": len(a)}  # k and N


def reliability_lines(top: dict, below: dict, g: dict) -> list:  # the two reliability lines
    """The Yes and No reliability lines of mode 1, built from the holdout figures."""
    x, y = reliability(top)  # rounded rates out of 10
    return [g["what_reliability"].format(x=x, y=y), g[outside_text_key(below["z"], y, g["similar_gap"])].format(z=below["z"], y=y)]  # Yes line, No line


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
