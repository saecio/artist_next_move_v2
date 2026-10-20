"""AppTest run of the proof of concept page app/poc_timeline.py. Prints only, writes no files.

Usage:
    venv/bin/python3 scripts/run_poc_apptest.py
"""

import re  # extraction of cards and badges from the HTML
import sys  # module path
import time  # timings
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from streamlit.testing.v1 import AppTest  # headless app runner

from app_data import load_demo_config  # demo settings
from poc_view import window_years  # window rule under test

cfg = load_demo_config()  # settings
keys = cfg["poc"]["keys"]  # widget keys of the page


def timeline_of(at: AppTest) -> str:  # HTML of the timeline
    """HTML of the timeline element."""
    return next(h.value for h in at.get("html") if "class='poc-line'" in h.value)  # the element that holds the cards


def predicted_card(at: AppTest) -> str:  # HTML of the predicted card
    """HTML of the card of the predicted year."""
    parts = re.split(r"(?=<div class='poc-card)", timeline_of(at))  # one piece per card
    card = next(p for p in parts if p.startswith("<div class='poc-card poc-pred'"))  # the highlighted card
    return card[: -len("</div>")] if parts[-1] is card else card  # the last card also holds the closing tag of the line


def table_of(at: AppTest) -> str:  # HTML of the prediction panel
    """HTML of the prediction panel, without the style block."""
    return next(h.value for h in at.get("html") if "class='poc-panel'" in h.value).split("</style>", 1)[1]  # panel element


def window_of(at: AppTest) -> tuple:  # years drawn and the earlier-years marker
    """Years drawn on the timeline and whether the earlier-years marker is present."""
    h = timeline_of(at)  # timeline element
    return [int(y) for y in re.findall(r"data-year='(\d+)'", h)], "poc-earlier" in h.split("</style>", 1)[1]  # years and marker


def mark_list(card: str) -> list:  # marks of a card
    """(color, hover title, mark) of the marks of a card."""
    return re.findall(r"<span class='poc-mark (\w+)' title='(\w+)'>(.*?)</span>", card)  # color, title and symbol


def new_app(name: str, user_on: bool) -> tuple:  # fresh app with the artist selected
    """Fresh app with the artist selected, optionally with the user prediction on."""
    a = next(x for lst in cfg["pool_b"]["pool_lists"].values() for x in cfg[lst] if x["name"] == name)  # artist entry from either pool
    at = AppTest.from_file(str(ROOT / "app/poc_timeline.py"), default_timeout=60).run()  # first run
    assert len(at.radio) == 0  # no pool selector on the page
    at.selectbox(key=keys["genre"]).select(a["genre"]).run()  # genre
    mixed = sorted((x["name"] for lst in cfg["pool_b"]["pool_lists"].values() for x in cfg[lst] if x["genre"] == a["genre"]), key=str.lower)  # mixed list of the genre, sorted by name
    at.selectbox(key=keys["artist"]).select_index(mixed.index(name)).run()  # artist
    if user_on:  # optional prediction
        at.checkbox(key=keys["user_pred"].format(id=a["discogs_id"])).check().run()  # checkbox
    return at, a["discogs_id"]  # app and id


def press(at: AppTest, aid: int, user_on: bool) -> None:  # press the action button of the step
    """Press the action button of the current step, with a yes answer first when the question is shown."""
    year = at.session_state[keys["state"]][aid]["year"]  # question year
    if user_on and at.session_state[keys["state"]][aid]["step"] == "ask":  # question step
        at.radio(key=keys["answer"].format(id=aid, year=year)).set_value("yes").run()  # answer yes
    at.button(key=keys["action"].format(id=aid)).click().run()  # lock, reveal or next
    assert not at.exception, [e.value for e in at.exception]  # no exception


def walk(name: str, user_on: bool, show: dict, stop_after: int | None = None) -> None:  # play one artist year by year
    """Play one artist year by year; print the badges of the years in show; stop after a year when asked."""
    t0 = time.time()  # clock
    at, aid = new_app(name, user_on)  # app
    print(f"\n===== {name} ({aid}), user prediction {'ON' if user_on else 'OFF'}")  # section title
    while True:  # until the end
        s = at.session_state[keys["state"]][aid]  # state
        year = s["year"]  # question year
        if user_on:  # lock first
            press(at, aid, user_on)  # lock
        if year in show:  # print the card before the reveal
            print(f"   {year} window: {window_of(at)}")  # years drawn and earlier-years marker
            print(f"   {year} before reveal: {mark_list(predicted_card(at))}")  # grey badges
            if show[year] == "html":  # print the card and the table HTML as well
                print("   card HTML before:", predicted_card(at))  # raw card
                print("   table HTML before:", table_of(at))  # raw table
        press(at, aid, False)  # reveal
        print(f"   {year} after reveal : {mark_list(predicted_card(at))}")  # colored badges
        if year in show and show[year] == "html":  # print the card and the table HTML as well
            print("   card HTML after :", predicted_card(at))  # raw card
            print("   table HTML after :", table_of(at))  # raw table
        if stop_after == year or at.button(key=keys["action"].format(id=aid)).disabled:  # stop or end of the data
            break  # done
        press(at, aid, False)  # next year
    assert not at.exception  # still no exception
    print(f"   done in {time.time() - t0:.2f} s, no expander left: {len(at.expander) == 0}")  # timing and the collapsed state


assert window_years(2018, 2010, 2026, 0, 0) == (list(range(2010, 2027)), False)  # 0 and 0 mean all years, nothing cut
assert window_years(2012, 2010, 2026, 5, 1) == ([2010, 2011, 2012, 2013], False)  # near the start nothing is cut
assert window_years(2018, 2010, 2026, 5, 1) == ([2013, 2014, 2015, 2016, 2017, 2018, 2019], True)  # earlier years cut
print("window rule checks passed")  # all asserts held
t_all = time.time()  # total clock
for on in (False, True):  # both modes
    walk("Taylor Swift", on, {2018: "html", 2020: None} if not on else {2018: "html", 2020: None}, stop_after=2021)  # 2018 to 2021, 2018 table shown
    walk("Olivia Rodrigo", on, {2023: None, 2024: None, 2026: None})  # 2019 to 2024 and 2026 (flagged and last data year)
walk("Lady Gaga", False, {2021: None}, stop_after=2021)  # WHAT Yes verdict
walk("Taylor Swift", False, {2024: None}, stop_after=2024)  # flagged year
from app_live import artist_timeline  # live data layer, to choose years that exist
from demo_game import scored_years  # playable years
for who in ("Kendrick Lamar", "Beyonce", "Peso Pluma", "boygenius"):  # pool B artists
    aid = next(x["discogs_id"] for x in cfg["artists_b"] if x["name"] == who)  # id
    tl = artist_timeline(aid)  # timeline
    playable = scored_years(tl, 2018)  # playable years
    walk(who, False, {playable[0]: None}, stop_after=playable[-1])  # play every playable year, show the first
    at, _ = new_app(who, False)  # fresh app for the table check
    while at.session_state[keys["state"]][aid]["year"] != playable[0]:  # walk to the first year
        press(at, aid, False)  # reveal
        press(at, aid, False)  # next
    press(at, aid, False)  # reveal
    if who == "boygenius":  # no WHAT score in any year
        assert cfg["poc"]["texts"]["no_what_score"] in table_of(at), "boygenius WHAT row must say there is no WHAT score"  # poc text of the missing score
        print(f"   boygenius {playable[0]}: WHAT row shows: {cfg['poc']['texts']['no_what_score']}")  # printed proof
legend = cfg["poc"]["texts"]["legend_reference"]  # fixed legend text
at, aid = new_app("Taylor Swift", False)  # flagged-year check
while at.session_state[keys["state"]][aid]["year"] != 2024:  # walk to 2024
    press(at, aid, False)  # reveal
    press(at, aid, False)  # next
press(at, aid, False)  # reveal
panel = table_of(at)  # panel HTML
cells = re.findall(r"<td class='poc-cell (\w+)'>(.*?)</td>", panel)  # result cells
print("flagged year 2024 result cells:", cells, "| legend present:", legend in panel)  # printed proof
assert all(c in ("green", "red") and m in ("✓*", "✗*") and (c == "green") == (m == "✓*") for c, m in cells), "flagged results must be green or red with the asterisk"  # color and asterisk
assert "amber" not in panel  # no amber left
assert legend in panel and "⏳" not in panel  # legend present, hourglass gone
print("\n== shared view functions of tab 6 (reality cells, year-end tag, side panel)")  # section title
import html  # escaping in the expected strings
import pandas as pd  # synthetic rows
import poc_view  # module under test
from poc_view import reality_item, side_panel_html, year_end_tag  # functions under test

route_cfg = cfg["route"]  # route settings
rt = {**route_cfg["texts"], "title_max_chars": route_cfg["title_max_chars"]}  # texts with the title limit
yn = {"yes": cfg["game"]["answer_yes"], "no": cfg["game"]["answer_no"]}  # Yes / No words
limit = rt["title_max_chars"]  # longest title before the ellipsis
long_title = "A" * (limit + 5) + " & more"  # a title above the limit with a character that needs escaping
cell = reality_item("yes", long_title, yn, rt)  # Yes with a long title
assert ("Yes: " + "A" * (limit - 1) + "\u2026") in cell and f"title='{html.escape(long_title, quote=True)}'" in cell, "a long title is cut with an ellipsis and keeps the full title in the tooltip"  # truncation and tooltip
short = reality_item("yes", "Short", yn, rt)  # a short title
assert "Yes: Short" in short and "\u2026" not in short, "a short title is not cut"  # no ellipsis
assert reality_item("yes", None, yn, rt) == "Yes" and reality_item("yes", float("nan"), yn, rt) == "Yes", "a Yes without a title is just Yes"  # no crash, no title
assert reality_item("no", "Ignored", yn, rt) == "No" and reality_item(None, None, yn, rt) == html.escape(rt["no_reality"]), "No and missing reality carry no title"  # other cases
assert year_end_tag(rt, 2019) == "No release in 2019 \u00b7 estimated from history", "the tag text"  # tag
assert not hasattr(poc_view, "release_cell") and "col_release" not in rt, "the Release column code and keys are gone"  # removed
row = pd.Series({"year": 2019, "is_year_end": True, "when_verdict": "no", "when2_verdict": "yes", "when_label": "no", "when2_label": "yes", "when_title": None, "when2_title": long_title, "show2": True})  # synthetic year-end row
model_only, full = side_panel_html(row, False, cfg["game"], route_cfg, cfg["poc"]), side_panel_html(row, True, cfg["game"], route_cfg, cfg["poc"])  # panel before and after the reveal
assert year_end_tag(rt, 2019) in model_only and model_only.count(rt["pill_next"]) == 1 and model_only.count(rt["pill_two"]) == 1 and "title=" not in model_only, "model cascade with the tag and no reality"  # before the reveal
assert full.count(rt["pill_next"]) == 2 and full.count(rt["pill_two"]) == 2 and "\u2026" in full, "reality cascade has the same structure, with the cut title"  # after the reveal
print("reality items, truncation and tooltip, year-end tag, side panel cascade checked")  # result
print(f"\ntotal time {time.time() - t_all:.2f} s")  # total
