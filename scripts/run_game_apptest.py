"""Smoke run of the tab 6 game with streamlit.testing.v1.AppTest, both modes. Prints only, writes no files.

Usage:
    venv/bin/python3 scripts/run_game_apptest.py
"""

import csv  # independent recomputation, no pandas
import sys  # module path
import time  # timings
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from streamlit.testing.v1 import AppTest  # headless app runner

from app_data import load_demo_config  # demo settings
from config import CONFIG  # project settings
from demo_game import scoreboard  # scoreboard of a history

cfg = load_demo_config()  # settings
CFG_RT = cfg["route"]["texts"]  # route texts: forecast title
last_complete_year = CONFIG["years"]["last_complete"]  # last year whose results count
g, keys = cfg["game"], cfg["game"]["keys"]  # game texts and keys
import html  # unescape of HTML texts
import re  # tags of the HTML
import pandas as pd  # tables
poc_tx = cfg["poc"]["texts"]  # poc texts: grey dash, marks, pills
gap = {r["target"]: round(100 * float(r["ece"])) for r in csv.DictReader(open(ROOT / "results/calibration/summary.csv"))}  # average score gap per target in whole points, read independently of the app
gap_text = lambda n: cfg["ui"]["point_one" if n == 1 else "point_many"].format(n=n)  # singular for one point, plural otherwise


def na_shown(at: AppTest) -> bool:  # the grey dash of a row without a WHAT score
    """Whether the page holds the grey dash with its tooltip."""
    return any(f"title='{poc_tx['what_na_tip']}'" in h.value for h in at.get("html"))  # dash element
CONFIG_MIN = CONFIG["what"]["min_previous_works"]  # earlier Works required for a WHAT score
with open(ROOT / "results/holdout_what/top10_lift.csv", encoding="utf-8") as f:  # top 10% file
    top = next(r for r in csv.DictReader(f) if r["model"] == "stage 0")  # stage 0 row
with open(ROOT / "results/holdout_what/row_scores.csv", encoding="utf-8") as f:  # holdout scores
    low = [float(r["label"]) for r in csv.DictReader(f) if float(r["stage0_score"]) < float(top["score_cutoff"])]  # labels of the rows below the cutoff
EXP = {"x": round(float(top["precision_top"]) * g["reliability_scale"]), "y": round(float(top["holdout_positive_rate"]) * g["reliability_scale"]), "z": round(sum(low) / len(low) * g["reliability_scale"])}  # independent x, y and z


def find(name: str) -> tuple:  # artist entry and pool key
    """Artist entry and pool key of a display name, searching both pools (the pool is not on screen)."""
    for pool, lst in cfg["pool_b"]["pool_lists"].items():  # each pool and its config list
        for x in cfg[lst]:  # artists of the pool
            if x["name"] == name:  # found
                return x, pool  # entry and pool


def pick(at: AppTest, name: str) -> int:  # select genre and artist from the mixed list
    """Select the genre and the artist from the mixed, name-sorted list; no pool widget exists."""
    assert len(at.radio) == 0 or all(r.label != "Artists" for r in at.radio)  # no pool selector on the page
    at.radio(key=keys["mode"]).set_value("one").run()  # select mode 1 explicitly (it is also the view tab 6 opens on)
    a, _ = find(name)  # entry
    at.selectbox(key=keys["genre"]).select(a["genre"]).run()  # genre filter
    names = sorted((x["name"] for lst in cfg["pool_b"]["pool_lists"].values() for x in cfg[lst] if x["genre"] == a["genre"]), key=str.lower)  # mixed list of the genre, sorted by name
    shown = [o.name if hasattr(o, "name") else str(o) for o in at.selectbox(key=keys["artist"]).options]  # options shown, as text
    assert len(shown) == len(names) and shown == sorted(shown, key=str.lower), "the artist list must be mixed and sorted by name"  # one mixed sorted list
    at.selectbox(key=keys["artist"]).select_index(names.index(name)).run()  # artist
    return a["discogs_id"]  # id


def show(at: AppTest, label: str) -> None:  # print the main texts of the page
    """Assert no exception and print the main texts of the page."""
    assert not at.exception, [e.value for e in at.exception]  # no exception on the page
    keep = ("Model", "My answer", "Reality", "Information", "**WHEN", "**Style", "Style", "In past tests", "This year may", "No WHAT", "No reality", "Last year", "User", "Session")  # line prefixes of the game texts
    lines = [m.value for m in list(at.markdown) + list(at.caption) + list(at.info) if m.value and (m.value.startswith(keep) or "hidden" in m.value)]  # game lines and hidden-count captions
    print(f"-- {label}: {len(at.exception)} exceptions, {len(at.button)} buttons")  # summary
    for s in at.subheader:  # section titles
        print("   #", s.value)  # print the result
    for tx in lines:  # game lines
        print("   ", tx[:120])  # print the result


def run_mode(user_on: bool, artist_name: str, steps: int) -> None:  # play the first years of one artist
    """Fresh app, pick an artist, play the first `steps` years."""
    aid = find(artist_name)[0]["discogs_id"]  # its id
    t0 = time.time()  # first run clock
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # first run
    print(f"\n===== {artist_name} ({aid}), user prediction {'ON' if user_on else 'OFF'}; first run {time.time() - t0:.2f} s")  # print the result
    pick(at, artist_name)  # pool, genre and artist
    if user_on:  # enable the optional prediction
        at.checkbox(key=keys["user_pred"].format(id=aid)).check().run()  # checkbox
    for n in range(steps):  # play the first years
        t1 = time.time()  # step clock
        if user_on:  # question and lock first
            at.radio(key=keys["answer"].format(id=aid, year=at.session_state[keys["state"]][aid]["year"])).set_value("yes").run()  # answer yes
            at.button(key=keys["lock"].format(id=aid)).click().run()  # lock
        if n == 0:  # print step A of the first year
            show(at, "step A")  # print the page state
        at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
        if n == steps - 1:  # print step B of the last year
            show(at, "step B")  # print the page state
        at.button(key=keys["next"].format(id=aid)).click().run()  # next year
        print(f"   step {n + 1} done in {time.time() - t1:.2f} s, state year {at.session_state[keys['state']][aid]['year']}, history {len(at.session_state[keys['state']][aid]['history'])}")  # print the result
    assert not at.exception  # still no exception
    if user_on:  # mid-game switch off keeps the history
        at.checkbox(key=keys["user_pred"].format(id=aid)).uncheck().run()  # untick
        h = at.session_state[keys["state"]][aid]["history"]  # history kept
        print("   after unticking, history user fields:", [x["user"] for x in h])  # print the result
    at.button(key=keys["restart"].format(id=aid)).click().run()  # restart
    print("   after restart, history length:", len(at.session_state[keys["state"]][aid]["history"]))  # print the result
    assert not at.exception  # no exception after restart


def play_to_end(artist_name: str, user_on: bool) -> None:  # play one artist to the last year
    """Play every year of one artist; print the page at 2024 and 2026 and assert the scoreboard n does not change there."""
    aid = find(artist_name)[0]["discogs_id"]  # its id
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # fresh app
    pick(at, artist_name)  # pool, genre and artist
    if user_on:  # enable the optional prediction
        at.checkbox(key=keys["user_pred"].format(id=aid)).check().run()  # checkbox
    print(f"\n===== full walk, {artist_name} ({aid}), user prediction {'ON' if user_on else 'OFF'}")  # section title
    while True:  # until the last year
        state = at.session_state[keys["state"]][aid]  # current state
        year = state["year"]  # question year
        if user_on:  # question and lock first
            at.radio(key=keys["answer"].format(id=aid, year=year)).set_value("yes").run()  # answer yes
            at.button(key=keys["lock"].format(id=aid)).click().run()  # lock
        n_before = scoreboard(state["history"])  # board before the reveal
        rev = [b for b in at.button if b.key == keys["reveal"].format(id=aid)]  # reveal button of the question
        if year >= CONFIG["years"]["partial"]:  # forecast year: no reality, no reveal step
            assert not rev, "no reveal button in the forecast year"  # no reveal
            assert any(CFG_RT["forecast_title"].format(year=year + 1) in i.value for i in at.info), "the forecast title is on screen"  # title
            n_after = n_before  # nothing recorded
        else:  # a normal year
            rev[0].click().run()  # reveal
            n_after = scoreboard(at.session_state[keys["state"]][aid]["history"])  # board after the reveal
            last = at.session_state[keys["state"]][aid]["history"][-1]  # entry of the year
            assert last["counted"] == (year + 1 <= CONFIG["years"]["last_complete"] and last["label"] is not None), f"counted flag of {year} follows the reality year"  # rule
            assert n_after["n"] == n_before["n"] + int(last["counted"]), f"board of {year} follows the counted flag"  # board
            if year == last_complete_year and last["label"] is not None:  # a reality in 2026: asterisk mark and its legend near the scoreboard
                assert poc_tx["legend_open_year"].format(year=CONFIG["years"]["last_complete"] + 1) in [c.value for c in at.caption], f"{year}: the open-year legend is on screen"  # legend
        if year in (2025, 2026):  # reality in 2026 or later
            assert n_before["n"] == n_after["n"] and n_before["user_n"] == n_after["user_n"], f"board changed in {year}"  # no change
        if year in (2024, 2025, 2026):  # requested years
            show(at, f"year {year}, n {n_before['n']} -> {n_after['n']}")  # page lines
        assert not at.exception  # no exception
        nxt = at.button(key=keys["next"].format(id=aid))  # next year button
        if nxt.disabled:  # end of the data
            break  # stop
        nxt.click().run()  # next year
    print(f"   last year played {year}, final n {n_after['n']}")  # end of the walk


def show_year(artist_name: str, target_year: int) -> None:  # play one artist up to a year and print both columns
    """Play one artist up to target_year, print the page lines of the WHEN and WHAT columns and check that WHAT never moves the scoreboard."""
    aid = find(artist_name)[0]["discogs_id"]  # its id
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # fresh app
    pick(at, artist_name)  # pool, genre and artist
    while at.session_state[keys["state"]][aid]["year"] != target_year:  # walk to the target year
        at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
        at.button(key=keys["next"].format(id=aid)).click().run()  # next year
    before = scoreboard(at.session_state[keys["state"]][aid]["history"])  # board before the target year
    show(at, f"{artist_name} {target_year}, step A")  # page lines of step A
    at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
    played = at.session_state[keys["state"]][aid]["history"][-1]  # entry of the target year
    after = scoreboard(at.session_state[keys["state"]][aid]["history"])  # board after the target year
    show(at, f"{artist_name} {target_year}, step B, WHEN counted {played['counted']}, n {before['n']} -> {after['n']}")  # page lines of step B
    assert after["n"] - before["n"] == int(played["counted"]), "the scoreboard changed for a reason other than the WHEN label"  # WHAT adds nothing
    texts = [m.value for m in at.markdown]  # every written line
    captions = [c.value for c in at.caption]  # every caption on the page
    pills = [h.value for h in at.get("html") if f">{poc_tx['likely']}<" in h.value or f">{poc_tx['not_likely']}<" in h.value]  # the WHAT verdict pill
    if pills:  # the WHAT column has a verdict
        yes = f">{poc_tx['likely']}<" in pills[0]  # verdict of this year
        want = g["what_reliability"].format(x=EXP["x"], y=EXP["y"], scale=g["reliability_scale"]) if yes else g["what_reliability_no_similar" if abs(EXP["z"] - EXP["y"]) <= 1 else "what_reliability_no_other"].format(z=EXP["z"], y=EXP["y"], scale=g["reliability_scale"])  # expected line from the independent numbers
        assert want in captions, f"the reliability line is not the expected one: {want}"  # the line depends on the verdict
        print(f"   reliability line checked ({'Yes' if yes else 'No'} verdict): {want}")  # shown for the record
    assert not at.exception  # no exception


t_all = time.time()  # total clock
run_mode(False, "Olivia Rodrigo", 3)  # 0 train rows, reaches 2021
run_mode(True, "Olivia Rodrigo", 3)  # same with the user prediction
run_mode(False, "Taylor Swift", 4)  # hidden re-recordings reach 2021 on reveal
run_mode(True, "Taylor Swift", 3)  # with the user prediction
for who in ("Olivia Rodrigo", "Taylor Swift"):  # the two requested artists
    for on in (False, True):  # both modes
        play_to_end(who, on)  # full walk
print("\n##### WHEN and WHAT columns")  # section title
for who, yr in (("Taylor Swift", 2020), ("Taylor Swift", 2023), ("Olivia Rodrigo", 2023), ("Olivia Rodrigo", 2019), ("Lady Gaga", 2021), ("Olivia Rodrigo", 2021), ("Tyler, the Creator", 2023)):  # below the cutoff with label, flagged label, no WHAT score, at or above the cutoff with label, at or above without label, at or above and flagged
    show_year(who, yr)  # play and print
print("\n##### pool B cases")  # section title
from app_live import artist_timeline  # live data layer, to choose years that exist
from demo_game import scored_years  # playable years
for who in ("Kendrick Lamar", "Beyonce", "Peso Pluma", "boygenius"):  # pool B artists
    aid = find(who)[0]["discogs_id"]  # id
    tl = artist_timeline(aid)  # timeline
    playable = scored_years(tl, 2018)  # playable years
    what_years = [int(y) for y in tl.loc[tl["stage0_score"].notna(), "year"]]  # years with a WHAT score
    print(f"\n{who} ({aid}): playable {playable}, WHAT years {what_years}")  # data facts
    if who == "boygenius":  # no WHAT score in any year
        assert not what_years, "boygenius must have no WHAT score"  # rule of the pool
    if who == "Peso Pluma":  # WHAT only from 2023
        assert what_years and min(what_years) >= 2023, "Peso Pluma WHAT must start in 2023"  # rule of the pool
    targets = {playable[0]} | ({what_years[0]} if what_years and what_years[0] in playable else set())  # first playable year and first year with a WHAT score
    for yr in sorted(targets):  # years to show
        show_year(who, yr)  # play and print both columns
    if who == "boygenius":  # every playable year shows the missing-score text
        for yr in playable:  # all 3 playable years
            at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # fresh app
            pick(at, who)  # artist
            while at.session_state[keys["state"]][aid]["year"] != yr:  # walk to the year
                at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
                at.button(key=keys["next"].format(id=aid)).click().run()  # next year
            want = poc_tx["what_na_tip"]  # expected tooltip of the grey dash
            assert na_shown(at), "the WHAT column must show the grey dash"  # step A cell
            at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
            assert na_shown(at), "the WHAT column must show the grey dash after the reveal"  # step B cell
            print(f"   boygenius {yr}: WHAT column shows: {want}")  # printed proof
print("\n##### release titles, year-end years and the scoreboard in mode 1")  # section title
import html  # unescape of the panel text
import re  # tags of the panel HTML
import pandas as pd  # tables
route = pd.read_csv(ROOT / "results/demo/route_table_year_end.csv")  # independent reference: verdicts, realities, flags of every artist-year
route["counted_1y"] = route["reality_1y"].notna() & (route["year"] + 1 <= CONFIG["years"]["last_complete"])  # counting rule by reality year, derived here, the stored flag follows the old rule
picks = pd.read_csv(ROOT / "results/demo/release_pick.csv")  # independent reference: release titles
rt_tx = cfg["route"]["texts"]  # route texts


def at_year(name: str, year: int) -> tuple:  # app at the step A of a year
    """Fresh app, mode 1, artist picked and walked to the question year (step A); returns the app and the artist id."""
    aid = find(name)[0]["discogs_id"]  # its id
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # fresh app
    pick(at, name)  # artist
    while at.session_state[keys["state"]][aid]["year"] != year:  # walk to the year
        at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
        at.button(key=keys["next"].format(id=aid)).click().run()  # next year
    return at, aid  # app and id


def block_raw(at: AppTest, label: str) -> str:  # HTML of one question block
    """HTML of the Timing or the Style block of the mode 1 panel."""
    return next(h.value for h in at.get("html") if "poc-blk" in h.value and f"<div class='poc-sub'>{label}</div>" in h.value)  # the block element


def lines_of(raw: str) -> list:  # visible lines of a block
    """Visible text lines of a block: style blocks, tags and tooltips (attributes) removed, one line per div."""
    txt = re.sub(r"<style>.*?</style>", "", raw, flags=re.S).replace("</div>", "\n")  # one line per div
    return [x.strip() for x in html.unescape(re.sub(r"<[^>]+>", "", txt)).split("\n") if x.strip()]  # tags removed


def short(title: str) -> str:  # title cut at the route limit
    """Same cut as the page: at most title_max_chars characters, the last one an ellipsis."""
    n = cfg["route"]["title_max_chars"]  # limit
    return title if len(title) <= n else title[: n - 1].rstrip() + "\u2026"  # ellipsis counts in the limit


def release(aid: int, year: int) -> str:  # release line of a year from the pick file
    """'Title (year)' of the rule B pick of the artist-year, or the year-only text when there is none."""
    t = picks[(picks["artist_id"] == aid) & (picks["year"] == year)]  # pick of the artist-year
    return poc_tx["release_line"].format(title=short(t.iloc[0]["title"]), year=year) if len(t) else poc_tx["release_year_only"].format(year=year)  # line


import json  # style lists stored as JSON text
new_styles = pd.read_csv(ROOT / "results/demo/what_new_styles.csv")  # the new styles file
all_works = pd.concat([pd.read_csv(ROOT / "results/demo/works_demo.csv"), pd.read_csv(ROOT / cfg["pool_b"]["paths"]["works"])])  # Works of both pools


def cap3(styles: list) -> str:  # at most three styles, then +N
    """Same rule as the page."""
    return ", ".join(styles[:3]) + (" " + poc_tx["more_styles"].format(n=len(styles) - 3) if len(styles) > 3 else "")  # names and +N


def want_timing(r, reveal: bool, answer: str | None = None) -> list:  # expected lines of the Timing block
    """Expected visible lines of the Timing block from the route table and the pick file."""
    yn = {"yes": g["answer_yes"], "no": g["answer_no"]}  # words
    two = r.verdict_1y == "no" or r.verdict_2y == "no"  # cascade rule
    out = [poc_tx["label_timing"], poc_tx["q_timing"]] + ([rt_tx["year_end_tag"].format(year=r.year)] if r.is_year_end else [])  # label, question, grey tag
    out += [f"{poc_tx['your_prediction']} {yn[answer]}"] if answer else []  # user answer, this block only
    out += [f"{poc_tx['prediction']} {yn[r.verdict_1y]}"] + ([f"{poc_tx['within_two'].format(n=g['second_horizon'])} {yn[r.verdict_2y]}"] if two else [])  # predictions
    if not reveal:  # nothing but ? before the reveal
        return out + [f"{poc_tx['reality']} {poc_tx['unknown']}"] + ([f"{poc_tx['reality_two'].format(n=g['second_horizon'])} {poc_tx['unknown']}"] if two else [])  # reality lines
    for name, lab, y in [(poc_tx["reality"], r.reality_1y, r.year + 1)] + ([(poc_tx["reality_two"].format(n=g["second_horizon"]), r.reality_2y, r.year + 1 if r.reality_1y == "yes" else r.year + 2)] if two else []):  # one reality per horizon
        out.append(f"{name} {rt_tx['not_yet'] if pd.isna(lab) else yn[lab]}")  # reality word
        if lab == "yes":  # a release happened
            out.append(release(r.artist_id, y))  # title and year
    return out  # all lines


ye_what = pd.read_csv(ROOT / "results/demo/what_scores_year_end.csv")  # independent reference: year-end WHAT scores
cutoff = float(pd.read_csv(ROOT / "results/holdout_what/top10_lift.csv").set_index("model").loc["stage 0", "score_cutoff"])  # independent reference: WHAT cutoff


def ref_what(r) -> dict | None:  # reference WHAT values of a route row
    """Verdict, score, new-styles year, label and release year of a route row, from the year-end file for a year-end row and from the route table and label file otherwise; None without a shown score."""
    if r.is_year_end:  # reference-only row
        y = ye_what[(ye_what["artist_id"] == r.artist_id) & (ye_what["year"] == r.year)]  # stored row
        if y.empty:  # fewer than 3 earlier Works
            return None  # grey dash
        y = y.iloc[0]  # the row
        nol = pd.isna(y["what_label"])  # score without a resolved label
        return {"verdict": "yes" if y["stage0_score"] >= cutoff else "no", "score": y["stage0_score"], "src": int(y["last_active_year"]), "label": None if nol else y["what_label"], "ny": None if nol else int(y["next_active_year"])}  # values
    if not r.what_available:  # no WHAT score
        return None  # grey dash
    lab = labels[(labels["artist_id"] == r.artist_id) & (labels["year"] == r.year)]  # stored label of the artist-year
    empty = lab.empty or pd.isna(lab.iloc[0]["label"])  # score without a label
    return {"verdict": r.what_verdict, "score": r.what_score, "src": r.year, "label": None if empty else lab.iloc[0]["label"], "ny": None if empty else int(lab.iloc[0]["next_active_year"])}  # values


def want_style(r, reveal: bool) -> list:  # expected lines of the Style block
    """Expected visible lines of the Style block from the route table, the year-end file, the label file, the new styles file and the pick file."""
    out = [poc_tx["label_style"], poc_tx["q_style"], poc_tx["style_scope"].format(year=r.year)]  # label, question, grey line with the shown year
    w = ref_what(r)  # reference values
    if w is None:  # no WHAT score
        return out + [poc_tx["what_na"]]  # grey dash
    out.append(f"{poc_tx['prediction']} {poc_tx['likely'] if w['verdict'] == 'yes' else poc_tx['not_likely']}")  # prediction word
    if not reveal:  # nothing but ?
        return out + [f"{poc_tx['reality']} {poc_tx['unknown']}"]  # reality
    if w["label"] is None:  # score without a label
        return out + [poc_tx["what_na_tip_no_later"] if r.is_year_end else g["what_no_reality"]]  # explicit empty state
    ny = w["ny"]  # year of the release the label refers to
    ns = json.loads(new_styles[(new_styles["artist_id"] == r.artist_id) & (new_styles["year"] == w["src"])].iloc[0]["new_styles"])  # new styles
    if not r.is_year_end:  # the route table holds the WHAT reality of Work rows only
        assert r.what_reality == ("yes" if w["label"] == 1 else "no"), "route table and label file disagree"  # same label
    if w["label"] == 1:  # new style
        assert ns, "a Yes label needs new styles"  # label rule
        return out + [g["what_reality_yes"].format(styles=cap3(ns)), release(r.artist_id, ny)]  # Yes, new styles, release
    assert not ns, "a No label has no new styles"  # label rule
    pk = picks[(picks["artist_id"] == r.artist_id) & (picks["year"] == ny)]  # picked release
    rs = json.loads(all_works[all_works["work_id"] == pk.iloc[0]["work_id"]].iloc[0]["styles"]) if len(pk) else []  # its styles
    return out + [g["what_reality_no"], release(r.artist_id, ny) + (f" \u00b7 {cap3(rs)} {poc_tx['used_before']}" if rs else "")]  # No, release and its styles


from what_new_styles import stored_labels, control_file  # stored labels of both pools and the control of the file
labels = stored_labels()  # reference: stored WHAT labels of both pools and the year they refer to
res = control_file()  # control of results/demo/what_new_styles.csv
assert res["rows"] > 0 and res["missing"] == 0 and res["label_mismatch"] == 0 and res["year_mismatch"] == 0, f"new styles control failed: {res}"  # label == (new styles not empty) for every stored label
print(f"new styles control: {res['rows']} stored labels, {res['label_mismatch']} label mismatches, {res['year_mismatch']} next_active_year mismatches")  # result


def case(name: str, year: int, show: bool = False) -> None:  # one mode 1 year against the route table
    """Play one artist-year in mode 1 and compare both blocks with the route table, the label file and the pick file, before and after the reveal."""
    r = next(route[(route["artist"] == name) & (route["year"] == year)].itertuples())  # reference row
    at, aid = at_year(name, year)  # step A
    before = scoreboard(at.session_state[keys["state"]][aid]["history"])  # board before the reveal
    raw_t = block_raw(at, poc_tx["label_timing"])  # Timing block
    assert lines_of(raw_t) == want_timing(r, False), f"{name} {year}: Timing lines {lines_of(raw_t)}"  # order, words and ?
    assert lines_of(block_raw(at, poc_tx["label_style"])) == want_style(r, False), f"{name} {year}: Style lines before the reveal"  # Style block
    assert f"title='{poc_tx['score_tip'].format(score=r.score_1y, gap=gap_text(gap['next_1y']))}'" in raw_t and "%" not in re.sub(r"<style>.*?</style>", "", raw_t, flags=re.S) and "not a probability" not in raw_t, f"{name} {year}: the score sits in the tooltip with the calibration gap, never as a percentage"  # tooltip
    if ref_what(r) is not None:  # a WHAT prediction word has its own score tooltip
        assert f"title='{poc_tx['score_tip'].format(score=ref_what(r)['score'], gap=gap_text(gap['what']))}'" in block_raw(at, poc_tx["label_style"]), f"{name} {year}: WHAT score tooltip"  # tooltip
    page = " ".join([m.value for m in list(at.markdown) + list(at.caption)])  # visible markdown and captions
    assert "score" not in page.lower().replace("scoreboard", ""), f"{name} {year}: no score text on the page"  # scores only in tooltips
    at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
    after = scoreboard(at.session_state[keys["state"]][aid]["history"])  # board after the reveal
    assert after["n"] - before["n"] == int(bool(r.counted_1y)), f"{name} {year}: the scoreboard must follow counted_1y"  # counted flag only
    lt, ls = lines_of(block_raw(at, poc_tx["label_timing"])), lines_of(block_raw(at, poc_tx["label_style"]))  # blocks after the reveal
    assert lt == want_timing(r, True), f"{name} {year}: Timing after the reveal {lt}"  # reality lines
    assert ls == want_style(r, True), f"{name} {year}: Style after the reveal {ls}"  # reality lines
    assert not at.exception  # no exception
    print(f"   {name} {year}: year-end {bool(r.is_year_end)}, counted {bool(r.counted_1y)}, n {before['n']} -> {after['n']}")  # proof
    if show:  # example rows for the report
        print("      Timing:", " | ".join(lt), "\n      Style:", " | ".join(ls))  # the two blocks as text


ye = route[route["is_year_end"]]  # year-end rows of the route table
case("Taylor Swift", 2020)  # Work year, Yes with a title
case("Olivia Rodrigo", 2019)  # Work year, Yes with a title
a_no = route[~route["is_year_end"] & (route["reality_1y"] == "no") & route["counted_1y"]].iloc[0]  # a Work year whose reality is No
case(a_no["artist"], int(a_no["year"]))  # Work year, reality No
case("boygenius", 2019)  # year-end year, counted, cascade with the 2-year line
c_yes = ye[(ye["reality_1y"] == "yes") & ye["counted_1y"]].iloc[0]  # a counted year-end row whose reality is Yes
case(c_yes["artist"], int(c_yes["year"]), show=True)  # year-end year with a title (example, gap year)
c_unc = ye[~ye["counted_1y"] & ye["reality_1y"].notna()].iloc[0]  # a year-end row whose reality is in 2026 (2025)
case(c_unc["artist"], int(c_unc["year"]))  # year-end year that must not change the scoreboard
w_yes = route[route["what_available"] & (route["what_reality"] == "yes") & ~route["is_year_end"]].iloc[0]  # WHAT reality Yes
case(w_yes["artist"], int(w_yes["year"]), show=True)  # example: new styles
w_no = route[route["what_available"] & (route["what_reality"] == "no") & ~route["is_year_end"]].iloc[0]  # WHAT reality No
case(w_no["artist"], int(w_no["year"]), show=True)  # example: all used before
print("mode 1 panel lines, new styles, year-end rows and scoreboard checks passed")  # all asserts held
print("\n##### mode 1 layout: compact bar, window, cards, panel, expander, unseen warning")  # section title
from demo_game import names  # readable list from JSON text
from poc_view import window_years  # window rule under test
poc = cfg["poc"]  # poc settings


def card_pieces(at: AppTest) -> list:  # cards of the timeline as (year, visible text, whole HTML)
    """Year, visible text (tags and tooltips removed) and HTML of every card of the timeline."""
    raw = next(h.value for h in at.get("html") if "class='poc-line'" in h.value)  # timeline element
    parts = re.split(r"(?=<div class='poc-card)", raw.split("</style>", 1)[1])[1:]  # one piece per card
    return [(int(re.search(r"data-year='(\d+)'", c).group(1)), html.unescape(re.sub(r"<[^>]+>", " ", c)), c) for c in parts]  # year, text, html


def layout_checks(name: str) -> None:  # one artist, three states
    """Bar widgets, window length, card text, the two blocks, expander and scoreboard in the ask, prediction and reality states."""
    aid = find(name)[0]["discogs_id"]  # its id
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # fresh app
    pick(at, name)  # artist
    at.checkbox(key=keys["user_pred"].format(id=aid)).check().run()  # user prediction on, so the ask state exists
    for _ in range(3):  # a few years in, so the window is full
        year = at.session_state[keys["state"]][aid]["year"]  # question year
        at.radio(key=keys["answer"].format(id=aid, year=year)).set_value("yes").run()  # answer
        at.button(key=keys["lock"].format(id=aid)).click().run()  # lock
        at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
        at.button(key=keys["next"].format(id=aid)).click().run()  # next year
    year = at.session_state[keys["state"]][aid]["year"]  # question year of the checks
    works = pd.read_csv(ROOT / "results/demo/works_demo.csv")  # independent reference: Works
    for state in ("ask", "model", "reveal"):  # the three states
        if state == "model":  # after the lock
            at.radio(key=keys["answer"].format(id=aid, year=year)).set_value("no").run()  # answer
            at.button(key=keys["lock"].format(id=aid)).click().run()  # lock
        if state == "reveal":  # after the reveal
            before = scoreboard(at.session_state[keys["state"]][aid]["history"])  # board before
            at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
        assert not at.exception, [e.value for e in at.exception]  # no exception
        bar = [at.selectbox(key=keys["genre"]), at.selectbox(key=keys["artist"]), at.checkbox(key=keys["user_pred"].format(id=aid)), at.button(key=keys["restart"].format(id=aid))]  # compact bar widgets
        assert all(w is not None for w in bar) and any("poc-chips" in h.value for h in at.get("html")), f"{state}: bar widgets and chips"  # bar
        assert at.checkbox(key=keys["user_pred"].format(id=aid)).label == g["user_prediction_label"] == "My prediction", f"{state}: checkbox label"  # shortened label
        assert len([b for b in at.button if b.key in (keys["lock"].format(id=aid), keys["reveal"].format(id=aid), keys["next"].format(id=aid))]) == 1, f"{state}: exactly one action button"  # one action
        cards = card_pieces(at)  # cards of the timeline
        win = window_years(year, CONFIG_FIRST, max(int(works["year"].max()), year + 1), poc["years_before"], poc["years_after"])[0]  # expected window
        assert [c[0] for c in cards] == win and len(win) == poc["years_before"] + poc["years_after"] + 1, f"{state}: window is {poc['years_before']} before, the year, {poc['years_after']} after"  # window length
        assert ("poc-earlier" in next(h.value for h in at.get("html") if "class='poc-line'" in h.value)) == (year - poc["years_before"] > CONFIG_FIRST), f"{state}: earlier-years marker"  # marker
        for y, text, raw in cards:  # no genres or styles printed on a card
            for _, w in works[(works["artist_id"] == aid) & (works["year"] == y)].iterrows():  # Works of the card year
                for field in ("genres", "styles"):  # both lists
                    shown = names(w[field])  # readable list
                    assert not shown or not re.search(r"(?<!\w)" + re.escape(shown) + r"(?!\w)", text), f"{state}: {field} printed on the {y} card"  # only in the tooltip
            assert "overflow:hidden" not in raw and "height:" not in raw, f"{state}: no fixed height or clipping inside a card"  # nothing clipped
        left = next(h.value for h in at.get("html") if "<div class='poc-who'>" in h.value)  # left panel
        assert f"<div class='poc-who'>{year}</div>" in left, f"{state}: the year heads the left panel"  # year
        lt, ls = lines_of(block_raw(at, poc["texts"]["label_timing"])), lines_of(block_raw(at, poc["texts"]["label_style"]))  # the two blocks
        assert lt[:2] == [poc["texts"]["label_timing"], poc["texts"]["q_timing"]] and ls[:3][0] == poc["texts"]["label_style"] and ls[2] == poc["texts"]["style_scope"].format(year=year), f"{state}: block heads and the grey Style line with the shown year"  # structure
        assert any(x.startswith(poc["texts"]["prediction"]) for x in lt) and any(x.startswith(poc["texts"]["reality"]) for x in lt), f"{state}: prediction and reality lines"  # lines
        assert (poc["texts"]["prediction"] + " " + poc["texts"]["unknown"] in lt) == (state == "ask") and not any(x.startswith(poc["texts"]["your_prediction"]) for x in ls), f"{state}: prediction is ? before the lock, the user answer never in the Style block"  # state rule
        assert any(x.startswith(poc["texts"]["your_prediction"]) for x in lt) == (state != "ask"), f"{state}: the user answer shows in the Timing block after the lock"  # user line
        if state != "reveal":  # before the reveal: every reality line is only ?
            assert all(x.endswith(poc["texts"]["unknown"]) for x in lt + ls if x.startswith(poc["texts"]["reality"])) and len(lt) <= 8, f"{state}: reality lines show only ?"  # no title, no year
        assert "score" not in " ".join(lt + ls).lower(), f"{state}: no score text in the blocks"  # scores only in tooltips
        assert any(e.label == rt_tx["how_label"] for e in at.expander), f"{state}: one How to read this expander"  # expander
        body = [c.value for c in at.caption]  # captions, the expander holds the long lines
        assert rt_tx["legend_cascade"].format(n=g["second_horizon"]) in body and g["what_years_warning"].format(n=CONFIG_MIN) in body, f"{state}: legend, cascade sentence, independence sentence and warning live in the expander"  # long text
        assert g["unseen_warning"] not in [w.value for w in at.warning], f"{state}: a seen artist has no unseen warning"  # no warning
        if state == "reveal":  # reality state
            after = scoreboard(at.session_state[keys["state"]][aid]["history"])  # board after
            played = at.session_state[keys["state"]][aid]["history"][-1]  # entry of the year
            assert after["n"] - before["n"] == int(played["counted"]), "scoreboard follows counted_1y"  # unchanged rule
        print(f"   {name} {year} {state}: window {win[0]} to {win[-1]}, {len(cards)} cards, bar, blocks and expander ok")  # proof


CONFIG_FIRST = CONFIG["years"]["demo_works_first_year"]  # first year of the window data
layout_checks("A$AP Rocky")  # the artist with the most text
layout_checks("Taylor Swift")  # many Works per year
at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # fresh app
pick(at, "Olivia Rodrigo")  # an artist the model has not seen
assert g["unseen_warning"] in [w.value for w in at.warning], "the unseen-artist line must be visible"  # one visible line
assert g["unseen_warning"] not in [c.value for e in at.expander for c in e.caption], "the unseen-artist line is not inside the expander"  # outside the expander
print("unseen-artist line visible above the bar for Olivia Rodrigo")  # proof
print("mode 1 layout, panel lines and unseen warning checks passed")  # all asserts held
print(f"\ntotal AppTest time {time.time() - t_all:.2f} s")  # print the result
