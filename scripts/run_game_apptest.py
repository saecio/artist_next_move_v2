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
g, keys = cfg["game"], cfg["game"]["keys"]  # game texts and keys
CONFIG_MIN = CONFIG["what"]["min_previous_works"]  # earlier Works required for a WHAT score
with open(ROOT / "results/holdout_what/top10_lift.csv", encoding="utf-8") as f:  # top 10% file
    top = next(r for r in csv.DictReader(f) if r["model"] == "stage 0")  # stage 0 row
with open(ROOT / "results/holdout_what/row_scores.csv", encoding="utf-8") as f:  # holdout scores
    low = [float(r["label"]) for r in csv.DictReader(f) if float(r["stage0_score"]) < float(top["score_cutoff"])]  # labels of the rows below the cutoff
EXP = {"x": round(float(top["precision_top"]) * 10), "y": round(float(top["holdout_positive_rate"]) * 10), "z": round(sum(low) / len(low) * 10)}  # independent x, y and z


def find(name: str) -> tuple:  # artist entry and pool key
    """Artist entry and pool key of a display name, searching both pools (the pool is not on screen)."""
    for pool, lst in cfg["pool_b"]["pool_lists"].items():  # each pool and its config list
        for x in cfg[lst]:  # artists of the pool
            if x["name"] == name:  # found
                return x, pool  # entry and pool


def pick(at: AppTest, name: str) -> int:  # select genre and artist from the mixed list
    """Select the genre and the artist from the mixed, name-sorted list; no pool widget exists."""
    assert len(at.radio) == 0 or all(r.label != "Artists" for r in at.radio)  # no pool selector on the page
    at.radio(key=keys["mode"]).set_value("one").run()  # mode 1, mode 2 opens first
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
    keep = ("Model", "My answer", "Reality", "Information", "**WHEN", "**WHAT", "WHAT", "Yes: this", "No: this", "In past tests", "This year may", "No WHAT", "No reality", "Years from", "Last year", "User", "Session")  # line prefixes of the game texts
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
        at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
        n_after = scoreboard(at.session_state[keys["state"]][aid]["history"])  # board after the reveal
        if year in (2024, 2026):  # requested years
            assert n_before["n"] == n_after["n"] and n_before["user_n"] == n_after["user_n"], f"board changed in {year}"  # no change
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
    if any(x.startswith(("Yes: this", "No: this")) for x in texts):  # the WHAT column has a verdict
        yes = any(x.startswith("Yes: this") for x in texts)  # verdict of this year
        want = g["what_reliability"].format(x=EXP["x"], y=EXP["y"]) if yes else g["what_reliability_no_similar" if abs(EXP["z"] - EXP["y"]) <= 1 else "what_reliability_no_other"].format(z=EXP["z"], y=EXP["y"])  # expected line from the independent numbers
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
            want = g["what_missing"].format(n=CONFIG_MIN)  # expected text
            assert want in [m.value for m in at.markdown], "the WHAT column must show what_missing"  # step A text
            at.button(key=keys["reveal"].format(id=aid)).click().run()  # reveal
            assert want in [m.value for m in at.markdown], "the WHAT column must show what_missing after the reveal"  # step B text
            print(f"   boygenius {yr}: WHAT column shows: {want}")  # printed proof
print(f"\ntotal AppTest time {time.time() - t_all:.2f} s")  # print the result
