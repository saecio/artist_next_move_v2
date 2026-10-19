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
from demo_game import scoreboard  # scoreboard of a history

cfg = load_demo_config()  # settings
g, keys = cfg["game"], cfg["game"]["keys"]  # game texts and keys
genres = list(dict.fromkeys(a["genre"] for a in cfg["artists"]))  # genre options in order
with open(ROOT / "results/holdout_what/top10_lift.csv", encoding="utf-8") as f:  # top 10% file
    top = next(r for r in csv.DictReader(f) if r["model"] == "stage 0")  # stage 0 row
with open(ROOT / "results/holdout_what/row_scores.csv", encoding="utf-8") as f:  # holdout scores
    low = [float(r["label"]) for r in csv.DictReader(f) if float(r["stage0_score"]) < float(top["score_cutoff"])]  # labels of the rows below the cutoff
EXP = {"x": round(float(top["precision_top"]) * 10), "y": round(float(top["holdout_positive_rate"]) * 10), "z": round(sum(low) / len(low) * 10)}  # independent x, y and z


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
    a = next(x for x in cfg["artists"] if x["name"] == artist_name)  # artist entry
    aid = a["discogs_id"]  # its id
    t0 = time.time()  # first run clock
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # first run
    print(f"\n===== {artist_name} ({aid}), user prediction {'ON' if user_on else 'OFF'}; first run {time.time() - t0:.2f} s")  # print the result
    at.selectbox[0].select(a["genre"]).run()  # genre filter
    names = [x["name"] for x in cfg["artists"] if x["genre"] == a["genre"]]  # filtered artists
    at.selectbox[1].select_index(names.index(artist_name)).run()  # artist
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
    a = next(x for x in cfg["artists"] if x["name"] == artist_name)  # artist entry
    aid = a["discogs_id"]  # its id
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # fresh app
    at.selectbox[0].select(a["genre"]).run()  # genre filter
    at.selectbox[1].select_index([x["name"] for x in cfg["artists"] if x["genre"] == a["genre"]].index(artist_name)).run()  # artist
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
    a = next(x for x in cfg["artists"] if x["name"] == artist_name)  # artist entry
    aid = a["discogs_id"]  # its id
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # fresh app
    at.selectbox[0].select(a["genre"]).run()  # genre filter
    at.selectbox[1].select_index([x["name"] for x in cfg["artists"] if x["genre"] == a["genre"]].index(artist_name)).run()  # artist
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
print(f"\ntotal AppTest time {time.time() - t_all:.2f} s")  # print the result
