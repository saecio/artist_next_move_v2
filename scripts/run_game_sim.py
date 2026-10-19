"""Walk-through simulation of the tab 6 game for three artists in both modes. Prints only, writes no files.

Usage:
    venv/bin/python3 scripts/run_game_sim.py
"""

import sys  # module path
import time  # timings
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

import pandas as pd  # tables

from app_data import load_demo_config  # demo settings
from app_live import artist_timeline, load_tables  # live data layer
from config import CONFIG  # project settings
from demo_game import hidden_counts, new_state, next_year, play, scored_years, scoreboard, session_total, year_cards, first_uncounted_year  # game functions

cfg = load_demo_config()["game"]  # game settings
THRESHOLD = cfg["threshold"]  # model rule
ARTISTS = [7592689, 1124645, 4778]  # 0 train rows, hidden re-recordings, a regular artist
stored = load_tables()["when_stored"]  # stored scores and labels, independent source for the check
stored = stored[stored["target"] == "next_1y"]  # next_1y only


def walk(aid: int, user_on: bool) -> list:  # play every year of one artist
    """Play every year of one artist with a user who always answers yes."""
    tl = artist_timeline(aid)  # timeline
    years = scored_years(tl, CONFIG["years"]["holdout_start"])  # playable years
    state, year = new_state(years, user_on), years[0]  # state and first year
    while year is not None:  # until the last playable year
        row = tl[tl["year"] == year].iloc[0]  # representative Work
        state["history"].append(play(row, "yes" if user_on else None, THRESHOLD))  # question, step A and step B in one call
        year = next_year(years, year)  # next playable year
    return state["history"]  # played years


def expected(aid: int, history: list) -> pd.DataFrame:  # counted rows from the stored scores
    """Stored rows of the representative Works of the played years that must be counted."""
    tl = artist_timeline(aid)  # timeline, representative Work per year
    reps = tl[tl["year"].isin([h["year"] for h in history])][["year", "work_id"]].rename(columns={"year": "snapshot_year"})  # played representative Works
    s = stored[stored["artist_id"] == aid].merge(reps, on=["snapshot_year", "work_id"])  # stored rows of those Works
    return s[s["label"].notna() & ~s["is_projection"] & ~s["label_after_holdout"]]  # counted rows


t0 = time.time()  # clock
for user_on in (False, True):  # both modes
    print(f"\n===== mode: user prediction {'ON (always yes)' if user_on else 'OFF'} =====")  # print the result
    histories = []  # for the session total
    for aid in ARTISTS:  # three artists
        h = walk(aid, user_on)  # play the artist
        histories.append(h)  # keep for the total
        board = scoreboard(h)  # scoreboard of the game logic
        tl = artist_timeline(aid)  # timeline for the check
        rep = tl.set_index("year")  # by year
        ref = expected(aid, h)  # stored counted rows
        ref_rep = ref.set_index("snapshot_year")  # by year
        model_ok = int(sum(((rep.loc[y, "next_1y"] >= THRESHOLD) == (r["label"] == 1)) for y, r in ref_rep.iterrows()))  # model right, from live score
        stored_ok = int(sum(((r["score"] >= THRESHOLD) == (r["label"] == 1)) for _, r in ref.iterrows()))  # model right, from stored score
        user_ok = int((ref["label"] == 1).sum())  # always yes is right when the label is 1
        print(f"\nartist {aid}: train rows {tl['artist_train_rows'].iloc[0]}, played years {[x['year'] for x in h]}, counted {board['n']}")  # print the result
        print("scoreboard:", board)  # print the result
        print(f"check: counted {board['n']} vs stored {len(ref)}: {board['n'] == len(ref)}; model right {board['model_right']} vs live {model_ok} vs stored-score {stored_ok}: {board['model_right'] == model_ok == stored_ok}", end="")  # print the result
        print(f"; user right {board['user_right']} vs stored {user_ok}: {board['user_right'] == user_ok}" if user_on else "; user fields None: " + str(board["user_n"] is None and board["user_right"] is None and board["agree"] is None))  # print the result
        print("not counted:", [(x["year"], x["label"]) for x in h if not x["counted"]])  # print the result
    print("\nsession total:", session_total(histories))  # print the result

print("\n== 7592689 in 2021 (0 train rows)")  # print the result
row = artist_timeline(7592689).query("year == 2021").iloc[0]  # the requested year
print(play(row, "yes", THRESHOLD), "| title:", row["title"], "| next_1y", round(row["next_1y"], 4), "| WHAT:", row["stage0_score"], row["what_label"])  # print the result
print("\n== 1124645 hidden re-recordings per year:", hidden_counts(load_tables()["works"], 1124645))  # print the result
print("\n== card cap, 7592689 in 2021 and 1124645 in 2021")  # print the result
works = load_tables()["works"]  # Works of the demo artists
for aid in (7592689, 1124645):  # the requested case and a case with hidden re-recordings
    aw = works[works["artist_id"] == aid]  # this artist's Works
    in_year = aw[aw["year"] == 2021]  # file rows of the year
    for hide in (True, False):  # key on and off
        yc = year_cards(aw, 2021, hide, cfg["max_cards_per_year"])  # cards of the year
        visible = int((~in_year["is_rerecording"]).sum()) if hide else len(in_year)  # visible Works by direct count
        print(f"{aid} hide={hide}: Works in file {yc['total']} (direct {len(in_year)}), visible {visible}, cards drawn {len(yc['cards'])}, +{yc['more']} more, cap {cfg['max_cards_per_year']}")  # print the result
        assert yc["total"] == len(in_year) and len(yc["cards"]) == min(visible, cfg["max_cards_per_year"]) and yc["more"] == max(visible - cfg["max_cards_per_year"], 0)  # cap and more count
print("cap checks passed")  # print the result
print("\n== uncounted years, 7592689 and 1124645 (label_after_holdout and last data year)")  # section title
cut = first_uncounted_year(stored.assign(target="next_1y"))  # first uncounted year from the stored scores
print("first uncounted year from the data:", cut)  # derived, not hard-coded
for aid in (7592689, 1124645):  # the two requested artists
    for user_on in (False, True):  # both modes
        h = walk(aid, user_on)  # full walk
        for i, x in enumerate(h):  # every played year
            if x["year"] >= cut:  # years shown but not counted
                before = scoreboard(h[:i])  # board before the year
                after = scoreboard(h[:i + 1])  # board after the year
                print(f"{aid} user={user_on} {x['year']}: model {x['model']}, user {x['user']}, label {x['label']}, counted {x['counted']}, n {before['n']} -> {after['n']}, user_n {before['user_n']} -> {after['user_n']}")  # one line per year
                assert not x["counted"] and before == after, "an uncounted year changed the scoreboard"  # board unchanged
print("uncounted-year checks passed")  # all asserts held
print(f"\nsimulation time {time.time() - t0:.2f} s")  # print the result
