"""Standalone proof of concept: tab 6 on one screen, with the prediction badges on the cards of the timeline."""

import sys  # module search path
from pathlib import Path  # repo root

import streamlit as st  # UI

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))  # make src importable

from app_data import all_artists, artists_by_genre, load_demo_config  # noqa: E402  demo settings and artist filter
from app_live import artist_timeline, load_tables  # noqa: E402  live data layer
from config import CONFIG  # noqa: E402  project settings
from demo_game import (  # noqa: E402  game logic reused as it is
    first_uncounted_year, hidden_counts, new_state, next_year, play, scored_years, scoreboard, session_total,  # imported game functions
)  # close the import
from poc_view import chips_html, css, marks, notes, panel_html, table_rows, timeline_html, window_years  # noqa: E402  HTML builders

cfg = load_demo_config()  # all texts, paths and artists
g, poc = cfg["game"], cfg["poc"]  # game texts and poc section
keys, t = poc["keys"], poc["texts"]  # widget keys and poc texts
yn = {"yes": g["answer_yes"], "no": g["answer_no"]}  # Yes / No words
st.set_page_config(page_title=poc["page_title"], layout="wide")  # wide layout, set before any other call
st.html(css(poc))  # page-level style, once per run
games = st.session_state.setdefault(keys["state"], {})  # one state per artist, poc_ keys only
bar = st.columns([1.1, 1.3, 1.2, 2.0, 0.9, 2.6], vertical_alignment="bottom")  # compact bar: selectors, checkbox, action, restart, chips
genres = sorted(dict.fromkeys(a["genre"] for a in all_artists(cfg)))  # genres of both pools, sorted
genre = bar[0].selectbox(t["genre_label"], [cfg["ui"]["all_genres"]] + genres, key=keys["genre"])  # genre filter
options = artists_by_genre(cfg, None if genre == cfg["ui"]["all_genres"] else genre)  # filtered artists of both pools
artist = bar[1].selectbox(t["artist_label"], options, format_func=lambda a: a["name"], key=keys["artist"])  # artist
aid = artist["discogs_id"]  # selected artist id
tl = artist_timeline(aid)  # one row per year, live scores and stored labels
years = scored_years(tl, CONFIG["years"]["holdout_start"])  # playable years
state = games.setdefault(aid, new_state(years))  # this artist's state
pred_key = keys["user_pred"].format(id=aid)  # checkbox key of this artist
if pred_key not in st.session_state:  # first render of this artist
    st.session_state[pred_key] = state["pred"]  # restore the kept choice
state["pred"] = bar[2].checkbox(g["user_prediction_label"], key=pred_key)  # optional user prediction
if state["year"] is None:  # no playable year
    st.info(g["empty_text"])  # empty state
    st.stop()  # nothing else to draw
year, hist = state["year"], state["history"]  # current year and history
step = "model" if state["step"] == "ask" and not state["pred"] else state["step"]  # the question is skipped without a user prediction
row = tl[tl["year"] == year].iloc[0]  # representative Work of the question year
last = int(tl["year"].max())  # last year of the data
nxt = next_year(years, year)  # next playable year
with bar[3]:  # action button of the current step
    if step == "ask":  # question and lock
        pick = st.radio(g["question"].format(year=year), list(yn), format_func=yn.get, index=None, horizontal=True, label_visibility="collapsed", key=keys["answer"].format(id=aid, year=year))  # yes / no
        if st.button(t["action_lock"], key=keys["action"].format(id=aid), disabled=pick is None):  # lock
            state["answer"], state["step"] = pick, "model"  # keep the answer
            st.rerun()  # redraw with the prediction
    elif step == "model":  # prediction shown, reveal next
        if st.button(t["action_reveal"], key=keys["action"].format(id=aid)):  # reveal
            hist.append(play(row, state["answer"], g["threshold"]))  # record the year
            state["step"] = "reveal"  # reality step
            st.rerun()  # redraw with the reality
    else:  # reality shown, next year
        if st.button(t["action_next"], key=keys["action"].format(id=aid), disabled=nxt is None):  # next year
            state.update(year=nxt, step="ask", answer=None)  # move on, history kept
            st.rerun()  # redraw
        if nxt is None:  # end of the data
            st.caption(g["last_year_text"])  # nothing to play
top = load_tables()["what_top10"]  # WHAT cutoff and holdout figures
board, total = scoreboard(hist), session_total([s["history"] for s in games.values()])  # artist and session boards
lines = [g["board_model"].format(right=board["model_right"], n=board["n"]) if board["n"] else g["board_none"]]  # model chip
if board["user_n"]:  # the user played counted years
    lines.append(g["board_user"].format(right=board["user_right"], n=board["user_n"]))  # user chip
if total["n"]:  # some year counted in the session
    lines.append(g["board_total"].format(right=total["model_right"], n=total["n"]))  # session chip
bar[5].html(chips_html(lines, poc))  # scoreboard chips
if bar[4].button(t["restart"], key=keys["restart"].format(id=aid)):  # restart this artist
    games[aid] = new_state(years, state["pred"])  # fresh state, same checkbox
    st.rerun()  # redraw
st.markdown(f"**{g['question'].format(year=year)}**")  # question on one line
played = hist[-1] if step == "reveal" else None  # this year's entry after the reveal
until = min(year + 1, last) if step == "reveal" else year  # last year with visible Works
works = load_tables()["works"]  # Works of the demo artists
aw = works[works["artist_id"] == aid]  # this artist's Works
window = window_years(year, CONFIG["years"]["demo_works_first_year"], max(last, year + 1), poc["years_before"], poc["years_after"])  # years of the window
st.html(timeline_html(aw, hidden_counts(works, aid), window, until, year + 1, marks(row, played, top, g["threshold"], year >= last), g, poc))  # timeline with the two marks on the predicted year
st.html(panel_html(table_rows(row, played, top, g["threshold"], state["answer"] if state["pred"] else None, g, poc), notes(row, played, top, load_tables()["what_below"], first_uncounted_year(load_tables()["when_stored"]), yn, g, poc, CONFIG["what"]["min_previous_works"]), poc))  # always visible prediction panel
