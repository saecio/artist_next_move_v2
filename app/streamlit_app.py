"""Streamlit demo: thin page that builds the tabs from app_data and the config."""

import sys  # module search path
from pathlib import Path  # repo root

import pandas as pd  # missing-value checks
import streamlit as st  # UI

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))  # make src importable

from app_data import (  # noqa: E402  data access functions
    all_artists, artists_by_genre, load_demo_config,  # imported names
)  # close the import
from app_live import artist_timeline, load_tables  # noqa: E402  live data layer
from config import CONFIG  # noqa: E402  project settings
from style_tab import forecast_table, mark_for, style_summary, style_years, with_reality, year_status  # noqa: E402  tab 7 logic
from demo_multi import advance_stage, multi_rows, multi_signature, multi_summary, hidden_lines, picker_options, play_year, playable_years, reliability_lines, route_ids, row_marks  # noqa: E402  mode 2 logic, reused by mode 1
from poc_view import card_marks, chips_html, how_lines, multi_table_html, style_html, style_table_html, timeline_html, timing_html, window_years, work_panel_html, year_end_tag  # noqa: E402  mode 2 table, mode 1 timeline, panels and tag
from demo_game import (  # noqa: E402  pure game functions
    hidden_counts, new_state, next_year, scored_years, scoreboard, session_total, first_uncounted_year,  # imported names
)  # close the import

st.set_page_config(layout="wide")  # wide page, so the compact bar and the seven-card timeline fit on one line

def render_multi(cfg: dict) -> None:  # tab 6 mode 2: widgets only, the rows and the HTML come from src/
    """Mode 2: prepared selection, year and artist pickers, and the read-only table."""
    rt, g, ui = cfg["route"], cfg["game"], cfg["ui"]  # route, game and ui settings
    k, tx = rt["keys"], rt["texts"]  # widget keys and texts
    st.caption(rt["selection_note"])  # one line: prepared selection, disclosure
    works = load_tables()["works"]  # Works of both pools, only for the last data year
    years = list(range(CONFIG["years"]["holdout_start"], int(works["year"].max()) + 1))  # 2018 to the last data year
    with st.expander(tx["change_label"], expanded=False):  # the pickers, collapsed by default
        c1, c2 = st.columns([1, 2])  # year and genre side by side
        first = {} if k["year"] in st.session_state else {"index": years.index(rt["year"])}  # the prepared year only at the first draw, a widget with a session value must not get index
        year = c1.selectbox(tx["year_label"], years, key=k["year"], **first)  # year selector
        genres = sorted(dict.fromkeys(a["genre"] for a in all_artists(cfg)))  # genres of both pools
        genre = c2.selectbox(tx["genre_label"], [ui["all_genres"]] + genres, key=k["genre"])  # genre filter
        names = {int(a["discogs_id"]): a["name"] for a in all_artists(cfg)}  # id -> name
        chosen = st.session_state.get(k["artists"], route_ids(cfg))  # current selection, the prepared one at first
        ids = st.multiselect(tx["pick_label"], picker_options(cfg, None if genre == ui["all_genres"] else genre, chosen), default=route_ids(cfg) if k["artists"] not in st.session_state else None, format_func=names.get, max_selections=rt["max_artists"], key=k["artists"])  # artist picker, capped
        st.caption(tx["cap_note"].format(n=rt["max_artists"]))  # the cap
    top = load_tables()["what_top10"]  # WHAT cutoff and holdout figures
    state = advance_stage(st.session_state.get(k["stage"]), multi_signature(year, ids), None)  # reset when the year or the artists changed
    st.session_state[k["stage"]] = state  # keep the state, mode 1 keys are not touched
    if state["stage"] == "placeholder" and st.button(tx["predict_button"], key=k["predict"], type="primary", width="stretch"):  # step 1
        st.session_state[k["stage"]] = advance_stage(state, multi_signature(year, ids), "predict")  # show the model verdicts
        st.rerun()  # redraw
    if state["stage"] == "predict" and st.button(tx["reveal_button"], key=k["reveal"], type="primary", width="stretch"):  # step 2, only after predict
        st.session_state[k["stage"]] = advance_stage(state, multi_signature(year, ids), "reveal")  # show the realities
        st.rerun()  # redraw
    rows = multi_rows(cfg, ids, year, g["threshold"], top["score_cutoff"], rt["max_artists"], state["stage"])  # table rows, selection order until predict
    if rows.empty:  # nothing selected, or no selected artist has a row
        for line in hidden_lines(cfg, ids, year):  # the hidden-artists lines
            st.caption(line)  # one caption each
        return  # nothing to draw
    st.html(multi_table_html(rows, multi_summary(rows), g, rt, cfg["poc"], row_marks, year, state["stage"]))  # the table for this stage
    for line in hidden_lines(cfg, ids, year):  # the hidden-artists lines under the table
        st.caption(line)  # one caption each
    lines = how_lines(state["stage"], reliability_lines(top, load_tables()["what_below"], g), cfg["poc"]["texts"]["legend_reference"], tx["legend_cascade"], tx["independent_note"])  # reliability lines, legend and cascade sentence, from Predict on
    if lines:  # not in the placeholder stage
        with st.expander(tx["how_label"], expanded=False):  # collapsed by default
            for line in lines:  # one line each
                st.caption(line)  # small text


cfg = load_demo_config()  # all texts, paths and artists
notes, ui = cfg["notices"], cfg["ui"]  # shortcuts to text blocks
tabs = st.tabs(cfg["tabs"])  # one tab per title, in config order

for i in (0, 1, 2, 3, 4, 7):  # tabs without content yet
    with tabs[i]:  # open the tab
        st.header(cfg["tabs"][i])  # tab title
        st.info(notes["pending_notice"])  # pending text

with tabs[5]:  # Time Machine demo
    st.header(cfg["tabs"][5])  # tab title
    g, rt = cfg["game"], cfg["route"]  # game and route settings
    modes = list(g["mode_options"])  # mode keys
    for key in (rt["keys"]["year"], rt["keys"]["genre"], rt["keys"]["artists"], g["keys"]["genre"], g["keys"]["artist"]):  # widgets of the mode that is not shown
        if key in st.session_state:  # Streamlit drops the value of a widget that is not drawn
            st.session_state[key] = st.session_state[key]  # keep it, so a switch of mode loses nothing
    mode = st.radio(g["mode_label"], modes, format_func=g["mode_options"].get, index=modes.index(g["default_mode"]), horizontal=True, key=g["keys"]["mode"])  # mode switch, mode 2 first
    if mode == "many":  # mode 2: several artists, read-only table
        render_multi(cfg)  # the page only calls the renderer
    else:  # mode 1: one artist
        g, keys, poc = cfg["game"], cfg["game"]["keys"], cfg["poc"]  # game texts, widget keys and poc settings
        rt = cfg["route"]["texts"]  # route texts: year-end tag, WHAT texts, expander label
        yn = {"yes": g["answer_yes"], "no": g["answer_no"]}  # Yes / No words
        games = st.session_state.setdefault(keys["state"], {})  # one state per artist
        warn = st.empty()  # slot above the bar for the unseen-artist line, filled once the artist is known
        bar = st.columns([1.1, 1.3, 1.6, 2.8, 0.9], vertical_alignment="bottom")  # compact bar: selectors, checkbox, action, restart
        chips_slot = st.empty()  # slot for the scoreboard chips under the bar, filled at the end of the run
        genres = sorted(dict.fromkeys(a["genre"] for a in all_artists(cfg)))  # genres of both pools, sorted
        genre = bar[0].selectbox(ui["genre_label"], [ui["all_genres"]] + genres, key=keys["genre"])  # genre filter
        options = artists_by_genre(cfg, None if genre == ui["all_genres"] else genre)  # filtered artists of both pools
        artist = bar[1].selectbox(ui["artist_label"], options, format_func=lambda a: a["name"], key=keys["artist"])  # pick by name
        aid = artist["discogs_id"]  # selected artist id
        tl = artist_timeline(aid)  # one row per year, live scores and stored labels
        years = playable_years(aid, tl, CONFIG["years"]["holdout_start"])  # playable years: Work years and year-end years
        if tl["artist_train_rows"].iloc[0] == 0:  # artist absent from training
            warn.warning(g["unseen_warning"])  # one visible line above the bar
        ye_years = set(years) - set(scored_years(tl, CONFIG["years"]["holdout_start"]))  # years without a dated Work
        state = games.setdefault(aid, new_state(years))  # this artist's state
        pred_key = keys["user_pred"].format(id=aid)  # checkbox key of this artist
        if pred_key not in st.session_state:  # first render of this artist
            st.session_state[pred_key] = state["pred"]  # restore the kept choice
        state["pred"] = bar[2].checkbox(g["user_prediction_label"], key=pred_key)  # optional user prediction
        if state["year"] is None:  # no playable year
            st.info(g["empty_text"])  # empty state
        else:  # game running
            year, hist = state["year"], state["history"]  # current year and history
            step = "model" if state["step"] == "ask" and not state["pred"] else state["step"]  # the question is skipped without a user prediction
            row = tl[tl["year"] == year].iloc[0]  # representative Work of the question year, empty values in a year-end year
            top = load_tables()["what_top10"]  # WHAT cutoff and holdout figures
            mr = multi_rows(cfg, [aid], year, g["threshold"], top["score_cutoff"], 1).iloc[0]  # the mode 2 row of the question year: both horizons, realities and release titles
            last = int(tl["year"].max())  # last year of the data
            nxt = next_year(years, year)  # next playable year
            played = hist[-1] if step == "reveal" else None  # this year's entry after the reveal
            with bar[3]:  # action button of the current step
                if step == "ask":  # the user predicts first
                    rc, bc = st.columns([1, 1.4], vertical_alignment="center")  # answer and lock side by side
                    pick = rc.radio(g["question"].format(year=year), list(yn), format_func=yn.get, index=None, horizontal=True, label_visibility="collapsed", key=keys["answer"].format(id=aid, year=year))  # yes / no
                    if bc.button(g["lock_label"], key=keys["lock"].format(id=aid), disabled=pick is None):  # lock
                        state["answer"], state["step"] = pick, "model"  # keep the answer
                        st.rerun()  # redraw with the prediction
                elif step == "model":  # prediction shown, reveal next
                    if st.button(g["reveal_label"], key=keys["reveal"].format(id=aid)):  # reveal
                        hist.append(play_year(mr, state["answer"]))  # record the year, counted by the counted_1y flag
                        state["step"] = "reveal"  # reality step
                        st.rerun()  # redraw with the reality
                else:  # reality shown, next year
                    if st.button(g["next_label"], key=keys["next"].format(id=aid), disabled=nxt is None):  # next year
                        state.update(year=nxt, step="ask", answer=None)  # move on, history kept
                        st.rerun()  # redraw
                    if nxt is None:  # end of the data
                        st.caption(g["last_year_text"])  # nothing to play
            if bar[4].button(g["restart_label"], key=keys["restart"].format(id=aid)):  # restart this artist
                games[aid] = new_state(years, state["pred"])  # fresh state, same checkbox
                st.rerun()  # redraw
            st.markdown(f"**{g['question'].format(year=year)}**")  # question of the current year
            works = load_tables()["works"]  # Works of the demo artists
            ye = {y: year_end_tag(rt, y) for y in ye_years}  # year-end tag of each year without a Work
            window = window_years(year, CONFIG["years"]["demo_works_first_year"], max(last, year + 1), poc["years_before"], poc["years_after"])  # years of the window
            until = min(year + 1, last) if step == "reveal" else year  # last year drawn
            st.html(timeline_html(works[works["artist_id"] == aid], hidden_counts(works, aid), window, until, year + 1, card_marks(mr, played is not None, row_marks), g, poc, ye))  # one-row timeline, marks on the predicted year
            lc, tc, sc = st.columns([1, 1.2, 1.2])  # left: the year and the Work, then the Timing and the Style blocks
            with lc:  # the year and its representative Work
                st.html(work_panel_html(row, year, poc["texts"]["gap_card"] if mr["is_year_end"] else None, poc))  # year heading and Work title
            tc.html(timing_html(mr, step, state["answer"] if state["pred"] else None, g, cfg["route"], poc))  # Timing block: the user answer only here
            sc.html(style_html(mr, step, g, cfg["route"], poc))  # Style block, independent of the Timing block
            cut = first_uncounted_year(load_tables()["when_stored"])  # first year shown but not counted, from the data
            extra = [g["what_years_warning"].format(n=CONFIG["what"]["min_previous_works"])]  # WHAT coverage
            if cut is not None:  # some year is flagged
                extra.append(g["uncounted_note"].format(year=cut))  # fixed neutral line
            if played and mr["what_available"] and pd.notna(mr["what_label"]) and not mr["what_safe"]:  # WHAT label after the last holdout year
                extra.append(g["what_delay_note"])  # cataloguing-delay note
            with st.expander(rt["how_label"], expanded=False):  # all long explanations, closed by default
                for line in how_lines("predict", reliability_lines(top, load_tables()["what_below"], g), poc["texts"]["legend_reference"], rt["legend_cascade"], rt["independent_note"]) + extra:  # reliability, legend, cascade sentence and notes
                    st.caption(line)  # small text
        board, total = scoreboard(state["history"]), session_total([s["history"] for s in games.values()])  # artist and session boards
        chips = [g["board_model"].format(right=board["model_right"], n=board["n"]) if board["n"] else g["board_none"]]  # model chip
        if board["user_n"]:  # user played counted years
            chips += [g["board_user"].format(right=board["user_right"], n=board["user_n"]), g["board_agree"].format(right=board["agree"], n=board["user_n"])]  # user chips
        if total["n"]:  # some year counted in the session
            chips.append(g["board_total"].format(right=total["model_right"], n=total["n"]))  # session chip
            if total["user_n"]:  # user played in the session
                chips.append(g["board_total_user"].format(right=total["user_right"], n=total["user_n"]))  # session user chip
        chips_slot.html(chips_html(chips, poc))  # scoreboard chips, one wrapping line under the bar

with tabs[6]:  # Global style forecast
    st.header(cfg["tabs"][6])  # tab title
    stt = cfg["style_tab"]  # tab 7 settings
    sk, sx = stt["keys"], stt["texts"]  # widget keys and texts
    st.warning(notes["style_notice"])  # forecast is global
    st.caption(stt["headline"])  # one neutral visible line
    with st.expander(sx["validation_label"], expanded=False):  # closed, on the page from the start
        st.caption(stt["validation_text"])  # one text: tested years, hits, repeat_last, ranges, few years
    year7 = st.selectbox(sx["year_label"], style_years(cfg), key=sk["year"])  # every target year, none hidden
    status = year_status(year7, cfg)  # kind and status line
    (st.info if status["kind"] != "none" else st.warning)(status["text"])  # reason of a flagged year or the no-forecast text
    if status["kind"] != "none":  # a forecast exists for this year
        state7 = advance_stage(st.session_state.get(sk["stage"]), multi_signature(year7, []), None)  # reset when the year changed
        st.session_state[sk["stage"]] = state7  # keep the state, tab 6 keys are not touched
        if state7["stage"] == "placeholder" and st.button(sx["predict_button"], key=sk["predict"], type="primary", width="stretch"):  # step 1
            st.session_state[sk["stage"]] = advance_stage(state7, multi_signature(year7, []), "predict")  # show the forecast
            st.rerun()  # redraw
        if state7["stage"] == "predict" and st.button(sx["reveal_button"], key=sk["reveal"], type="primary", width="stretch"):  # step 2
            st.session_state[sk["stage"]] = advance_stage(state7, multi_signature(year7, []), "reveal")  # show the reality
            st.rerun()  # redraw
        table7 = with_reality(forecast_table(year7, stt["top_n"]), year7)  # forecast top N with reality, shown by stage
        st.html(style_table_html(table7, style_summary(year7, stt["top_n"], cfg), status["kind"] == "flagged", stt, cfg["poc"], mark_for, state7["stage"], stt["top_n"]))  # the table for this stage
        if state7["stage"] != "placeholder":  # reading notes from Predict on
            with st.expander(sx["how_label"], expanded=False):  # collapsed by default
                for line in (cfg["poc"]["texts"]["legend_reference"], stt["growth_note"], stt["range_note"]):  # legend, range and noise notes
                    st.caption(line)  # small text
