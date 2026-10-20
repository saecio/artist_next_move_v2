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
from demo_multi import advance_stage, multi_rows, multi_signature, multi_summary, hidden_lines, picker_options, reliability_lines, route_ids, row_marks  # noqa: E402  mode 2 logic
from poc_view import how_lines, multi_table_html, style_table_html  # noqa: E402  mode 2 table
from demo_game import (  # noqa: E402  pure game functions
    card_status, hidden_counts, names, new_state, next_year, play, scored_years, scoreboard, session_total, verdict, year_cards, first_uncounted_year, flag, reliability, outside_text_key,  # imported names
)  # close the import

def render_multi(cfg: dict) -> None:  # tab 6 mode 2: widgets only, the rows and the HTML come from src/
    """Mode 2: prepared selection, year and artist pickers, and the read-only table."""
    rt, g, ui = cfg["route"], cfg["game"], cfg["ui"]  # route, game and ui settings
    k, tx = rt["keys"], rt["texts"]  # widget keys and texts
    st.caption(rt["selection_note"])  # one line: prepared selection, disclosure
    works = load_tables()["works"]  # Works of both pools, only for the last data year
    years = list(range(CONFIG["years"]["holdout_start"], int(works["year"].max()) + 1))  # 2018 to the last data year
    with st.expander(tx["change_label"], expanded=False):  # the pickers, collapsed by default
        c1, c2 = st.columns([1, 2])  # year and genre side by side
        year = c1.selectbox(tx["year_label"], years, index=years.index(rt["year"]), key=k["year"])  # year selector
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
    st.html(multi_table_html(rows, multi_summary(rows), g, rt, cfg["poc"], row_marks, CONFIG["what"]["min_previous_works"], year, state["stage"]))  # the table for this stage
    for line in hidden_lines(cfg, ids, year):  # the hidden-artists lines under the table
        st.caption(line)  # one caption each
    lines = how_lines(state["stage"], reliability_lines(top, load_tables()["what_below"], g), cfg["poc"]["texts"]["legend_reference"], tx["legend_cascade"])  # reliability lines, legend and cascade sentence, from Predict on
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
        genres = sorted(dict.fromkeys(a["genre"] for a in all_artists(cfg)))  # genres of both pools, sorted
        genre = st.selectbox(ui["genre_label"], [ui["all_genres"]] + genres, key=cfg["game"]["keys"]["genre"])  # genre filter
        options = artists_by_genre(cfg, None if genre == ui["all_genres"] else genre)  # filtered artists of both pools
        artist = st.selectbox(ui["artist_label"], options, format_func=lambda a: a["name"], key=cfg["game"]["keys"]["artist"])  # pick by name
        st.caption(ui["id_caption"].format(id=artist["discogs_id"]))  # keep the Discogs id visible
        g, keys = cfg["game"], cfg["game"]["keys"]  # game texts and widget keys
        aid = artist["discogs_id"]  # selected artist id
        tl = artist_timeline(aid)  # one row per year, live scores and stored labels
        years = scored_years(tl, CONFIG["years"]["holdout_start"])  # playable years
        games = st.session_state.setdefault(keys["state"], {})  # one state per artist
        state = games.setdefault(aid, new_state(years))  # this artist's state
        st.warning(g["what_years_warning"].format(n=CONFIG["what"]["min_previous_works"]))  # WHAT coverage
        if tl["artist_train_rows"].iloc[0] == 0:  # artist absent from training
            st.warning(g["unseen_warning"])  # model has not seen the artist
        pred_key = keys["user_pred"].format(id=aid)  # checkbox key of this artist
        if pred_key not in st.session_state:  # first render of this artist
            st.session_state[pred_key] = state["pred"]  # restore the kept choice
        state["pred"] = st.checkbox(g["user_prediction_label"], key=pred_key)  # optional user prediction
        if state["year"] is None:  # no playable year
            st.info(g["empty_text"])  # empty state
        else:  # game running
            year, hist = state["year"], state["history"]  # current year and history
            step = "model" if state["step"] == "ask" and not state["pred"] else state["step"]  # the question is skipped without a user prediction
            row = tl[tl["year"] == year].iloc[0]  # representative Work of the question year
            last = int(tl["year"].max())  # last year of the data
            until = min(year + 1, last) if step == "reveal" else year  # last year drawn
            works = load_tables()["works"]  # Works of the demo artists
            aw = works[works["artist_id"] == aid]  # this artist's Works
            hidden = hidden_counts(works, aid)  # hidden re-recordings per year
            for start in range(0, len(tl), 6):  # rows of six cards
                for col, (_, r) in zip(st.columns(6), tl.iloc[start:start + 6].iterrows()):  # one column per year
                    y = int(r["year"])  # card year
                    yc = year_cards(aw, y, g["hide_rerecordings"], g["max_cards_per_year"])  # Works drawn this year
                    status = card_status(yc["total"], yc["cards"], y, until)  # what to draw
                    with col.container(border=True):  # card frame
                        st.markdown(f"**{y}**")  # year
                        if status == "card":  # normal card
                            for c in yc["cards"]:  # one block per Work
                                st.write(c["title"])  # title
                                st.caption(names(c["genres"]))  # genres
                                st.caption(names(c["styles"]))  # styles
                            if yc["more"]:  # more Works than the cap
                                st.caption(g["more_line"].format(n=yc["more"]))  # "+N more" line
                        else:  # future, gap or hidden
                            st.write({"future": g["future_card"], "gap": g["gap_card"], "hidden": g["hidden_card"]}[status])  # state text
                        if g["hide_rerecordings"] and y <= until and hidden.get(y):  # hidden re-recordings in this year
                            st.caption(g["hidden_line"].format(n=hidden[y]))  # count line
            st.subheader(g["question"].format(year=year))  # question
            yn = {"yes": g["answer_yes"], "no": g["answer_no"]}  # option texts
            if step == "ask":  # user predicts first
                pick = st.radio(g["question"].format(year=year), list(yn), format_func=yn.get, index=None, key=keys["answer"].format(id=aid, year=year))  # yes / no
                if st.button(g["lock_label"], key=keys["lock"].format(id=aid), disabled=pick is None):  # lock
                    state["answer"], state["step"] = pick, "model"  # keep the answer
                    st.rerun()  # redraw with step A
            if step in ("model", "reveal"):  # step A is always visible after the ask
                wc, sc = st.columns(2)  # WHEN column and WHAT column
                top = load_tables()["what_top10"]  # WHAT cutoff and holdout figures
                rel = reliability(top)  # x and y of the reliability line
                played = hist[-1] if step == "reveal" else None  # this year's entry after the reveal
                with wc:  # WHEN column
                    st.markdown(g["when_title"])  # column title
                    st.write(g["model_score"].format(year=year, score=row["next_1y"]))  # live score
                    st.write(g["model_verdict"].format(verdict=yn[verdict(row["next_1y"], g["threshold"])]))  # model verdict
                    if state["answer"]:  # user predicted this year
                        st.write(g["user_answer"].format(answer=yn[state["answer"]]))  # user answer
                    if played:  # step B
                        if played["label"] is not None:  # stored label exists, counted or not
                            st.write(g["reality_" + played["label"]].format(year=year + 1))  # reality text of the stored label
                        else:  # last data year, no stored label
                            st.info(g["no_reality"].format(year=last))  # no reality row, no scoreboard change
                        lab2 = row["label_next_2y"]  # stored next_2y label
                        part2 = "" if pd.isna(lab2) else g["label_part"].format(label=yn["yes" if lab2 == 1 else "no"])  # label part, omitted when missing
                        st.write(g["next2_info"].format(score=row["next_2y"], label_part=part2))  # next_2y info
                with sc:  # WHAT column, never counted
                    st.markdown(g["what_title"])  # column title
                    if pd.notna(row["stage0_score"]):  # WHAT score exists for this artist-year
                        st.write(g["what_yes" if verdict(row["stage0_score"], top["score_cutoff"]) == "yes" else "what_no"])  # verdict, Yes at or above the cutoff
                        if verdict(row["stage0_score"], top["score_cutoff"]) == "yes":  # Yes verdict
                            st.caption(g["what_reliability"].format(x=rel[0], y=rel[1]))  # reliability of the top 10% group
                        else:  # No verdict
                            below = load_tables()["what_below"]  # holdout rows below the cutoff
                            st.caption(g[outside_text_key(below["z"], rel[1], g["similar_gap"])].format(z=below["z"], y=rel[1]))  # reliability outside the group
                        st.caption(g["what_score"].format(score=row["stage0_score"]))  # small score number
                        if played and pd.notna(row["what_label"]):  # step B with a stored label
                            st.write(g["what_reality_yes" if row["what_label"] == 1 else "what_reality_no"])  # stored WHAT label
                            if flag(row["what_label_after_holdout"]):  # label after the last holdout year
                                st.caption(g["what_delay_note"])  # cataloguing-delay note
                        elif played:  # step B without a stored label
                            st.write(g["what_no_reality"])  # explicit empty state
                    else:  # no WHAT score for this artist-year
                        st.write(g["what_missing"].format(n=CONFIG["what"]["min_previous_works"]))  # reason, in both steps
            if step == "model" and st.button(g["reveal_label"], key=keys["reveal"].format(id=aid)):  # reveal
                hist.append(play(row, state["answer"], g["threshold"]))  # record the year
                state["step"] = "reveal"  # step B
                st.rerun()  # redraw with step B
            if step == "reveal":  # step B
                nxt = next_year(years, year)  # next playable year
                if nxt is None:  # end of the data
                    st.info(g["last_year_text"])  # nothing to play
                if st.button(g["next_label"], key=keys["next"].format(id=aid), disabled=nxt is None):  # next year
                    state.update(year=nxt, step="ask", answer=None)  # move on, history kept
                    st.rerun()  # redraw
            if st.button(g["restart_label"], key=keys["restart"].format(id=aid)):  # restart this artist
                games[aid] = new_state(years, state["pred"])  # fresh state, same checkbox
                st.rerun()  # redraw
        st.subheader(g["scoreboard_title"])  # scoreboard
        for title, board in ((None, scoreboard(state["history"])), ("total", session_total([s["history"] for s in games.values()]))):  # artist and session
            if not board["n"]:  # nothing counted
                if title is None:  # artist board only
                    st.write(g["board_none"])  # empty state
                continue  # next board
            st.write(g["board_model" if title is None else "board_total"].format(right=board["model_right"], n=board["n"]))  # model vs reality
            if board["user_n"]:  # user played
                st.write(g["board_user" if title is None else "board_total_user"].format(right=board["user_right"], n=board["user_n"]))  # user vs reality
                if title is None:  # artist board only
                    st.write(g["board_agree"].format(right=board["agree"], n=board["user_n"]))  # user vs model
        cut = first_uncounted_year(load_tables()["when_stored"])  # first year shown but not counted, from the data
        if cut is not None:  # some year is flagged
            st.caption(g["uncounted_note"].format(year=cut))  # fixed neutral line

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
