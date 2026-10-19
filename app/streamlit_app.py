"""Streamlit demo: thin page that builds the tabs from app_data and the config."""

import sys  # module search path
from pathlib import Path  # repo root

import pandas as pd  # missing-value checks
import streamlit as st  # UI

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))  # make src importable

from app_data import (  # noqa: E402  data access functions
    add_projection_flag, artists_by_genre, load_demo_config, load_predictions,  # imported names
)  # close the import
from app_live import artist_timeline, load_tables  # noqa: E402  live data layer
from config import CONFIG  # noqa: E402  project settings
from demo_game import (  # noqa: E402  pure game functions
    card_status, hidden_counts, names, new_state, next_year, play, scored_years, scoreboard, session_total, verdict, year_cards, first_uncounted_year, flag, reliability, outside_text_key,  # imported names
)  # close the import

cfg = load_demo_config()  # all texts, paths and artists
notes, ui = cfg["notices"], cfg["ui"]  # shortcuts to text blocks
first_year, last_year = cfg["validation_years"]  # selectable range
tabs = st.tabs(cfg["tabs"])  # one tab per title, in config order

for i in (0, 1, 2, 3, 4, 7):  # tabs without content yet
    with tabs[i]:  # open the tab
        st.header(cfg["tabs"][i])  # tab title
        st.info(notes["pending_notice"])  # pending text

with tabs[5]:  # Time Machine demo
    st.header(cfg["tabs"][5])  # tab title
    genres = list(dict.fromkeys(a["genre"] for a in cfg["artists"]))  # genres in config order
    genre = st.selectbox(ui["genre_label"], [ui["all_genres"]] + genres)  # genre filter
    options = artists_by_genre(cfg["artists"], None if genre == ui["all_genres"] else genre)  # filtered artists
    artist = st.selectbox(ui["artist_label"], options, format_func=lambda a: a["name"])  # pick by name
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
                        st.caption(g[outside_text_key(below["z"], rel[1])].format(z=below["z"], y=rel[1]))  # reliability outside the group
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
    st.warning(notes["style_notice"])  # forecast is global
    st.warning(notes["projection_notice"])  # years after the last validation year
    preds = load_predictions(cfg["paths"]["predictions"], cfg["predictions_columns"])  # cached forecast
    st.dataframe(add_projection_flag(preds, last_year, ui["projection_column"]), hide_index=True)  # raw table with flag
