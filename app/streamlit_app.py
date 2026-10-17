"""Streamlit demo: thin page that builds the tabs from app_data and the config."""

import sys  # module search path
from pathlib import Path  # repo root

import streamlit as st  # UI

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))  # make src importable

from app_data import (  # noqa: E402  data access functions
    add_projection_flag, artist_year_rows, artists_by_genre,
    load_demo_config, load_predictions, load_row_scores,
)

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
    year = st.selectbox(ui["year_label"], list(range(first_year, last_year + 1)))  # validation years only
    st.warning(notes["what_notice"])  # scores are for ranking only
    st.warning(notes["projection_notice"])  # years after the last validation year
    scores = load_row_scores(cfg["paths"]["row_scores"], cfg["row_scores_columns"])  # cached scores
    rows = artist_year_rows(scores, artist["discogs_id"], year)  # rows of this artist and year
    if rows.empty:  # no data for this selection
        st.info(ui["empty_rows"])  # empty state from the config
    else:  # rows found
        st.dataframe(rows, hide_index=True)  # raw table

with tabs[6]:  # Global style forecast
    st.header(cfg["tabs"][6])  # tab title
    st.warning(notes["style_notice"])  # forecast is global
    st.warning(notes["projection_notice"])  # years after the last validation year
    preds = load_predictions(cfg["paths"]["predictions"], cfg["predictions_columns"])  # cached forecast
    st.dataframe(add_projection_flag(preds, last_year, ui["projection_column"]), hide_index=True)  # raw table with flag
