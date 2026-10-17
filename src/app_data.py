"""Data access for the demo app: config, cached CSV loaders and row filters."""

import pandas as pd  # tables
import streamlit as st  # cache decorator
import yaml  # reads the demo config

from config import ROOT  # project root, independent of the working directory

DEMO_CONFIG_PATH = ROOT / "app" / "demo_config.yaml"  # location of the demo settings


def load_demo_config() -> dict:
    """Read app/demo_config.yaml and return it as a dict."""
    with open(DEMO_CONFIG_PATH, encoding="utf-8") as f:  # the single settings file of the app
        return yaml.safe_load(f)  # plain nested dict


@st.cache_data
def load_row_scores(path: str, columns: list) -> pd.DataFrame:
    """Read the WHAT holdout scores with only `columns` and artist_id as int."""
    df = pd.read_csv(ROOT / path, usecols=columns)  # only the columns the app needs
    df["artist_id"] = df["artist_id"].astype("int64")  # float64 in the file, ids are whole numbers
    return df  # cached by Streamlit


@st.cache_data
def load_predictions(path: str, columns: list) -> pd.DataFrame:
    """Read the global style forecast with only `columns`."""
    return pd.read_csv(ROOT / path, usecols=columns)  # cached by Streamlit


def artists_by_genre(artists: list, genre: str | None) -> list:
    """Return the demo artists of `genre`, or all of them when genre is None."""
    return [a for a in artists if genre is None or a["genre"] == genre]  # keep matching entries


def artist_year_rows(row_scores: pd.DataFrame, artist_id: int, year: int) -> pd.DataFrame:
    """Return the row_scores rows of one artist id in one year."""
    mask = (row_scores["artist_id"] == artist_id) & (row_scores["year"] == year)  # both conditions
    return row_scores[mask]  # may be empty


def add_projection_flag(predictions: pd.DataFrame, last_year: int, column: str) -> pd.DataFrame:
    """Return a copy of predictions with boolean `column`: target_year > last_year."""
    out = predictions.copy()  # do not modify the cached frame
    out[column] = out["target_year"] > last_year  # True for years without ground truth
    return out  # forecast table with the flag
