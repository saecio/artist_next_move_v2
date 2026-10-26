"""Data access for the demo app: config, cached CSV loaders and row filters."""

import pandas as pd  # tables
import streamlit as st  # cache decorator
import yaml  # reads the demo config

from config import ROOT  # project root, independent of the working directory

DEMO_CONFIG_PATH = ROOT / "app" / "demo_config.yaml"  # location of the demo settings


@st.cache_data(show_spinner=False)  # cache the parsed settings, callers get their own copy
def load_demo_config() -> dict:  # read the demo settings
    """Read app/demo_config.yaml and return it as a dict."""
    with open(DEMO_CONFIG_PATH, encoding="utf-8") as f:  # the single settings file of the app
        cfg = yaml.safe_load(f)  # plain nested dict
    cfg["poc"]["gap_points"] = gap_points(cfg)  # average score gap per target, whole points
    cfg["poc"]["gap_text"] = {k: point_text(cfg, n) for k, n in cfg["poc"]["gap_points"].items()}  # the same gaps as text ("1 point", "2 points"), for the tab 6 tooltip, the closing slide and tab 5
    return cfg  # settings with the calibration gaps


def point_text(cfg: dict, n: int) -> str:  # a number of points as text
    """Singular text for one point, plural text for any other number, both from the config."""
    return cfg["ui"]["point_one" if n == 1 else "point_many"].format(n=n)  # config text


def gap_points(cfg: dict) -> dict:  # average gap between score and reality
    """target -> round(100 x ECE) from results/calibration/summary.csv, in whole points."""
    summary = pd.read_csv(ROOT / cfg["static_tabs"]["paths"]["calibration_summary"]).set_index("target")  # one row per target
    return {t: round(100 * float(e)) for t, e in summary["ece"].items()}  # whole points


@st.cache_data  # cache the table
def load_row_scores(path: str, columns: list) -> pd.DataFrame:  # cached WHAT holdout scores
    """Read the WHAT holdout scores with only `columns` and artist_id as int."""
    df = pd.read_csv(ROOT / path, usecols=columns)  # only the columns the app needs
    df["artist_id"] = df["artist_id"].astype("int64")  # float64 in the file, ids are whole numbers
    return df  # cached by Streamlit


@st.cache_data  # cache the table
def load_predictions(path: str, columns: list) -> pd.DataFrame:  # cached style forecast
    """Read the global style forecast with only `columns`."""
    return pd.read_csv(ROOT / path, usecols=columns)  # cached by Streamlit


def all_artists(cfg: dict) -> list:  # artists of both pools in one list, sorted by name
    """Artists of every pool in pool_b.pool_lists, in one list sorted by name (case-insensitive)."""
    mixed = [x for lst in cfg["pool_b"]["pool_lists"].values() for x in cfg[lst]]  # both pools together
    return sorted(mixed, key=lambda x: x["name"].lower())  # the pool is not shown, only used to pick files


def artists_by_genre(cfg: dict, genre: str | None) -> list:  # artists of both pools of one genre
    """Return the artists of both pools of `genre`, or all of them when genre is None, sorted by name."""
    return [a for a in all_artists(cfg) if genre is None or a["genre"] == genre]  # keep matching entries


def artist_year_rows(row_scores: pd.DataFrame, artist_id: int, year: int) -> pd.DataFrame:  # rows of one artist and year
    """Return the row_scores rows of one artist id in one year."""
    mask = (row_scores["artist_id"] == artist_id) & (row_scores["year"] == year)  # both conditions
    return row_scores[mask]  # may be empty


def add_projection_flag(predictions: pd.DataFrame, last_year: int, column: str) -> pd.DataFrame:  # forecast table with the projection flag
    """Return a copy of predictions with boolean `column`: target_year > last_year."""
    out = predictions.copy()  # do not modify the cached frame
    out[column] = out["target_year"] > last_year  # True for years without ground truth
    return out  # forecast table with the flag
