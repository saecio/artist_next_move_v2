"""Live data layer of the demo app: cached asset loaders, live WHEN, WHAT and style scores, artist timeline."""

import json  # ordered column lists

import joblib  # style models
import pandas as pd  # tables
import streamlit as st  # cache decorators
from xgboost import XGBClassifier  # WHEN and WHAT models

from app_data import load_demo_config  # demo settings
from config import CONFIG, ROOT  # project settings and root
from demo_game import outside_rate  # holdout rate below the WHAT cutoff
from live_style_assets import FIRST, LAST, PROJECTION, rescore  # style rescoring and its target years
from when_demo import TARGETS, demo_artist_ids  # WHEN targets and the demo artist ids

FIRST_SHOWN = CONFIG["years"]["demo_works_first_year"]  # first year of the timeline
WHEN_STORED = ["label", "is_projection", "label_after_holdout"]  # stored WHEN columns spread by target


def _live() -> dict:  # live section of the demo config
    """The live section of app/demo_config.yaml."""
    return load_demo_config()["live"]  # asset paths and file names


@st.cache_resource  # cache the shared models
def load_models() -> dict:  # load the fitted models
    """Fitted models read from disk: WHEN per target, WHAT stage 0 and the style forests per target year."""
    live, folder = _live(), ROOT / _live()["folder"]  # settings and live folder
    models = {}  # name -> fitted model
    for target in TARGETS:  # one WHEN model per target
        models[f"when_{target}"] = XGBClassifier()  # empty model
        models[f"when_{target}"].load_model(folder / live["when_models"][target])  # native JSON file
    models["what"] = XGBClassifier()  # empty model
    models["what"].load_model(folder / live["what_model"])  # native JSON file
    style = ROOT / live["style_folder"]  # style folder
    models["style"] = {y: joblib.load(style / live["style_model_pattern"].format(year=y)) for y in range(FIRST, LAST + 1)}  # per-year forests
    models["style"][PROJECTION] = joblib.load(ROOT / CONFIG["paths"]["style_bundle"])  # stored 2024 bundle, read only
    return models  # shared by every session


@st.cache_data  # cache the tables
def load_tables() -> dict:  # load the cached tables
    """Feature files and stored files; the stored files are cut to the demo artists."""
    live, folder, ids = _live(), ROOT / _live()["folder"], demo_artist_ids()  # settings, live folder, demo ids
    what = pd.read_csv(ROOT / live["what_scores"])  # stored WHAT scores of all artists
    what["artist_id"] = what["artist_id"].astype("int64")  # whole-number ids
    style = ROOT / live["style_folder"]  # style folder
    top = pd.read_csv(ROOT / live["what_top10"]).set_index("model").loc[live["what_top10_model"]]  # stage 0 row of the holdout top 10% file
    pb = load_demo_config()["pool_b"]["paths"]  # pool B file paths
    def both(a_path, b_key):  # pool A file followed by the pool B file
        return pd.concat([pd.read_csv(a_path), pd.read_csv(ROOT / pb[b_key])], ignore_index=True)  # one table, pool A rows first and unchanged
    return {  # open the table dict
        "columns": json.loads((folder / live["columns"]).read_text()),  # ordered model columns
        "when_features": both(folder / live["when_features"], "when_features"),  # WHEN feature rows, both pools
        "what_features": both(folder / live["what_features"], "what_features"),  # WHAT feature rows, both pools
        "when_stored": both(ROOT / live["when_scores"], "when_scores"),  # stored WHEN scores, both pools
        "what_top10": top.loc[["score_cutoff", "precision_top", "holdout_positive_rate"]].astype(float).to_dict(),  # WHAT cutoff and holdout figures of the top 10% group
        "what_below": outside_rate(what, float(top["score_cutoff"])),  # holdout rows below the cutoff: counts, rate and z
        "what_labels": both(folder / live["what_labels"], "what_labels"),  # stored WHAT labels and after-holdout flags, both pools
        "what_stored": what[what["artist_id"].isin(ids)],  # stored WHAT scores, demo artists
        "works": both(ROOT / live["works"], "works"),  # Works of the demo artists, both pools
        "panel": pd.read_csv(style / live["style_panel"]),  # style panel rows
        "percentiles": pd.read_csv(style / live["style_percentiles"]),  # residual percentiles
        "predictions": pd.read_csv(ROOT / live["style_predictions"]),  # stored style forecast, validation display only
    }  # close the table dict


def score_when(artist_id: int) -> pd.DataFrame:  # live WHEN scores of one artist
    """Live next_1y and next_2y scores for every Work of the artist in the WHEN feature file."""
    models, tables = load_models(), load_tables()  # cached assets
    rows = tables["when_features"]  # feature rows of all demo artists
    rows = rows[rows["artist_id"] == artist_id]  # this artist only
    out = rows[["artist_id", "snapshot_year", "work_id"]].copy()  # row identity
    for target in TARGETS:  # one score per target
        out[target] = models[f"when_{target}"].predict_proba(rows[tables["columns"][f"when_{target}"]])[:, 1]  # probability of the positive class
    return out.reset_index(drop=True)  # clean index


def score_what(artist_id: int) -> pd.DataFrame:  # live WHAT scores of one artist
    """Live stage 0 score for every artist-year in the WHAT feature file, with the stored label and baseline (c)."""
    models, tables = load_models(), load_tables()  # cached assets
    rows = tables["what_features"]  # feature rows of all demo artists
    rows = rows[rows["artist_id"] == artist_id]  # this artist only
    out = rows[["artist_id", "year"]].copy()  # row identity
    if rows.empty:  # artist without any WHAT score
        return out.assign(stage0_score=pd.Series(dtype=float), label=pd.Series(dtype=float), label_after_holdout=pd.Series(dtype=bool), baseline_c_score=pd.Series(dtype=float))  # empty table with the same columns
    out["stage0_score"] = models["what"].predict_proba(rows[tables["columns"]["what_stage0"]])[:, 1]  # live score
    labels = tables["what_labels"][["artist_id", "year", "label", "label_after_holdout"]]  # stored label and after-holdout flag
    baseline = tables["what_stored"][["artist_id", "year", "baseline_c_score"]]  # lookup baseline, holdout rows only
    return out.merge(labels, on=["artist_id", "year"], how="left").merge(baseline, on=["artist_id", "year"], how="left").reset_index(drop=True)  # label, flag and baseline next to the score


def artist_timeline(artist_id: int) -> pd.DataFrame:  # one timeline row per year
    """One row per year from the first shown year to the last year of the data; years without Works stay empty."""
    tables = load_tables()  # cached tables
    works = tables["works"]  # Works of the demo artists
    last_year = int(works["year"].max())  # last year of the data
    rep = works[(works["artist_id"] == artist_id) & works["is_representative"]]  # representative Work of each year
    cols = ["year", "work_id", "title", "genres", "styles", "is_rerecording"]  # Work columns shown
    base = pd.DataFrame({"year": range(FIRST_SHOWN, last_year + 1)}).merge(rep[cols], on="year", how="left")  # gap years stay as empty rows
    base.insert(0, "artist_id", artist_id)  # same id on every row
    base = base.merge(score_when(artist_id).drop(columns=["artist_id", "snapshot_year"]), on="work_id", how="left")  # live WHEN scores
    stored = tables["when_stored"]  # stored WHEN labels
    stored = stored[stored["artist_id"] == artist_id]  # this artist only
    wide = stored.pivot(index="work_id", columns="target", values=WHEN_STORED)  # stored columns spread by target
    wide.columns = [f"{name}_{target}" for name, target in wide.columns]  # flat names such as label_next_1y
    base = base.merge(wide.reset_index(), on="work_id", how="left")  # stored labels and flags
    what = score_what(artist_id)[["year", "stage0_score", "label", "label_after_holdout", "baseline_c_score"]].rename(columns={"label": "what_label", "label_after_holdout": "what_label_after_holdout"})  # WHAT covers every eligible year from the holdout start
    base = base.merge(what, on="year", how="left")  # live WHAT score, stored label and baseline
    base["artist_train_rows"] = stored["artist_train_rows"].iloc[0] if len(stored) else pd.NA  # for the seen / not seen warning
    return base  # one row per year


def style_live(target_year: int) -> pd.DataFrame:  # style table of one target year
    """Random Forest and repeat_last table of one target year, rescored from the saved models, panel rows and percentiles."""
    tables = load_tables()  # cached tables
    model = {target_year: load_models()["style"][target_year]}  # only the model of this year
    return rescore(tables["panel"], model, tables["percentiles"])  # same columns as predictions.csv
