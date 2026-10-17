"""Live-scoring assets of the demo: refit the final WHEN and WHAT models, feature rows of the demo artists, control."""

import json  # column lists and control result
import time  # runtime of the step

import pandas as pd  # dataframes

from backtesting import holdout_gap, split_holdout  # same training rows as the WHEN holdout run
from config import CONFIG, ROOT  # project settings and root
from models import FINAL_XGB_PARAMS, WHAT_FEATURES, _what_matrix, build_features, make_what_xgboost, make_xgboost  # locked models and matrices
from tuning import read_log  # trial log reader
from what_target import build_artist_years, build_what_features, build_what_table  # WHAT table and inputs
from when_demo import HOLDOUT_START, TARGETS, demo_artist_ids  # demo years, WHEN targets, demo artist ids

TOLERANCE = 0.00005  # largest accepted absolute score difference (4 decimals)
WHAT_TARGET = "new_style_year"  # WHAT target
WHAT_LOG = ROOT / CONFIG["paths"]["results"] / "trials/xgb_what_new_style_year.jsonl"  # trial log with the stage0 row
RESULTS = ROOT / CONFIG["paths"]["results"]  # results folder
OUT_DIR = RESULTS / "demo/live"  # folder of all live assets, must not exist before the run
WHEN_ROWS = RESULTS / "demo/when_demo_scores.csv"  # stored WHEN scores
WHAT_ROWS = RESULTS / "holdout_what/row_scores.csv"  # stored WHAT scores


def refuse_if_written() -> None:
    """Stop before any work when the output folder already exists."""
    if OUT_DIR.exists():  # never overwrite
        raise SystemExit(f"{OUT_DIR.relative_to(ROOT)} already exists, nothing was run")  # stop here


def when_assets(ids: list) -> tuple:
    """Refit WHEN next_1y and next_2y, return (models, demo feature rows, columns, control rows)."""
    snapshots = pd.read_parquet(ROOT / CONFIG["paths"]["snapshots"])  # cached snapshots
    final_train, _ = holdout_gap(split_holdout(snapshots)["development"])  # training rows of the holdout run
    rows = snapshots[(snapshots["year"] >= HOLDOUT_START) & snapshots["artist_id"].isin(ids)].copy()  # demo snapshots from the holdout start on
    rows["artist_id"] = rows["artist_id"].astype("int64")  # whole-number ids
    matrix = build_features(rows)  # model input columns
    keys = pd.DataFrame({"artist_id": rows["artist_id"], "snapshot_year": rows["year"], "work_id": rows["work_id"]})  # row identity
    features = pd.concat([keys, matrix], axis=1)  # one small table for both targets
    stored = pd.read_csv(WHEN_ROWS)  # stored scores of both targets
    models, columns, control = {}, {}, []  # outputs
    for target in TARGETS:  # one fit per target
        fit_fn, _ = make_xgboost(target, **FINAL_XGB_PARAMS[target])  # locked configuration and seed
        model = fit_fn(final_train)  # fitted on the holdout training rows
        columns[target] = list(model.feature_names_in_)  # columns the model expects, read from the model
        assert columns[target] == list(matrix.columns), "model columns differ from the feature rows"  # same names and order
        scored = keys.assign(live=model.predict_proba(matrix[columns[target]])[:, 1])  # live scores
        both = scored.merge(stored[stored["target"] == target], on=["artist_id", "snapshot_year", "work_id"], how="outer", indicator=True)  # stored vs live
        assert (both["_merge"] == "both").all(), f"{target}: rows differ between live and stored scores"  # same rows
        control.append({"model": f"when_{target}", "rows": len(both), "max_abs_diff": float((both["live"] - both["score"]).abs().max())})  # control row
        models[f"when_{target}"] = model  # keep the fitted model
    return models, features, columns, control  # all WHEN pieces


def what_assets(ids: list) -> tuple:
    """Refit WHAT stage 0, return (model, demo feature rows, columns, control row)."""
    params = {r["trial"]: r for r in read_log(WHAT_LOG)}["stage0"]["params"]  # stage0 parameters from the log, not typed
    works = pd.read_parquet(ROOT / CONFIG["paths"]["works"])  # cleaned Works table
    table = build_what_table(build_artist_years(works))  # labelled artist-year rows
    features = build_what_features(works)  # WHAT feature table
    start, end = HOLDOUT_START, CONFIG["years"]["holdout_end"]  # holdout years
    train = table[(table["year"] < start) & (table["next_active_year"] < start)]  # label resolved before the holdout
    holdout = table[(table["year"] >= start) & (table["year"] <= end) & (table["next_active_year"] <= end)]  # holdout rows, label rule
    fit_fn, _ = make_what_xgboost(features, target=WHAT_TARGET, **params)  # stage 0 with the logged parameters
    model = fit_fn(train)  # fitted on the training rows
    assert model.get_params()["n_estimators"] == params["n_estimators"], "logged parameters not in the model"  # parameter check
    demo = holdout[holdout["artist_id"].isin(ids)]  # demo artists' holdout rows
    matrix = _what_matrix(demo, features, WHAT_FEATURES)  # model input columns
    columns = list(model.feature_names_in_)  # columns the model expects, read from the model
    assert columns == list(matrix.columns), "model columns differ from the feature rows"  # same names and order
    keys = pd.DataFrame({"artist_id": demo["artist_id"].astype("int64"), "year": demo["year"]})  # row identity
    scored = keys.assign(live=model.predict_proba(matrix[columns])[:, 1])  # live scores
    stored = pd.read_csv(WHAT_ROWS)  # stored scores
    stored = stored[stored["artist_id"].isin(ids)]  # demo artists only
    both = scored.merge(stored, on=["artist_id", "year"], how="outer", indicator=True)  # stored vs live
    assert (both["_merge"] == "both").all(), "what: rows differ between live and stored scores"  # same rows
    control = {"model": "what_stage0", "rows": len(both), "max_abs_diff": float((both["live"] - both["stage0_score"]).abs().max())}  # control row
    return model, pd.concat([keys, matrix], axis=1), columns, control  # all WHAT pieces


def run() -> None:
    """Refit, control, then write everything with exclusive creation. Writes nothing if the control fails."""
    start = time.time()  # clock of the step
    refuse_if_written()  # stop early
    ids = demo_artist_ids()  # the 40 demo artists
    when_models, when_features, when_columns, control = when_assets(ids)  # WHEN pieces
    what_model, what_features, what_columns, what_control = what_assets(ids)  # WHAT pieces
    control = pd.DataFrame(control + [what_control])  # control table
    print(control.to_string(index=False), flush=True)  # maximum differences per model
    if (control["max_abs_diff"] > TOLERANCE).any():  # control failed
        raise SystemExit("control failed, nothing was written")  # stop before any file
    OUT_DIR.mkdir(parents=True)  # fails if it exists
    models = {**when_models, "what_stage0": what_model}  # three models
    written = []  # files for the report
    for name, model in models.items():  # native XGBoost format
        path = OUT_DIR / f"{name}.json"  # model file
        model.get_booster().save_model(str(path))  # native format, not pickle
        written.append(path)  # keep
    for name, frame in (("when_features", when_features), ("what_features", what_features)):  # feature rows
        path = OUT_DIR / f"{name}.csv"  # feature file
        with open(path, "x") as handle:  # exclusive creation
            frame.to_csv(handle, index=False)  # small table
        written.append(path)  # keep
    for name, payload in (("columns.json", {"when_next_1y": when_columns["next_1y"], "when_next_2y": when_columns["next_2y"], "what_stage0": what_columns}),):  # expected columns
        path = OUT_DIR / name  # columns file
        with open(path, "x") as handle:  # exclusive creation
            json.dump(payload, handle, indent=1)  # ordered column lists
        written.append(path)  # keep
    path = OUT_DIR / "control.csv"  # control result
    with open(path, "x") as handle:  # exclusive creation
        control.to_csv(handle, index=False)  # maximum differences
    written.append(path)  # keep
    for p in written:  # report
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1024:.1f} KB")  # file and size
    print(f"WHEN feature rows {len(when_features)}, WHAT feature rows {len(what_features)}, runtime {time.time() - start:.0f} s")  # counts and runtime
