"""Live-scoring assets of the style forecast: per-year Random Forests, percentile table, panel rows, control.

Design fixed in docs/decisions.md (entries "Demo: live scoring with the final models" and the style
forecast entries it cites). Reuses the walk-forward, the count logic and the model helpers of style_forecast.
"""

import io  # in-memory model files for the size check
import json  # ordered column list
import time  # runtime

import joblib  # model files
import numpy as np  # arrays
import pandas as pd  # tables

from config import CONFIG, ROOT  # project settings and root
from style_forecast import LOCKED_RF_PARAMS, build_panel, count_bounds, feature_columns, fit_model, predict_bundle, walk_forward  # forecast functions
from style_series import keep_dated_styled, load_works  # series from the first phase

TOLERANCE = 0.00005  # largest accepted absolute difference (4 decimals)
SIZE_CAP_MB = 150  # largest accepted total size of the per-year models
FIRST, LAST = CONFIG["years"]["holdout_start"], CONFIG["years"]["holdout_end"] - 1  # test target years (2018 to 2022)
BASE_YEAR = CONFIG["years"]["holdout_end"]  # projection base year; its target year is the next one
PROJECTION = BASE_YEAR + 1  # projected target year
METHODS = ["random_forest", "repeat_last"]  # methods scored live
COUNT_KEY = {"random_forest": "random_forest", "repeat_last": "repeat_ct"}  # count method that goes with each method
VALUE_COLUMNS = ["expected_growth", "expected_works", "range_low", "range_high"]  # compared columns
OUT_DIR = ROOT / CONFIG["paths"]["style_live"]  # output folder, must not exist
PREDICTIONS = ROOT / CONFIG["paths"]["style_predictions"]  # stored table to reproduce
BUNDLE = ROOT / CONFIG["paths"]["style_bundle"]  # stored 2024 model, read only

def percentile_table(results: pd.DataFrame) -> pd.DataFrame:
    """10th and 90th percentile of the log residuals per target year and method, earlier target years pooled (all scored years for the projection)."""
    rows = []  # one row per target year and method
    for year in list(range(FIRST, LAST + 1)) + [PROJECTION]:  # test years, then the projection
        bounds = count_bounds(results[results["target"] < year])  # residuals of the folds scored before the year (every fold for the projection)
        for m in METHODS:  # both live methods
            rows.append({"target_year": year, "method": m, "q10": bounds[COUNT_KEY[m]][0], "q90": bounds[COUNT_KEY[m]][1]})  # percentiles
    return pd.DataFrame(rows)  # small table

def rescore(rows: pd.DataFrame, models: dict, pct: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the live rows (same columns as predictions.csv) from panel rows, one model per target year and the percentile table."""
    parts = []  # one block per target year and method
    for year, bundle in models.items():  # target year and its fitted forest
        frame = rows[rows["origin"] == year - 1]  # styles eligible at the origin before the target year
        score = {"random_forest": pd.Series(predict_bundle(bundle, frame), index=frame.index), "repeat_last": frame["gr0"]}  # ranking score per method
        expected = {"random_forest": frame["base"] * np.exp(score["random_forest"]), "repeat_last": frame["c_t"].clip(lower=1)}  # expected Works per method
        for m in METHODS:  # both methods
            q = pct[(pct["target_year"] == year) & (pct["method"] == m)].iloc[0]  # percentiles of this year and method
            parts.append(pd.DataFrame({
                "style": frame["style"], "target_year": year, "method": m,  # identifiers
                "expected_growth": np.exp(score[m]) - 1 if m == "random_forest" else score[m],  # growth over the base
                "expected_works": expected[m], "range_low": expected[m] * np.exp(q["q10"]), "range_high": expected[m] * np.exp(q["q90"]),  # count and range
                "rank": score[m].rank(ascending=False, method="first", na_option="bottom").astype(int),  # rank within the target year
            }))
    return pd.concat(parts, ignore_index=True)  # all rows stacked

def compare(live: pd.DataFrame, stored: pd.DataFrame, label: str) -> pd.DataFrame:
    """Compare live rows with the stored table, print the report and return the maximum differences per column and method."""
    stored = stored[stored["method"].isin(METHODS)]  # live methods only
    merged = live.merge(stored, on=["style", "target_year", "method"], how="outer", suffixes=("", "_stored"), indicator=True)  # row by row
    counts = pd.DataFrame({"live": live.groupby(["target_year", "method"]).size(), "stored": stored.groupby(["target_year", "method"]).size()})  # rows per year and method
    rows = []  # report rows
    for m in METHODS:  # per method
        both = merged[(merged["method"] == m) & (merged["_merge"] == "both")]  # matched rows
        sub = merged[merged["method"] == m]  # all rows of the method
        for c in VALUE_COLUMNS + ["rank"]:  # compared columns, rank reported apart
            rows.append({"check": label, "method": m, "column": c, "max_abs_diff": float((both[c] - both[c + "_stored"]).abs().max()),  # largest difference
                         "matched": len(both), "only_live": int((sub["_merge"] == "left_only").sum()), "only_stored": int((sub["_merge"] == "right_only").sum())})  # row match counts
    report = pd.DataFrame(rows)  # one row per method and column
    print(f"\n== {label}: rows per target year and method (live vs stored)")
    print(counts.to_string())
    print(f"\n== {label}: maximum absolute differences (rank is reported apart and does not fail the control)")
    print(report.to_string(index=False))
    return report

def passes(report: pd.DataFrame, live: pd.DataFrame, stored: pd.DataFrame) -> bool:
    """True when row counts match exactly and every value column is inside the tolerance (rank excluded)."""
    stored = stored[stored["method"].isin(METHODS)]  # live methods only
    same_counts = live.groupby(["target_year", "method"]).size().equals(stored.groupby(["target_year", "method"]).size())  # exact row counts
    values = report[report["column"].isin(VALUE_COLUMNS)]  # value columns only
    return bool(same_counts and (values["only_live"] + values["only_stored"]).sum() == 0 and (values["max_abs_diff"] <= TOLERANCE).all())  # all checks

def run() -> None:
    """Refit, control, size check, write, reload check."""
    if OUT_DIR.exists():  # never overwrite
        raise SystemExit(f"{OUT_DIR.relative_to(ROOT)} exists, nothing was run")  # stop before any work
    start = time.time()  # runtime clock
    kept, _ = keep_dated_styled(load_works())  # Works with a year and a style
    kept = kept[kept["year"] <= BASE_YEAR]  # nothing after the base year is read
    panel = build_panel(kept, BASE_YEAR)  # style by origin rows
    results = walk_forward(panel, last=LAST, allow_test=True, rf_params=LOCKED_RF_PARAMS)  # development years then test years, locked forest
    cols = feature_columns(panel)  # ordered inputs
    models = {}  # fitted forest per test target year
    for year in range(FIRST, LAST + 1):  # same training rows as walk_forward
        train = panel[(panel["origin"] <= year - 2) & panel["label"].notna()]  # origins up to year - 2 with a known label
        models[year] = fit_model("random_forest", train, cols, LOCKED_RF_PARAMS)  # locked forest and seed
    models[PROJECTION] = joblib.load(BUNDLE)  # stored model of the projection, read only
    pct = percentile_table(results)  # residual percentiles
    keep_cols = ["style", "origin", "c_t", "base", "gr0"] + cols  # columns the live rows need
    rows = panel[panel["origin"].isin([y - 1 for y in models])][keep_cols]  # panel rows of every origin needed
    stored = pd.read_csv(PREDICTIONS)  # table to reproduce
    live = rescore(rows, models, pct)  # rebuilt rows
    control = compare(live, stored, "control")  # comparison before saving
    if not passes(control, live, stored):  # any failure stops the step
        raise SystemExit("control failed, nothing was written")  # write nothing
    blobs = {}  # compressed model files in memory
    for year in range(FIRST, LAST + 1):  # per-year forests only
        buffer = io.BytesIO()  # in-memory file
        joblib.dump(models[year], buffer, compress=3)  # compressed joblib
        blobs[year] = buffer.getvalue()  # bytes
    total_mb = sum(len(b) for b in blobs.values()) / 1e6  # total size
    print(f"\nmodels in memory: {total_mb:.1f} MB (cap {SIZE_CAP_MB} MB)")
    if total_mb > SIZE_CAP_MB:  # size cap
        raise SystemExit("size cap exceeded, nothing was written")  # write nothing
    OUT_DIR.mkdir(parents=True)  # fails if it exists
    for year, blob in blobs.items():  # one file per test year
        with open(OUT_DIR / f"random_forest_{year}.joblib", "xb") as handle:  # exclusive creation
            handle.write(blob)  # model bytes
    for name, frame in (("percentiles.csv", pct), ("panel_rows.csv", rows), ("control.csv", control)):  # small tables
        with open(OUT_DIR / name, "x") as handle:  # exclusive creation
            frame.to_csv(handle, index=False)  # table
    with open(OUT_DIR / "columns.json", "x") as handle:  # exclusive creation
        json.dump({"random_forest": models[FIRST]["columns"]}, handle, indent=1)  # ordered inputs read from the fitted model
    reloaded = {y: joblib.load(OUT_DIR / f"random_forest_{y}.joblib") for y in range(FIRST, LAST + 1)}  # models from disk
    reloaded[PROJECTION] = joblib.load(BUNDLE)  # stored projection model
    again = rescore(pd.read_csv(OUT_DIR / "panel_rows.csv"), reloaded, pd.read_csv(OUT_DIR / "percentiles.csv"))  # rescore from the saved files
    check = compare(again, stored, "reload")  # same comparison
    print("\nreload check passed:", passes(check, again, stored))
    print("\n== files")
    for path in sorted(OUT_DIR.iterdir()):  # sizes from disk
        print(f"{path.name}  {path.stat().st_size / 1024:.1f} KB")
    print("rows: panel_rows", len(rows), "percentiles", len(pct), "live", len(live))
    print(f"runtime {time.time() - start:.0f} s")
