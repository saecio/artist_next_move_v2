"""Extra live style assets: the Random Forest of target year 2023, its panel rows, percentiles and the actual counts, with controls."""

import hashlib  # SHA-256 of the existing files
import io  # in-memory model file for the size check
import json  # ordered column list of the saved model
import time  # runtime of the step

import joblib  # model file
import numpy as np  # NaN-aware comparisons
import pandas as pd  # tables

from config import CONFIG, ROOT  # project settings and root
from live_style_assets import COUNT_KEY, METHODS, OUT_DIR, PREDICTIONS, SIZE_CAP_MB, TOLERANCE, rescore  # same rescoring and limits as the first style assets step
from style_forecast import LOCKED_RF_PARAMS, build_panel, count_bounds, feature_columns, fit_model, predict_bundle, style_table, top_hits, walk_forward  # same procedure and builders
from style_series import keep_dated_styled, load_works  # series from the first phase

TARGET = CONFIG["years"]["holdout_end"]  # target year of the new model (2023)
ORIGIN = TARGET - 1  # origin year, styles eligible here are forecast
FIRST_TEST = CONFIG["years"]["holdout_start"]  # first final test target year
LAST_TEST = TARGET - 1  # last final test target year
REALITY_YEAR = TARGET + 1  # display only reality year (2024), the projection year
NEW = {"model": OUT_DIR / f"random_forest_{TARGET}.joblib", "panel": OUT_DIR / "panel_rows_extra.csv", "percentiles": OUT_DIR / "percentiles_extra.csv", "actuals": OUT_DIR / "actuals.csv", "control": OUT_DIR / "control_extra.csv"}  # new files, never overwritten
TEST_LOG = ROOT / "results/style_forecast/test_20261015_173215.out"  # log of the final test, read only, source of the stored hits per year


def hashes() -> dict:  # SHA-256 of the existing files
    """SHA-256 of every existing file under the live style folder and results/style_forecast, except the new files."""
    skip = {p.resolve() for p in NEW.values()}  # files this step creates
    folders = [OUT_DIR, ROOT / "results/style_forecast"]  # folders that must not change
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for f in folders for p in sorted(f.rglob("*")) if p.is_file() and p.name != ".DS_Store" and p.resolve() not in skip}  # path -> hash


def stored_hits() -> dict:  # hits per test year from the log
    """Random Forest top 10 hits per final test target year, read from the stored test log."""
    lines = TEST_LOG.read_text(encoding="utf-8").split("\n")  # log lines
    i = next(n for n, l in enumerate(lines) if l.startswith("== hits@10 per target year"))  # start of the table
    head = lines[i + 1].split()  # method names
    col = head.index("random_forest")  # column of the Random Forest
    return {int(l.split()[0]): float(l.split()[1 + col]) for l in lines[i + 2:i + 2 + (LAST_TEST - FIRST_TEST + 1)]}  # year -> hits


def actuals_table(panel: pd.DataFrame, kept_all: pd.DataFrame) -> pd.DataFrame:  # actual Works per style and target year
    """Actual counts and labels for the target years FIRST_TEST to TARGET from the panel, and the display only count of REALITY_YEAR from the style table."""
    parts = []  # one block per target year
    for year in range(FIRST_TEST, TARGET + 1):  # final test years and the flagged 2023
        f = panel[panel["origin"] == year - 1]  # styles eligible at the origin before the target year
        parts.append(pd.DataFrame({"style": f["style"].to_numpy(), "target_year": year, "actual_works": f["c_next"].to_numpy(), "label": f["label"].to_numpy()}))  # c_next and label of build_panel
    table = style_table(kept_all[kept_all["year"] <= REALITY_YEAR], REALITY_YEAR)  # style by year counts up to the display year
    g = panel[panel["origin"] == TARGET]  # styles of the projection
    parts.append(pd.DataFrame({"style": g["style"].to_numpy(), "target_year": REALITY_YEAR, "actual_works": table.loc[g["style"], REALITY_YEAR].to_numpy(), "label": np.nan}))  # display only, no label
    return pd.concat(parts, ignore_index=True)  # all rows


def run() -> None:  # whole step
    """Refit the 2023 forest, control, then write the new files with exclusive creation; nothing is written if a control fails."""
    start = time.time()  # clock of the step
    for p in NEW.values():  # never overwrite
        if p.exists():  # a new file exists
            raise SystemExit(f"{p.relative_to(ROOT)} already exists, nothing was run")  # stop early
    before = hashes()  # hashes before the run
    kept_all, _ = keep_dated_styled(load_works())  # Works with a year and a style
    kept = kept_all[kept_all["year"] <= TARGET]  # nothing after the base year is read for the model
    panel = build_panel(kept, TARGET)  # style by origin rows
    cols = feature_columns(panel)  # ordered inputs
    train = panel[(panel["origin"] <= TARGET - 2) & panel["label"].notna()]  # walk_forward rule: origins up to T - 2 with a known label
    model = fit_model("random_forest", train, cols, LOCKED_RF_PARAMS)  # locked forest and seed
    keep_cols = ["style", "origin", "c_t", "base", "gr0"] + cols  # columns the live rows need
    rows = panel[panel["origin"] == ORIGIN][keep_cols]  # panel rows of the origin
    results = walk_forward(panel, last=LAST_TEST, allow_test=True, rf_params=LOCKED_RF_PARAMS)  # folds scored before the target year, for the residuals
    bounds = count_bounds(results[results["target"] < TARGET])  # residual percentiles, earlier target years only
    pct = pd.DataFrame([{"target_year": TARGET, "method": m, "q10": bounds[COUNT_KEY[m]][0], "q90": bounds[COUNT_KEY[m]][1]} for m in METHODS])  # percentile rows of the new year
    live = rescore(rows, {TARGET: model}, pct)  # live rows of the new year
    actuals = actuals_table(panel, kept_all)  # actual counts and labels
    out = []  # control rows
    def add(name, n, unmatched, diff, limit):  # one control row
        out.append({"control": name, "rows": n, "unmatched": unmatched, "max_abs_diff": float(diff), "limit": limit, "passed": bool(diff <= limit and unmatched == 0)})  # pass rule
    wf = walk_forward(panel, first=TARGET, last=TARGET, allow_test=True, rf_params=LOCKED_RF_PARAMS).set_index("style")  # independent single-year run of the procedure
    mine = pd.Series(predict_bundle(model, rows), index=rows["style"].to_numpy())  # predicted log growth of the new model
    both = mine.index.intersection(wf.index)  # shared styles
    add("1 new model equals an independent walk_forward fit for the target year", len(both), len(mine.index.symmetric_difference(wf.index)), (mine[both] - wf.loc[both, "random_forest"]).abs().max(), TOLERANCE)  # scores reproduce
    cut = build_panel(kept_all[kept_all["year"] <= ORIGIN], ORIGIN)  # panel from Works dated up to the origin only
    cut_rows = cut[cut["origin"] == ORIGIN][keep_cols].set_index("style")  # origin rows from the cut data
    full_rows = rows.set_index("style")  # origin rows from the full build
    shared = full_rows.index.intersection(cut_rows.index)  # shared styles
    diff = (full_rows.loc[shared, [c for c in keep_cols if c not in ("style", "origin")]].astype(float) - cut_rows.loc[shared, [c for c in keep_cols if c not in ("style", "origin")]].astype(float)).abs().to_numpy().max()  # feature difference
    add("2 point-in-time: origin rows from Works dated up to the origin equal the full build", len(shared), len(full_rows.index.symmetric_difference(cut_rows.index)), diff, 1e-9)  # features equal
    stored_pct = pd.read_csv(OUT_DIR / "percentiles.csv")  # stored percentiles
    s24 = stored_pct[(stored_pct["target_year"] == TARGET + 1)].set_index("method")  # stored projection row, same pool of folds
    add("3 percentiles equal the stored projection row (same folds, target years 1993 to 2022)", len(METHODS), 0, max(abs(pct.set_index("method").loc[m, c] - s24.loc[m, c]) for m in METHODS for c in ("q10", "q90")), 1e-9)  # pool equal
    add("4 model columns equal columns.json", len(cols), int(cols != json.loads((OUT_DIR / "columns.json").read_text())["random_forest"]), 0.0, 0.0)  # column control
    pred = pd.read_csv(PREDICTIONS)  # stored forecast table
    pred = pred[(pred["method"] == "random_forest") & pred["target_year"].between(FIRST_TEST, LAST_TEST)]  # test years, Random Forest
    merged = pred.merge(actuals[["style", "target_year", "label"]], on=["style", "target_year"], how="left")  # label of every stored row
    mine_hits = {int(y): float(top_hits(g["expected_growth"].to_numpy(), g["label"].to_numpy(), 10)) for y, g in merged.groupby("target_year")}  # hits per year
    want = stored_hits()  # hits from the stored log
    add("5 hits at 10 per test year from actuals.csv equal the stored test log", len(want), len(set(mine_hits) ^ set(want)), max(abs(mine_hits[y] - want[y]) for y in want), 0.0)  # hits reproduce
    table = style_table(kept_all[kept_all["year"] <= REALITY_YEAR], REALITY_YEAR)  # independent counts
    ind = actuals.assign(count=lambda a: [table.loc[s, y] for s, y in zip(a["style"], a["target_year"])])  # count of each style and year from the table
    add("6 actual Works equal an independent count from the style table", len(ind), 0, (ind["actual_works"] - ind["count"]).abs().max(), 0.0)  # counts equal
    add("7 no duplicated (style, target_year) in actuals", len(actuals), int(actuals.duplicated(["style", "target_year"]).sum()), 0.0, 0.0)  # key control
    add("8 forecast rows equal the eligible styles of the origin, ranks 1 to n", len(live[live["method"] == "random_forest"]), int(len(rows) != len(live[live["method"] == "random_forest"]) or sorted(live[live["method"] == "random_forest"]["rank"]) != list(range(1, len(rows) + 1))), 0.0, 0.0)  # row and rank control
    buffer = io.BytesIO()  # in-memory model file
    joblib.dump(model, buffer, compress=3)  # compressed joblib, as in the first step
    existing = sum(p.stat().st_size for p in OUT_DIR.glob("random_forest_*.joblib"))  # existing per-year models
    total_mb = (existing + len(buffer.getvalue())) / 1e6  # total size of the per-year models
    add(f"9 total size of the per-year models under {SIZE_CAP_MB} MB", 6, 0, max(total_mb - SIZE_CAP_MB, 0.0), 0.0)  # size cap
    table_c = pd.DataFrame(out)  # control table
    pd.set_option("display.width", 250, "display.max_colwidth", 100)  # readable table
    print(table_c.to_string(index=False), flush=True)  # control output
    if not table_c["passed"].all():  # any failure stops the step
        raise SystemExit("a control failed, nothing was written")  # write nothing
    with open(NEW["model"], "xb") as handle:  # exclusive creation
        handle.write(buffer.getvalue())  # model bytes
    for key, frame in (("panel", rows), ("percentiles", pct), ("actuals", actuals)):  # small tables
        with open(NEW[key], "x", encoding="utf-8", newline="") as handle:  # exclusive creation
            frame.to_csv(handle, index=False)  # table
    reloaded = joblib.load(NEW["model"])  # model from disk
    again = rescore(pd.read_csv(NEW["panel"]), {TARGET: reloaded}, pd.read_csv(NEW["percentiles"]))  # rescore from the saved files
    reload_diff = max(float((again[c].to_numpy() - live[c].to_numpy()).__abs__().max()) for c in ("expected_growth", "expected_works", "range_low", "range_high"))  # reload difference
    out.append({"control": "10 reload check: saved model and files rescore to the same rows", "rows": len(again), "unmatched": int(len(again) != len(live)), "max_abs_diff": reload_diff, "limit": TOLERANCE, "passed": bool(reload_diff <= TOLERANCE and len(again) == len(live))})  # reload control
    out.append({"control": f"11 existing files unchanged ({len(before)} files hashed)", "rows": len(before), "unmatched": int(before != hashes()), "max_abs_diff": 0.0, "limit": 0.0, "passed": before == hashes()})  # hash control
    table_c = pd.DataFrame(out)  # full control table
    with open(NEW["control"], "x", encoding="utf-8", newline="") as handle:  # exclusive creation
        table_c.to_csv(handle, index=False)  # control table
    print("\n" + table_c.tail(2).to_string(index=False))  # reload and hash controls
    if not table_c["passed"].all():  # a late control failed, the files stay for inspection
        raise SystemExit("a control failed after writing")  # report
    print(f"\ntrain rows {len(train)} (origins up to {TARGET - 2}), forecast rows {len(rows)}, actuals rows {len(actuals)}, per-year model total {total_mb:.1f} MB (cap {SIZE_CAP_MB} MB)")  # figures
    print(f"percentiles {TARGET}: " + ", ".join(f"{r['method']} q10 {r['q10']:.4f} q90 {r['q90']:.4f}" for _, r in pct.iterrows()))  # residual quantiles
    print("hits at 10 per test year (actuals.csv vs stored log):", {y: (mine_hits[y], want[y]) for y in sorted(want)})  # hits
    print("styled Works per year:", {y: int(table[y].sum()) for y in (2023, 2024)})  # counts up to the display year
    print("\nfiles:")  # section title
    for key, p in NEW.items():  # sizes from disk
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.1f} KB" + (f"  {len(pd.read_csv(p))} rows" if p.suffix == ".csv" else ""))  # name, size, rows
    print(f"runtime {time.time() - start:.0f} s")  # runtime
