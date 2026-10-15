"""Score the final test target years 2018 to 2022 once, then project 2024 and write the outputs.

Same walk-forward procedure and settings as the development script, no pass rule.
Refuses to run when any output exists. Outputs are written to a temporary folder and renamed
only when everything is complete:
    results/style_forecast/predictions.csv             predictions for 2018 to 2022 and the 2024 projection
    results/style_forecast/models/<model>.joblib       one fitted model per model (fitted on origins up to 2022)

Usage:
    venv/bin/python3 scripts/run_style_forecast_test.py
"""

import sys  # module search path and exit
from pathlib import Path  # project root

import joblib  # model files
import pandas as pd  # tables

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from style_forecast import BASELINES, LOCKED_RF_PARAMS, MODELS, boot_ci, build_panel, count_metrics, count_table, hits_table, long_table, paired_differences, projection, spearman_table, walk_forward  # forecast functions
from style_series import keep_dated_styled, load_works  # series from the first phase

TEST_FIRST, TEST_LAST, BASE_YEAR = 2018, 2022, 2023  # test target years and the projection base year
OUT_DIR = ROOT / "results/style_forecast"  # folder of this experiment
OUT_CSV = OUT_DIR / "predictions.csv"  # final table
OUT_MODELS = OUT_DIR / "models"  # final model folder
TMP = OUT_DIR / ".tmp"  # temporary folder, renamed away at the end


def report(results: pd.DataFrame, first: int) -> pd.DataFrame:
    """Print the test tables for target years from `first` on and return the count rows of those years."""
    methods = ["random"] + BASELINES + MODELS  # every method in report order
    test = results[results["target"] >= first]  # scored years only
    for k in (10, 5):  # both cut-offs
        table = hits_table(test, k)  # hits per year
        print(f"\n== hits@{k} per target year")
        print(table[["eligible"] + methods].round(2).to_string())
        print(f"-- hits@{k} mean over the {len(table)} years and 95% interval (bootstrap over years, 2,000 draws, seed 1)")
        print(pd.DataFrame({m: boot_ci(table[m].to_numpy(dtype=float)) for m in methods}, index=["mean", "low", "high"]).T.round(2).to_string())
    spearman = spearman_table(test)  # rank correlation per year
    print("\n== Spearman on the styles scored by every method, per year")
    print(spearman.round(3).to_string())
    print("mean over the years:")
    print(spearman[BASELINES + MODELS].mean().round(3).to_string())
    counts = count_table(results)  # ranges use the residuals of all earlier folds, development first
    counts = counts[counts["target"] >= first]  # metrics for the scored years
    print(f"\n== count metrics, target years {first} to {test['target'].max()}, nominal coverage 0.80")
    print(count_metrics(counts, results).round(3).to_string(index=False))
    print("\n== descriptive paired differences, each model minus repeat last year (mean and 95% interval, bootstrap over years, 2,000 draws, seed 1)")
    print(f"-- development years, {results[results['target'] < first]['target'].min()} to {first - 1}")
    print(paired_differences(results[results["target"] < first]).round(3).to_string())  # development part
    print(f"-- test years, {first} to {test['target'].max()}")
    print(paired_differences(test).round(3).to_string())  # test part
    return counts


def main(first: int = TEST_FIRST, last: int = TEST_LAST, base_year: int = BASE_YEAR, write: bool = True) -> None:
    """Score years first..last once, project base_year + 1, optionally write the outputs."""
    if write and any(p.exists() for p in (TMP, OUT_CSV, OUT_MODELS)):  # never run twice
        sys.exit("an output already exists, nothing was run")  # stop before reading any test year
    kept, _ = keep_dated_styled(load_works())  # Works with a year and a style
    kept = kept[kept["year"] <= base_year]  # nothing after the base year is read
    panel = build_panel(kept, base_year)  # style by origin rows
    results = walk_forward(panel, last=last, allow_test=True, rf_params=LOCKED_RF_PARAMS)  # development years first, then the test years, locked forest
    counts = report(results, first)  # print the test tables
    frame, bundles = projection(panel, results, base_year, LOCKED_RF_PARAMS)  # fit on origins up to base_year - 1 with the locked forest, predict origin base_year
    top = frame.sort_values("repeat_last", ascending=False, na_position="last").head(10)  # recommended ranking, top 10
    shown = top[["style", "repeat_last", "c_t", "base"]].rename(columns={"repeat_last": "growth", "c_t": f"works_{base_year}"})  # ranking columns
    for m in MODELS:  # expected Works and range of each model
        shown[m] = [f"{e:.0f} ({lo:.0f}-{hi:.0f})" for e, lo, hi in zip(top[m + "_exp"], top[m + "_lo"], top[m + "_hi"])]
    print(f"\n== projection for {base_year + 1}: top 10 by repeat last year, expected Works (range 10th-90th percentile)")
    print(shown.round(3).to_string(index=False))
    if not write:  # dry run, nothing is written
        return  # stop here
    scored = counts.merge(results[["target", "style", "gr0"] + MODELS], on=["target", "style"])  # attach the scores to the count columns
    scored["repeat_last"] = scored["gr0"]  # recommended ranking score
    table = pd.concat([long_table(scored), long_table(frame)], ignore_index=True)  # test years then the projection
    TMP.mkdir()  # temporary folder, fails if it exists
    (TMP / "models").mkdir()  # temporary model folder
    with open(TMP / "predictions.csv", "x") as handle:  # exclusive mode
        table.to_csv(handle, index=False)  # the one table
    for m, bundle in bundles.items():  # one file per fitted model
        with open(TMP / "models" / f"{m}.joblib", "xb") as handle:  # exclusive mode
            joblib.dump(bundle, handle)  # model, columns and scaling statistics
    assert len(list((TMP / "models").iterdir())) == len(MODELS) and (TMP / "predictions.csv").stat().st_size > 0, "outputs incomplete"  # everything is there
    OUT_DIR.mkdir(exist_ok=True)  # experiment folder (already there when the development file was written)
    (TMP / "models").rename(OUT_MODELS)  # results/style_forecast/models/
    (TMP / "predictions.csv").rename(OUT_CSV)  # results/style_forecast/predictions.csv
    TMP.rmdir()  # temporary folder is empty now
    print("\nwritten:", OUT_CSV, "and", OUT_MODELS)


if __name__ == "__main__":
    main()
