"""Duplicate sensitivity analysis on the development partition (collapse to the earliest work of each artist and title).

Stage pilot: collapse the works, rebuild both data versions, reproduce the stored dev per-fold PR-AUC on the original data,
and only if they match fit one fold on both versions and project the run time of the full batch.

Usage:
    venv/bin/python3 scripts/run_duplicate_sensitivity.py pilot
"""

import sys  # arguments, module search path and exit
import time  # clocks
from pathlib import Path  # project root

import numpy as np  # arrays
import pandas as pd  # dataframes

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from duplicate_sensitivity import TARGETS, WHAT_TARGET, analyze, bootstrap_sets, build_version, collapse_works, configs, fit_fold, get_folds, point_metrics, verdicts  # analysis pieces
from tuning import read_log  # trial log reader

pd.set_option("display.width", 250)  # readable printed tables


def stored_pr_auc(target: str, config: str) -> list:
    """Dev per-fold PR-AUC already stored in the repo for one target and configuration, with its source."""
    if target == WHAT_TARGET:  # WHAT values come from the adoption comparison
        column = "stage0_pr_auc" if config == "stage0" else "candidate_pr_auc"  # stage 0 or candidate column
        return pd.read_csv(ROOT / "results/trials/xgb_what_compare.csv")[column].tolist(), "results/trials/xgb_what_compare.csv"  # values and source
    if config == "stage0":  # WHEN stage 0 values come from the trial log
        row = {str(r["trial"]): r for r in read_log(ROOT / f"results/trials/xgb_{target}.jsonl")}["stage0"]  # logged stage0 trial
        return row["pr_auc"], f"results/trials/xgb_{target}.jsonl (trial stage0)"  # values and source
    return pd.read_csv(ROOT / f"results/trials/xgb_final_{target}.csv")["pr_auc"].tolist(), f"results/trials/xgb_final_{target}.csv"  # WHEN final values


def run_all(version: dict, target: str) -> dict:
    """Fit every configuration on every fold of one target. Returns config -> (rows of all folds, fit seconds per fold)."""
    folds = get_folds(version, target)  # five development folds
    out = {}  # config -> result
    for name, params in configs(target).items():  # stage 0 and final
        parts, times = [], []  # per fold rows and seconds
        for k in range(len(folds)):  # every fold
            rows, seconds = fit_fold(version, folds, target, params, k)  # fit and score one fold
            parts.append(rows)  # keep the scores
            times.append(seconds)  # keep the time
            print(f"  {target} {name} fold {k + 1}: {seconds:.1f} s", flush=True)  # progress
        out[name] = (pd.concat(parts, ignore_index=True), times)  # all folds of this configuration
    return out  # results of the target


STAGE = sys.argv[1] if len(sys.argv) == 2 else ""  # pilot or full
if STAGE not in ("pilot", "full"):  # only two stages exist
    sys.exit("usage: run_duplicate_sensitivity.py pilot|full")

# Step 2: collapse.
works = pd.read_parquet(ROOT / "data/processed/works_checkpoint.parquet")  # cleaned Works table
collapsed, info = collapse_works(works)  # definition A
print("== STEP 2: collapse ==")
print(pd.Series(info).to_string(), flush=True)

# Step 3: rebuild both versions, development rows only.
print("== STEP 3: rebuild ==")
start = time.time()  # clock of the rebuild
original = build_version(works)  # original works
collapsed_v = build_version(collapsed, low_volume_years=original["low"])  # collapsed works, same low-volume years
del works, collapsed  # free memory, only the development tables stay
print(f"rebuild seconds {time.time() - start:.0f}", flush=True)
for name, v in (("original", original), ("collapsed", collapsed_v)):  # sizes of the development tables
    print(f"{name}: WHEN dev rows {len(v['dev'])}, next_1y positives {int(v['dev']['next_1y'].sum())}, next_2y positives {int(v['dev']['next_2y'].sum())}, WHAT dev rows {len(v['table'])}, WHAT positives {int(v['table'][WHAT_TARGET].sum())}", flush=True)

# Reproduction of the stored dev per-fold PR-AUC on the original data.
print("== REPRODUCTION on the original data (4 decimals) ==")
frames, seconds = {}, {}  # (target, config) -> rows, fit seconds
bad = 0  # mismatching values
for target in TARGETS:  # one target at a time
    result = run_all(original, target)  # all fits of the target
    for config, (rows, times) in result.items():
        got = [point_metrics(g["y"].to_numpy(), g["score"].to_numpy())["pr_auc"] for _, g in rows.groupby("fold")]  # PR-AUC per fold
        want, source = stored_pr_auc(target, config)  # stored values
        ok = [round(a, 4) == round(b, 4) for a, b in zip(got, want)]  # 4 decimal comparison
        bad += len(ok) - sum(ok)  # count mismatches
        print(f"{target} {config} [{source}]\n  stored {[round(x, 4) for x in want]}\n  refit  {[round(x, 4) for x in got]}\n  match  {ok}", flush=True)
        frames[(target, config)], seconds[(target, config)] = rows, times  # keep for the projection
if bad:  # stop before the pilot
    sys.exit(f"{bad} value(s) do not match the stored ones, pilot not run")
print("all stored values reproduced", flush=True)

if STAGE == "full":  # full batch: collapsed fits, intervals, verdicts
    out_dir = ROOT / "results/duplicate_sensitivity"  # new result folder
    out_dir.mkdir(exist_ok=True)  # created by the shell redirect of the log if needed
    print("== STEP 5: full batch (collapsed fits) ==", flush=True)
    results = []  # one dict per target, configuration and group
    for target in TARGETS:  # one target at a time
        collapsed_res = run_all(collapsed_v, target)  # all collapsed fits of the target
        both = pd.concat([frames[(target, "stage0")]["artist_id"], collapsed_res["stage0"][0]["artist_id"]])  # artists of both versions
        universe = pd.Index(pd.unique(both.to_numpy()))  # shared bootstrap artists
        for config in ("stage0", "final"):  # both configurations
            results += analyze(frames[(target, config)], collapsed_res[config][0], target, config, universe)  # metrics and intervals
            print(f"analysed {target} {config}", flush=True)
    results = pd.DataFrame(results)  # all groups
    results.to_csv(out_dir / "results.csv", mode="x", index=False)  # exclusive creation
    print("== full tables, point metrics of each version and original 95% intervals ==")
    print(results[["target", "config", "group", "n_orig", "n_coll", "orig_pr_auc", "orig_pr_auc_lo", "orig_pr_auc_hi", "coll_pr_auc", "orig_roc_auc", "orig_roc_auc_lo", "orig_roc_auc_hi", "coll_roc_auc", "orig_baseline", "orig_baseline_lo", "orig_baseline_hi", "coll_baseline", "orig_gain", "orig_gain_lo", "orig_gain_hi", "coll_gain"]].round(4).to_string(index=False))
    print("== common rows, paired difference collapsed minus original ==")
    print(results[["target", "config", "group", "n_common", "orig_common_pr_auc", "coll_common_pr_auc", "diff_pr_auc", "diff_pr_auc_lo", "diff_pr_auc_hi", "orig_common_gain", "coll_common_gain", "diff_gain", "diff_gain_lo", "diff_gain_hi"]].round(4).to_string(index=False))
    print("== STEP 6: verdicts (group fold5 carries the known look-ahead; columns with no_fold5 exclude it) ==")
    verdict = verdicts(results)  # decision rule
    verdict.to_csv(out_dir / "verdicts.csv", mode="x", index=False)  # exclusive creation
    print(verdict.T.to_string())
    for _, v in verdict.iterrows():
        print(f"VERDICT {v['target']}: {'MATERIAL' if v['material'] else 'NOT MATERIAL'} (primary rule); secondary reading, paired CI excludes zero: {v['paired_ci_excludes_zero']}")
    sys.exit(0)

# Step 4: pilot, WHEN next_1y, final configuration, fold 1, both versions, identical seed.
print("== STEP 4: pilot ==")
target, config = "next_1y", "final"  # pilot target and configuration
params = configs(target)[config]  # final params
pilot = {}  # version -> (rows, seconds)
for name, v in (("original", original), ("collapsed", collapsed_v)):  # both data versions
    folds = get_folds(v, target)  # folds of this version
    pilot[name] = fit_fold(v, folds, target, params, 0)  # fold 1
    print(f"{name}: fold 1 train rows {len(folds[0]['train'])}, test rows {len(folds[0]['test'])}, fit {pilot[name][1]:.1f} s", flush=True)
same = np.allclose(pilot["original"][0]["score"].to_numpy(), frames[(target, config)].query("fold == 1")["score"].to_numpy())  # pilot original fit equals the reproduction fit
print(f"original pilot fit identical to the reproduction fit: {same}", flush=True)

# Time projection.
fit_orig = sum(sum(t) for t in seconds.values())  # seconds of all original fits
fit_total = fit_orig * (1 + pilot["collapsed"][1] / pilot["original"][1])  # original plus collapsed, scaled by the pilot ratio
rows = pilot["original"][0]  # fold 1 rows of the original
sets = {s: (rows["y"].to_numpy(), rows["score"].to_numpy(), pd.factorize(rows["artist_id"])[0]) for s in ("a", "b", "c")}  # three identical sets, as in a paired group
n_art = int(sets["a"][2].max()) + 1  # artists of the sample
start = time.time()  # clock of the bootstrap timing
bootstrap_sets(sets, n_art, n_bootstrap=20)  # 20 draws, three sets
unit = (time.time() - start) / (20 * 3 * len(rows))  # seconds per row, draw and set
boot_rows = sum(len(frames[(t, c)]) for t in TARGETS for c in ("stage0", "final"))  # pooled rows of all target and configuration pairs
boot_total = unit * boot_rows * 2.8 * 3 * 1000  # fold groups plus pooled plus pooled without fold 5, three sets, 1000 draws
print(f"all original fits {fit_orig / 60:.1f} min; projected fits of both versions {fit_total / 60:.1f} min")
print(f"bootstrap unit {unit * 1e9:.1f} ns per row, draw and set; projected bootstrap {boot_total / 60:.1f} min")
print(f"projected total {(fit_total + boot_total) / 60:.1f} min (rebuild not included)", flush=True)
