"""Style forecast duplicate audit, step B: development-only refit on the original and the collapsed series."""

import numpy as np  # arrays
import pandas as pd  # tables

from style_duplicate_audit import WORKS_COLUMNS as AUDIT_COLUMNS, count_comparison, kept_versions  # step A pieces, reused without changes
from style_forecast import BASELINES, DEV_LAST_TARGET, LOCKED_RF_PARAMS, MODELS, boot_ci, build_panel, hits_table, pass_rule, spearman_table, walk_forward, walk_forward_tuned  # forecast pieces
from style_series import WORKS_PATH  # location of the cleaned Works table

WORKS_COLUMNS = AUDIT_COLUMNS + ["genres"]  # the panel also needs the genres of each Work
METHODS = ["random"] + BASELINES + MODELS  # every method in report order
SETTINGS = {"default": None, "locked": LOCKED_RF_PARAMS}  # default forest settings, and the locked forest (None means walk_forward)
K = 10  # ranking size of the decision metric
DECIMALS = 4  # decimals of the control comparison


def dev_works() -> pd.DataFrame:
    """Read the cleaned Works table with the audit columns and only the rows dated up to the last development year.

    The filter is applied while reading, so no later row is kept. A Work dated up to that year is never
    removed by a later one (the earliest year wins), so the collapse of these rows equals the full collapse.
    """
    return pd.read_parquet(WORKS_PATH, columns=WORKS_COLUMNS, filters=[("year", "<=", DEV_LAST_TARGET)])  # dated rows up to the last development year


def check_counts_against_step_a(kept_original: pd.DataFrame, kept_collapsed: pd.DataFrame, path) -> tuple:
    """Compare the counts of the development rows with the saved step A counts for the same years.

    Returns (rows compared, rows that differ). The saved table holds counts built from all works, so a
    difference would mean that the early filter changed the collapse.
    """
    new = count_comparison(kept_original, kept_collapsed)  # counts of the filtered works
    saved = pd.read_csv(path)  # step A counts, all years
    saved = saved[saved["year"] <= DEV_LAST_TARGET].drop(columns="difference")  # same years, same columns
    merged = new.drop(columns="difference").merge(saved, on=["style", "year"], how="outer", suffixes=("", "_saved"), indicator=True)  # row by row
    differ = (merged["_merge"] != "both") | (merged["original"] != merged["original_saved"]) | (merged["collapsed"] != merged["collapsed_saved"])  # a missing or unequal row
    return len(merged), int(differ.sum())  # compared and different


def run_version(kept: pd.DataFrame) -> dict:
    """Panel and walk-forward results of one data version for every setting, as {setting: (hits@10, Spearman)}."""
    panel = build_panel(kept, DEV_LAST_TARGET)  # style by origin rows, years up to the last development year
    out = {}  # setting -> tables
    for name, params in SETTINGS.items():  # default forest, then locked forest
        results = walk_forward(panel) if params is None else walk_forward_tuned(panel, params)  # one fit per model and target year
        out[name] = (hits_table(results, K).astype(float), spearman_table(results))  # decision tables
    return out  # all settings of this version


def control_against_dev_hits(hits: pd.DataFrame, path) -> pd.DataFrame:
    """Per-year differences between recomputed hits@10 (plus eligible styles) and the saved dev_hits.csv, at 4 decimals."""
    saved = pd.read_csv(path, index_col="target_year")  # saved per-year hits
    rows = {}  # column -> largest absolute difference and count of unequal years
    for column, new in [("eligible", hits["eligible"])] + [(f"hits10_{m}", hits[m]) for m in METHODS]:  # eligible styles and every method
        old = saved[column]  # saved values
        rows[column] = {"max_abs_diff": float((new.round(DECIMALS) - old.round(DECIMALS)).abs().max()), "years_different": int((new.round(DECIMALS) != old.round(DECIMALS)).sum())}  # comparison at 4 decimals
    return pd.DataFrame(rows).T  # one row per column


def per_year_table(setting: str, version: str, hits: pd.DataFrame, spearman: pd.DataFrame) -> pd.DataFrame:
    """Long table of per-year hits@10, Spearman and eligible styles for one setting and version."""
    rows = []  # one dict per method and year
    for method in METHODS:  # every method
        for year in hits.index:  # every development target year
            rows.append({"setting": setting, "version": version, "method": method, "target_year": int(year), "hits10": hits.loc[year, method], "spearman": spearman.loc[year, method] if method in spearman.columns else np.nan, "eligible": hits.loc[year, "eligible"]})  # random has no Spearman
    return pd.DataFrame(rows)  # long table


def status_table(setting: str, rule_original: dict, rule_collapsed: dict, hits_original: pd.DataFrame, hits_collapsed: pd.DataFrame) -> pd.DataFrame:
    """Status of the five conditions, the candidate and baseline, and the ranking of methods, on both versions."""
    rows = []  # one dict per item
    for name, (ok, value) in rule_original["conditions"].items():  # each condition
        ok_c, value_c = rule_collapsed["conditions"][name]  # same condition on the collapsed version
        rows.append({"section": "condition", "setting": setting, "key": name, "original": bool(ok), "collapsed": bool(ok_c), "changed": bool(ok) != bool(ok_c), "original_value": str(value), "collapsed_value": str(value_c)})  # status and numbers
    for key in ("candidate", "baseline", "recommended"):  # items that depend on the ranking of models
        rows.append({"section": "rule", "setting": setting, "key": key, "original": rule_original[key], "collapsed": rule_collapsed[key], "changed": rule_original[key] != rule_collapsed[key]})  # names
    order = lambda h: ",".join(h[METHODS].mean().sort_values(ascending=False, kind="stable").index)  # methods by mean dev hits@10, best first
    rows.append({"section": "ranking", "setting": setting, "key": "methods by mean dev hits@10", "original": order(hits_original), "collapsed": order(hits_collapsed), "changed": order(hits_original) != order(hits_collapsed)})  # ranking of methods
    return pd.DataFrame(rows)  # status table


def method_table(setting: str, hits_original: pd.DataFrame, spearman_original: pd.DataFrame, hits_collapsed: pd.DataFrame, spearman_collapsed: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """One row per method: mean hits@10, original 95% interval, paired difference with interval, Spearman and both verdict readings."""
    setting_changed = bool(status["changed"].any())  # any condition, candidate, baseline, recommendation or ranking changed
    rows = []  # one dict per method
    for method in METHODS:  # every method
        original, collapsed = hits_original[method].to_numpy(float), hits_collapsed[method].to_numpy(float)  # per-year hits
        mean, low, high = boot_ci(original)  # mean and 95% year-bootstrap interval on the original
        diff, diff_low, diff_high = boot_ci(collapsed - original)  # paired difference, same years resampled in both versions
        outside = bool(collapsed.mean() < low or collapsed.mean() > high)  # collapsed mean outside the original interval
        rows.append({
            "section": "method", "setting": setting, "key": method,  # identifiers
            "original_mean": mean, "original_ci_low": low, "original_ci_high": high, "collapsed_mean": collapsed.mean(),  # means and interval
            "diff_mean": diff, "diff_ci_low": diff_low, "diff_ci_high": diff_high,  # paired difference collapsed minus original
            "spearman_original": spearman_original[method].mean() if method in spearman_original.columns else np.nan,  # mean Spearman, none for random
            "spearman_collapsed": spearman_collapsed[method].mean() if method in spearman_collapsed.columns else np.nan,  # same on the collapsed version
            "primary_outside_ci": outside, "setting_status_changed": setting_changed,  # parts of the primary reading
            "primary_material": outside or setting_changed,  # primary reading of Entry 2
            "secondary_ci_excludes_zero": bool(diff_low > 0 or diff_high < 0),  # secondary reading of Entry 2
        })
    return pd.DataFrame(rows)  # method table
