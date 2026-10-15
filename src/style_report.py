"""
Reporting helpers of notebooks/style_forecast.ipynb: tables and charts of the style forecast.

Everything here reads saved files or the style by year table, nothing is fitted.
The test values are recomputed from the saved predictions and the series, and
the printed console report of the test run is parsed so both can be compared.

Usage:
    table = style_table(kept, 2023)
    projection_table(predictions, table, 2023)
"""

from __future__ import annotations

import io  # text buffer for the parsed console tables
import re  # patterns of the printed count and difference tables
from pathlib import Path  # file paths

import matplotlib.pyplot as plt  # charts
import numpy as np  # arrays
import pandas as pd  # tables

from style_forecast import BASELINES, DEV_FIRST_TARGET, MIN_BASE, MODELS, TRANCHES, boot_ci, count_metrics, hits_table, paired_differences, spearman_table  # forecast constants and functions
from tuning import read_log  # trial log reader

METHODS = ["random"] + BASELINES + MODELS  # every method in report order, development years
TEST_METHODS = ["random", "repeat_last", "mean_growth", "random_forest"]  # methods reported on the test years: locked forest and the baselines
TEST_SCORED = TEST_METHODS[1:]  # the same without random, for the Spearman
TEST_SETS = ["all eligible", "top 10 of random_forest"]  # style sets of the count metrics kept on the test years
TEST_COUNT_METHODS = ["random_forest", "repeat_ct", "mean3"]  # count methods kept on the test years
PHASE_COLORS = {"random": "#2a78d6", "grid": "#eb6834", "manual": "#1baf7a"}  # one color per tuning phase
INK, GRID = "#0b0b0b", "#e4e3df"  # text and grid colors


def base_check(table: pd.DataFrame, styles: list, base_year: int) -> pd.DataFrame:
    """Works of the base year, mean of the three previous years (base of the growth) and mean of the base year and the two before it (base of the expected Works)."""
    rows = table.loc[styles]  # style by year counts of the listed styles
    return pd.DataFrame({
        "works": rows[base_year],  # Works of the base year
        "growth_base": rows[[base_year - 3, base_year - 2, base_year - 1]].mean(axis=1),  # base of the growth of the base year
        "expected_base": rows[[base_year - 2, base_year - 1, base_year]].mean(axis=1),  # base of the expected Works of the next year
    })


def projection_table(pred: pd.DataFrame, table: pd.DataFrame, base_year: int) -> pd.DataFrame:
    """Top 10 of the projection by repeat last year with both bases, the small-denominator flag and the three models' expected Works."""
    top = pred[(pred["method"] == "repeat_last") & (pred["target_year"] == base_year + 1) & (pred["rank"] <= 10)].sort_values("rank")  # recommended ranking, top 10
    out = pd.DataFrame({"style": top["style"].to_numpy(), "growth": top["expected_growth"].to_numpy()})  # ranking columns
    bases = base_check(table, list(out["style"]), base_year).reset_index(drop=True)  # bases computed from the series
    out = pd.concat([out, bases], axis=1)  # growth next to its bases
    out["flag"] = out["growth_base"] < MIN_BASE  # growth base below the eligibility level
    out["growth_check"] = np.isclose(out["works"] / (1 + out["growth"]), out["growth_base"])  # Works over one plus growth equals the series base
    for m in MODELS:  # expected Works and range of each model
        part = pred[(pred["method"] == m) & (pred["target_year"] == base_year + 1)].set_index("style").loc[out["style"]]  # same styles in the same order
        out[m] = [f"{e:.0f} ({lo:.0f}-{hi:.0f})" for e, lo, hi in zip(part["expected_works"], part["range_low"], part["range_high"])]  # expected and range
    return out


def hits_summary(results: pd.DataFrame, k: int, methods: list = METHODS) -> tuple:
    """Top_k_styles per year and the mean with its 95% interval (bootstrap over years, 2,000 draws, seed 1) for the listed methods."""
    per_year = hits_table(results, k)  # hits per target year
    ci = pd.DataFrame({m: boot_ci(per_year[m].to_numpy(dtype=float)) for m in methods}, index=["mean", "low", "high"]).T  # mean and interval
    return per_year[["eligible"] + methods], ci  # both tables


def dev_hits_summary(path: Path, k: int) -> pd.DataFrame:
    """Five tranche means, pooled mean and 95% interval per method from the saved development hits."""
    dev = pd.read_csv(path, index_col="target_year")  # per-year hits of the development run
    rows = {}  # method -> values
    for m in METHODS:  # every method
        values = dev[f"hits{k}_{m}"].astype(float)  # hits per year
        row = {f"{a}-{b}": values.loc[a:b].mean() for a, b in TRANCHES}  # five tranche means
        row["pooled"], row["low"], row["high"] = boot_ci(values.to_numpy())  # pooled mean and interval
        rows[m] = row  # keep
    return pd.DataFrame(rows).T  # one row per method


def conditions_table(conditions: dict) -> pd.DataFrame:
    """Rule conditions as a table: name, True or False and the numbers behind it."""
    def show(value):  # round numbers, keep tuples
        """Round a number, or every number of a tuple, to 4 decimals for printing."""
        return tuple(round(float(v), 4) for v in value) if isinstance(value, tuple) else round(float(value), 4)  # printable numbers
    return pd.DataFrame([{"condition": re.sub(r"hits.10", "top_10_styles", name), "holds": bool(ok), "numbers": show(value)} for name, (ok, value) in conditions.items()])  # one row per condition, the rule text shows the metric under its report name


def rf_trials_frame(log_path: Path) -> pd.DataFrame:
    """One row per logged Random Forest trial with its settings and summary numbers (stage 0 has its own phase)."""
    rows = read_log(log_path)  # all finished trials
    return pd.DataFrame([{
        "trial": r["trial"], "phase": "stage0" if r["trial"] == "stage0" else r["phase"], **{k: "unlimited" if v is None else v for k, v in r["params"].items()},  # identifiers and settings, no depth limit shown as unlimited
        "mean_spearman": r["mean_spearman"], "mean_hits10": r["mean_hits10"], "seconds": r["seconds"],  # summary numbers
    } for r in rows])


def trial_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per phase: trials, best, median and worst mean Spearman, trials above stage 0 and the best trial."""
    stage0 = frame.loc[frame["phase"] == "stage0", "mean_spearman"].iloc[0]  # reference value
    tuned = frame[frame["phase"] != "stage0"]  # tuned trials only
    grouped = tuned.groupby("phase")["mean_spearman"]  # by phase
    out = pd.DataFrame({
        "trials": grouped.size(), "best": grouped.max(), "median": grouped.median(), "worst": grouped.min(),  # spread of the phase
        "above stage 0": grouped.apply(lambda s: int((s > stage0).sum())),  # trials above the reference
        "best trial": tuned.loc[grouped.idxmax(), ["phase", "trial"]].set_index("phase")["trial"],  # id of the best trial
    })
    return out.reindex(["random", "grid", "manual"])  # phase order


def pool_top(frame: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Best tuned trials of all phases together by mean Spearman."""
    cols = ["trial", "phase", "n_estimators", "max_depth", "min_samples_leaf", "max_features", "max_samples", "mean_spearman", "mean_hits10"]  # shown columns
    top = frame[frame["phase"] != "stage0"].sort_values("mean_spearman", ascending=False)[cols].head(n).reset_index(drop=True)  # top n
    return top.rename(columns={"mean_hits10": "mean_top_10_styles"})  # displayed copy under the report name


def _clean(ax) -> None:
    """Light grid, no top and right frame."""
    ax.spines[["top", "right"]].set_visible(False)  # recessive frame
    ax.grid(axis="y", color=GRID, linewidth=0.8)  # light horizontal grid
    ax.set_axisbelow(True)  # grid behind the marks


def plot_trials(frame: pd.DataFrame) -> None:
    """Chart 1: mean Spearman of every tuned trial in log order, colored by phase, with the stage 0 line."""
    stage0 = frame.loc[frame["phase"] == "stage0", "mean_spearman"].iloc[0]  # reference value
    fig, ax = plt.subplots(figsize=(10, 4.5))  # one panel
    for phase, color in PHASE_COLORS.items():  # one color per phase
        part = frame[frame["phase"] == phase]  # trials of the phase
        ax.scatter(part.index, part["mean_spearman"], s=36, color=color, edgecolor="white", linewidth=0.8, label=f"{phase} ({len(part)})")  # marks with a surface ring
    ax.axhline(stage0, color=INK, linestyle="--", linewidth=1.2)  # stage 0 line
    ax.text(frame.index.max(), stage0 + 0.0008, f"stage 0 {stage0:.4f}", ha="right", color=INK, fontsize=10)  # direct label
    best = frame[frame["phase"] != "stage0"]["mean_spearman"].idxmax()  # best tuned trial
    ax.annotate(f"best: {frame.loc[best, 'trial']} {frame.loc[best, 'mean_spearman']:.4f}", (best, frame.loc[best, "mean_spearman"]), xytext=(6, 8), textcoords="offset points", fontsize=10)  # direct label
    ax.set_xlabel("trial, in log order")  # x label
    ax.set_ylabel("mean Spearman over 25 years")  # y label
    ax.set_title("Random Forest tuning trials against stage 0")  # title
    ax.legend(frameon=False, loc="lower right")  # legend
    _clean(ax)  # light style
    plt.show()  # draw


def plot_hits(dev: pd.DataFrame, test: pd.DataFrame) -> None:
    """Chart 2: mean top_10_styles per method on the development years (all methods) and on the test years (locked forest and baselines), with 95% intervals."""
    fig, ax = plt.subplots(figsize=(10, 4.5))  # one panel
    for shift, table, color, label in [(-0.2, dev, "#2a78d6", "development 1993-2017"), (0.2, test, "#eb6834", "test 2018-2022")]:  # two series
        shown = [m for m in METHODS if m in table.index]  # methods present in this series
        x = np.array([METHODS.index(m) for m in shown])  # slot of each method
        mean = table.loc[shown, "mean"].to_numpy()  # bar heights
        err = [mean - table.loc[shown, "low"].to_numpy(), table.loc[shown, "high"].to_numpy() - mean]  # interval arms
        ax.bar(x + shift, mean, width=0.36, color=color, label=label, yerr=err, error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": INK})  # bars with intervals
    ax.set_xticks(np.arange(len(METHODS)))  # method ticks
    ax.set_xticklabels(METHODS)  # method names
    ax.set_ylim(0, max(dev["high"].max(), test["high"].max()) * 1.2)  # room for the legend above the bars
    ax.set_ylabel("mean top_10_styles")  # y label
    ax.set_title("top_10_styles by method, development and test years (95% interval over years)")  # title
    ax.legend(frameon=False)  # legend
    _clean(ax)  # light style
    plt.show()  # draw


def plot_flagged_works(table: pd.DataFrame, info: pd.DataFrame, base_year: int, first_year: int = 2015) -> None:
    """Chart 3: Works per year of the top 10 styles flagged for a small growth base, one panel per style, with the growth base line."""
    flagged = info[info["flag"]]  # styles with a growth base below the eligibility level
    years = list(range(first_year, base_year + 1))  # plotted years
    fig, axes = plt.subplots(1, len(flagged), figsize=(4 * len(flagged), 3.6), sharey=False, squeeze=False)  # one panel per flagged style
    for ax, (_, row) in zip(axes[0], flagged.iterrows()):  # one panel each
        ax.plot(years, table.loc[row["style"], years].to_numpy(), color="#2a78d6", linewidth=2, marker="o", markersize=5)  # Works per year
        ax.axhline(row["growth_base"], color=INK, linestyle=":", linewidth=1.2)  # mean of the three years before the base year
        ax.text(first_year, row["growth_base"], f" growth base {row['growth_base']:.0f}", va="bottom", fontsize=9, color=INK)  # direct label
        ax.set_title(f"{row['style']}, growth {row['growth']:.2f}")  # style and growth
        ax.set_xticks(years[::2])  # every second year
        _clean(ax)  # light style
    axes[0][0].set_ylabel("Works per year")  # y label on the first panel
    fig.suptitle(f"Works {first_year} to {base_year} of the top 10 styles with a growth base below {MIN_BASE}")  # figure title
    fig.tight_layout()  # no overlaps
    plt.show()  # draw


def series_rows(table: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """Eligible styles per target year from the series: counts, base, growth terms and label, in the order of the panel."""
    counts = table.to_numpy().astype(float)  # matrix styles by years
    years = np.array(table.columns)  # calendar years of the columns
    frames = []  # one frame per target year
    for target in range(first, last + 1):  # one target year at a time
        j = int(np.where(years == target - 1)[0][0])  # column of the origin year
        c = [counts[:, j - i] for i in range(6)]  # c[0] is the origin year, c[5] five years before
        base = (c[0] + c[1] + c[2]) / 3  # mean of the three latest years
        keep = base >= MIN_BASE  # eligible styles
        gr = [c[i] / np.where((c[i + 1] + c[i + 2] + c[i + 3]) > 0, (c[i + 1] + c[i + 2] + c[i + 3]) / 3, np.nan) - 1 for i in range(3)]  # smoothed growth at t, t-1, t-2
        frames.append(pd.DataFrame({
            "target": target, "style": table.index[keep], "c_t": c[0][keep], "base": base[keep], "c_next": counts[keep, j + 1],  # identifiers and counts
            "gr0": gr[0][keep], "gr1": gr[1][keep], "gr2": gr[2][keep],  # growth terms of the baselines
        }))
    out = pd.concat(frames, ignore_index=True)  # all years
    out["label"] = np.log(np.maximum(out["c_next"], 1) / out["base"])  # log growth of the target year over the base
    return out


def test_frame(pred: pd.DataFrame, table: pd.DataFrame, first: int, last: int) -> tuple:
    """Results-like frame of the test years from saved predictions and the series, with count columns, plus consistency checks."""
    rows = series_rows(table, DEV_FIRST_TARGET, last)  # eligible styles of every scored year
    rows["mean3_exp"] = rows["base"]  # count baseline: mean of the three latest years
    rows["mean3_lo"], rows["mean3_hi"] = np.nan, np.nan  # ranges filled below
    for year in range(first, last + 1):  # ranges use the residuals of all earlier folds
        eps = rows.loc[rows["target"] < year, "label"]  # log residuals of earlier folds
        sel = rows["target"] == year  # rows of this year
        rows.loc[sel, "mean3_lo"], rows.loc[sel, "mean3_hi"] = rows.loc[sel, "base"] * np.exp(eps.quantile(0.1)), rows.loc[sel, "base"] * np.exp(eps.quantile(0.9))  # range
    test = rows[rows["target"] >= first].reset_index(drop=True)  # scored years only
    test["mean_growth"] = test[["gr0", "gr1", "gr2"]].mean(axis=1)  # mean growth of t, t-1, t-2
    key = pd.MultiIndex.from_frame(test[["style", "target"]])  # (style, target year) of every row
    for m in MODELS + ["repeat_last"]:  # saved methods
        part = pred[pred["method"] == m].rename(columns={"target_year": "target"}).set_index(["style", "target"])  # saved rows of the method
        name = "repeat_ct" if m == "repeat_last" else m  # count columns that go with the method
        test[m] = part["expected_growth"].reindex(key).to_numpy()  # ranking score
        test[name + "_exp"] = part["expected_works"].reindex(key).to_numpy()  # expected Works
        test[name + "_lo"] = part["range_low"].reindex(key).to_numpy()  # range low
        test[name + "_hi"] = part["range_high"].reindex(key).to_numpy()  # range high
    actual = test["c_next"].clip(lower=1)  # actual count, at least 1 as in the label
    for name in MODELS + ["repeat_ct", "mean3"]:  # every count method
        test[name + "_ape"] = (test[name + "_exp"] - actual).abs() / actual  # absolute percentage error
        test[name + "_in"] = (actual >= test[name + "_lo"]) & (actual <= test[name + "_hi"])  # inside the range
    saved = pred[(pred["target_year"] >= first) & (pred["target_year"] <= last) & (pred["method"] == "repeat_last")].groupby("target_year").size()  # saved rows per year
    checks = {
        "rows per year equal to the saved predictions": bool((test.groupby("target").size().to_numpy() == saved.to_numpy()).all()),  # same eligible styles
        "no missing saved value": not test[MODELS + ["repeat_last"]].isna().any().any(),  # every row matched
        "saved repeat last score equals the series growth": bool(np.allclose(test["repeat_last"], test["gr0"])),  # same ranking score
        "saved expected Works equal base x exp(prediction)": all(np.allclose(test["base"] * (1 + test[m]), test[m + "_exp"]) for m in MODELS),  # same base as the series
    }
    return test, checks


def read_printed_test(path: Path) -> dict:
    """Parse the tables of the test console report into frames."""
    lines = Path(path).read_text().splitlines()  # report lines

    def table_after(prefix: str, start: int = 0) -> pd.DataFrame:
        """Table that follows the first line from `start` that matches the pattern `prefix` at its start."""
        i = next(n for n in range(start, len(lines)) if re.match(prefix, lines[n]))  # header line of the block
        body = []  # table lines
        for line in lines[i + 1:]:  # lines after the header
            if not line.strip() or line.startswith(("--", "==", "mean over")):  # end of the block
                break
            body.append(line)  # keep
        return pd.read_csv(io.StringIO("\n".join(body)), sep=r"\s+")  # first column becomes the index

    def at(prefix: str) -> int:
        """Line number of the first line that begins with `prefix`."""
        return next(n for n in range(len(lines)) if lines[n].startswith(prefix))  # position

    out = {}  # name -> frame
    for k in (10, 5):  # both cut-offs
        out[f"hits{k}"] = table_after(rf"== hits.{k} per target year")  # per year, the printed report names the metric with a separator before the cut-off
        out[f"hits{k}_mean"] = table_after(rf"-- hits.{k} mean over")  # mean and interval
    out["spearman"] = table_after("== Spearman on the styles scored by every method")  # per year
    i = at("mean over the years:")  # mean block
    out["spearman_mean"] = pd.read_csv(io.StringIO("\n".join(lines[i + 1:i + 6])), sep=r"\s+", header=None, index_col=0)[1]  # five means
    pattern = re.compile(r"^\s*(.+?)\s+(bayes_ridge|xgboost|random_forest|repeat_ct|mean3)\s+(\d+)\s+([\d.]+)\s+([\d.]+)$")  # one count row
    rows = [pattern.match(line).groups() for line in lines[at("== count metrics"):] if pattern.match(line)]  # all count rows
    out["counts"] = pd.DataFrame(rows, columns=["set", "method", "n", "median_ape", "coverage"]).astype({"n": int, "median_ape": float, "coverage": float})  # numeric columns
    for name, header in [("paired_dev", "-- development years"), ("paired_test", "-- test years")]:  # both difference tables
        diffs = table_after(header, at("== descriptive paired differences"))  # model and metric come as two index levels
        diffs.index = [" ".join(pair) for pair in diffs.index]  # one label per row, as in the recomputed table
        out[name] = diffs  # keep
    return out


def compare(recomputed, printed, decimals: int) -> tuple:
    """Largest absolute difference between a recomputed table and the printed one, and whether it is within the printed rounding."""
    a = np.asarray(recomputed, dtype=float)  # recomputed values
    b = np.asarray(printed, dtype=float)  # printed values
    diff = float(np.abs(a - b).max())  # largest difference
    return diff, diff <= 0.5 * 10.0 ** -decimals + 1e-9  # within half a unit of the last printed digit


def series_facts(kept: pd.DataFrame, dropped: dict, table: pd.DataFrame) -> pd.Series:
    """Size of the series: Works kept and dropped, styles, first and last year, Works counted in the style by year table."""
    facts = {"Works dated and tagged": len(kept), **{f"dropped, {reason}": n for reason, n in dropped.items()}}  # Works kept and dropped by reason
    facts.update({"styles in the table": table.shape[0], "first year": int(table.columns.min()), "last year": int(table.columns.max()), "style-year counts (a Work counts in each of its styles)": int(table.to_numpy().sum())})  # shape of the table
    return pd.Series(facts)  # one fact per row


def matches_saved_hits(results: pd.DataFrame, path: Path, methods: list) -> bool:
    """True when the top_10_styles of `methods` equal the saved development values in every year."""
    saved = pd.read_csv(path, index_col="target_year")  # per-year hits of the development run
    fresh = hits_table(results, 10)  # hits recomputed here
    return all((fresh[m].to_numpy(dtype=float) == saved[f"hits10_{m}"].to_numpy(dtype=float)).all() for m in methods)  # every cell equal


def matches_log_row(results: pd.DataFrame, log_path: Path, trial: str) -> bool:
    """True when the Random Forest top_10_styles and top_5_styles of `results` equal those of a trial row of the log in every year."""
    row = next(r for r in read_log(log_path) if r["trial"] == trial)  # logged trial
    return all((hits_table(results, k)["random_forest"].to_numpy(dtype=float) == np.array(row[f"hits{k}"], dtype=float)).all() for k in (10, 5))  # both cut-offs


def manual_table(log_path: Path) -> pd.DataFrame:
    """Manual trials with the setting each one changes, mean Spearman, difference against the base, top_10_styles and seconds."""
    rows = read_log(log_path)  # all trials
    tuned = [r for r in rows if r["phase"] in ("random", "grid") and r["trial"] != "stage0"]  # trials the base is chosen from
    base = max(tuned, key=lambda r: r["mean_spearman"])  # base of the manual phase
    out = [{
        "trial": r["trial"],  # manual trial id
        "changed": ", ".join(f"{k}={v}" for k, v in r["params"].items() if base["params"].get(k, "unset") != v),  # settings that differ from the base
        "mean_spearman": r["mean_spearman"], "difference": r["mean_spearman"] - base["mean_spearman"],  # score and gap to the base
        "mean_top_10_styles": r["mean_hits10"], "seconds": r["seconds"],  # descriptive numbers, shown under the report name
    } for r in rows if r["phase"] == "manual"]
    return pd.DataFrame(out)  # one row per manual trial


def side_by_side(recomputed: pd.DataFrame, printed: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Recomputed values (rounded as printed) next to the printed ones."""
    return pd.concat({"recomputed": recomputed.round(decimals), "printed": printed}, axis=1)  # two blocks of columns


def keep_test_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Count-metric rows kept on the test years: all eligible styles and the forest's own top 10, for the forest and the two count baselines."""
    return frame[frame["set"].isin(TEST_SETS) & frame["method"].isin(TEST_COUNT_METHODS)].reset_index(drop=True)  # same rows from recomputed and printed tables


def forest_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of a paired-difference table that belong to the Random Forest, with the metric shown under its report name."""
    rows = frame[frame.index.str.startswith("random_forest")]  # labels start with the model name
    return rows.rename(index=lambda label: re.sub(r"hits.10", "top_10_styles", label))  # displayed copy, the stored names stay as generated


def compare_all(test: pd.DataFrame, dev: pd.DataFrame, printed: dict) -> pd.DataFrame:
    """Printed test quantities of the reported methods against their recomputed values: largest difference and whether it is within the printed rounding."""
    rows = []  # one row per quantity

    def add(name, recomputed, printed_values, decimals):
        """Compare one recomputed quantity with its printed values and append a row to `rows`. Inputs: quantity name, recomputed values, printed values and decimals. Output: none."""
        diff, ok = compare(recomputed, printed_values, decimals)  # largest difference and rounding check
        rows.append({"quantity": name, "max abs difference": diff, "within printed rounding": ok})  # keep

    for k in (10, 5):  # both cut-offs
        per_year, ci = hits_summary(test, k, TEST_METHODS)  # recomputed tables of the reported methods
        add(f"top_{k}_styles per year", per_year[TEST_METHODS].to_numpy(), printed[f"hits{k}"][TEST_METHODS].to_numpy(), 2)  # per year
        add(f"top_{k}_styles mean and interval", ci.to_numpy(), printed[f"hits{k}_mean"].loc[TEST_METHODS].to_numpy(), 2)  # mean and interval
    spearman = spearman_table(test)  # Spearman per year on the common styles
    add("Spearman per year", spearman[TEST_SCORED].to_numpy(), printed["spearman"][TEST_SCORED].to_numpy(), 3)  # per year
    add("Spearman mean", spearman[TEST_SCORED].mean().to_numpy(), printed["spearman_mean"].loc[TEST_SCORED].to_numpy(), 3)  # mean
    metrics, shown = keep_test_counts(count_metrics(test, test)), keep_test_counts(printed["counts"])  # kept count rows, recomputed and printed
    same_rows = bool((metrics["set"].to_numpy() == shown["set"].to_numpy()).all() and (metrics["method"].to_numpy() == shown["method"].to_numpy()).all())  # same sets and methods in the same order
    add("count metrics (median APE and coverage)", metrics[["median_ape", "coverage"]].to_numpy(), shown[["median_ape", "coverage"]].to_numpy(), 3)  # values
    add("paired differences, test years", forest_rows(paired_differences(test)).to_numpy(), forest_rows(printed["paired_test"]).to_numpy(), 3)  # test differences
    add("paired differences, development years", forest_rows(paired_differences(dev)).to_numpy(), forest_rows(printed["paired_dev"]).to_numpy(), 3)  # development differences
    out = pd.DataFrame(rows)  # all rows
    out.loc[len(out)] = ["count rows in the same order", 0.0, same_rows]  # order check of the count table
    return out
