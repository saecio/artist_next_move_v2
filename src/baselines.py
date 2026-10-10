"""
Historical baselines and the kill gate, for both WHEN and WHAT.

docs/decisions.md is explicit that "historical baselines come before ML":
no Logistic Regression / Random Forest / XGBoost result means anything
until it is shown to beat a simple, honest baseline built only from an
artist's own past behaviour. This module builds those baselines, and
answers the other PENDING item in decisions.md ("whether WHAT survives
formal validation") with an actual number instead of a feeling: a kill
gate that only passes when the ML model's PR-AUC confidence interval is
clearly above the baseline's, not just numerically higher by chance.

Everything here takes a snapshots table shaped like the output of
`src.snapshots.build_snapshots` (one row per snapshot, with at least
`artist_id`, `history_count`, `historical_rate`, and whichever target
column is being evaluated). It does not read files and does not know
about Discogs, same as src/snapshots.py.

Usage:
    from src.baselines import (
        fit_when_baseline, score_when_baseline,
        fit_what_baseline, score_what_baseline,
        evaluate_kill_gate,
    )

    # fit_* only ever looks at the TRAIN split, never test, so the
    # baseline itself cannot leak future information either.
    when_baseline = fit_when_baseline(train_snapshots)
    train_scores = score_when_baseline(train_snapshots, when_baseline)
    test_scores = score_when_baseline(test_snapshots, when_baseline)

    gate = evaluate_kill_gate(
        y_true=test_snapshots["next_1y"],
        model_scores=model_test_scores,
        baseline_scores=test_scores,
        artist_ids=test_snapshots["artist_id"],
    )
    if not gate["model_beats_baseline"]:
        # same decision docs/decisions.md's kill gate describes: drop WHAT
        # (or whichever target this gate was run for) and do not build an
        # app page for it.
        ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Confidence level used to decide "clearly above", not just "higher".
# 90%, not 95%, is a deliberate choice: with artist-level bootstrap on a
# dataset this size, 95% CIs are wide enough that almost nothing would
# ever clear the gate. 90% is still a real, stated bar, not a rubber stamp.
KILL_GATE_CONFIDENCE = 0.90

# Number of bootstrap resamples for the kill gate's PR-AUC confidence
# intervals. 1,000 is the usual default for a bootstrap CI; this is not
# tuned, just a standard, round number.
N_BOOTSTRAP = 1_000


# --------------------------------------------------------------------------
# WHEN baseline: "this artist's own historical release rate, extrapolated."
# --------------------------------------------------------------------------
#
# historical_rate (works per year active so far) is already computed per
# snapshot by src.snapshots.build_snapshots. An artist who has historically
# put out roughly one Work a year is more likely to have a next Work soon
# than one who has put out one every five years; that ordering is exactly
# what a baseline needs to get right for PR-AUC (PR-AUC only cares about
# ranking, not calibration, so historical_rate is used directly as a score,
# no probability conversion).
#
# history_count == 0 (an artist's very first known Work) has no
# historical_rate at all (see src/snapshots.py's docstring: dividing by
# zero years of activity is left unknown, not forced to zero). Those rows
# fall back to the TRAIN set's median historical_rate, so the baseline
# still produces a score for every row without inventing a rate from
# nothing and without peeking at test data to decide what that fallback
# should be.


def fit_when_baseline(train_snapshots: pd.DataFrame) -> dict:
    """Learn the one number this baseline needs from TRAIN only: the
    fallback score for an artist with no history yet (history_count == 0).

    Returns a plain dict, e.g. `{"fallback_rate": 0.4}`, so it can be
    saved/logged like any other fitted artifact, not a surprise function
    closure.
    """
    with_history = train_snapshots.loc[train_snapshots["history_count"] > 0, "historical_rate"]
    # Median, not mean: historical_rate is a ratio of small integers for
    # most artists and can have a long right tail (a handful of very
    # prolific artists), the median is a more honest "typical" fallback.
    fallback_rate = float(with_history.median()) if len(with_history) else 0.0
    return {"fallback_rate": fallback_rate}


def score_when_baseline(snapshots: pd.DataFrame, fitted: dict) -> pd.Series:
    """Score every row for the WHEN baseline: historical_rate, or the
    fitted fallback for a first Work (history_count == 0 / rate missing).
    """
    scores = snapshots["historical_rate"].copy()
    scores = scores.fillna(fitted["fallback_rate"])
    return scores


# --------------------------------------------------------------------------
# WHAT baseline: "this target's own TRAIN-set base rate, nothing else."
# --------------------------------------------------------------------------
#
# Unlike WHEN, there is no per-artist historical signal for new_style
# already sitting in the snapshots table (src/snapshots.py only exposes it
# as a target, not as a running rate), and adding one would mean changing
# that module, which is left untouched here on purpose. So this is
# deliberately the simplest honest baseline possible: the same constant
# score for every row, equal to the TRAIN set's own positive rate. A model
# that cannot beat "guess the overall rate, every time" at ranking rows by
# how likely new_style is has not learned anything a human would call
# predictive, however good its raw numbers look in isolation.


def fit_what_baseline(train_snapshots: pd.DataFrame) -> dict:
    """Learn the TRAIN set's new_style positive rate, ignoring rows where
    new_style itself is unknown (see src/snapshots.py: no next Work, or no
    style tags to compare)."""
    known = train_snapshots["new_style"].dropna()
    positive_rate = float(known.mean()) if len(known) else 0.0
    return {"positive_rate": positive_rate}


def score_what_baseline(snapshots: pd.DataFrame, fitted: dict) -> pd.Series:
    """Score every row for the WHAT baseline: the fitted constant,
    repeated. A constant score cannot rank rows against each other, so
    this baseline's PR-AUC is expected to equal the positive rate itself;
    that is the point, it is the "no information" floor WHAT has to clear.
    """
    return pd.Series(fitted["positive_rate"], index=snapshots.index)


# --------------------------------------------------------------------------
# Kill gate
# --------------------------------------------------------------------------


def _bootstrap_pr_auc(y_true: np.ndarray, scores: np.ndarray, artist_ids: np.ndarray,
                       n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    """Artist-level bootstrap of PR-AUC: resample ARTISTS with replacement
    (not individual rows), then take every row belonging to the resampled
    artists. This matches "artist-level bootstrap" already planned in
    docs/methodology.md: rows from the same artist are correlated (same
    career, same style habits), so resampling rows directly would
    understate how uncertain this really is.
    """
    unique_artists = np.unique(artist_ids)
    # Precompute each artist's row positions once, instead of re-filtering
    # the full array on every one of n_bootstrap iterations.
    rows_by_artist = {a: np.flatnonzero(artist_ids == a) for a in unique_artists}

    pr_aucs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_artists = rng.choice(unique_artists, size=len(unique_artists), replace=True)
        row_idx = np.concatenate([rows_by_artist[a] for a in sampled_artists])
        y_sample = y_true[row_idx]
        # A resample that happens to contain only one class has no PR-AUC
        # (undefined); skipped rather than crashing the whole bootstrap,
        # and this is rare enough at realistic sample sizes not to bias
        # the resulting interval in practice.
        if len(np.unique(y_sample)) < 2:
            pr_aucs[i] = np.nan
            continue
        pr_aucs[i] = average_precision_score(y_sample, scores[row_idx])
    return pr_aucs


def evaluate_kill_gate(
    y_true: pd.Series,
    model_scores: pd.Series,
    baseline_scores: pd.Series,
    artist_ids: pd.Series,
    confidence: float = KILL_GATE_CONFIDENCE,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 1,
) -> dict:
    """Decide whether a model clearly beats its baseline on PR-AUC, using
    non-overlapping bootstrap confidence intervals, not a bare point-estimate
    comparison (a model can look better by chance on one split).

    `y_true`, `model_scores`, `baseline_scores`, `artist_ids` must all be
    the same length, already restricted to rows where `y_true` is known
    (drop the NaN / censored rows described in src/snapshots.py's
    next_1y / next_2y / new_style docs *before* calling this).

    Rule: the gate passes (`model_beats_baseline: True`) only when the
    LOWER bound of the model's CI is strictly above the UPPER bound of the
    baseline's CI, at the given `confidence` (two-sided, so a 90%
    confidence level reads off the 5th and 95th percentiles of each
    bootstrap distribution).

    Returns a plain dict with both point estimates, both CIs, and the
    gate decision, meant to be logged/printed in full, not just the one
    boolean, so a human reviewing this later can see exactly how close the
    call was.
    """
    y = np.asarray(y_true)
    model_s = np.asarray(model_scores)
    baseline_s = np.asarray(baseline_scores)
    artists = np.asarray(artist_ids)

    assert len(y) == len(model_s) == len(baseline_s) == len(artists), (
        "evaluate_kill_gate: y_true, model_scores, baseline_scores and "
        "artist_ids must all be the same length and row-aligned."
    )
    assert not pd.isna(y).any(), (
        "evaluate_kill_gate: y_true contains missing values. Drop censored "
        "/ unknown rows (see src/snapshots.py) before calling this, do not "
        "pass them in and hope they're ignored."
    )
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    assert n_pos > 0 and n_neg > 0, (
        f"evaluate_kill_gate: y_true has {n_pos} positive and {n_neg} negative "
        "rows, so PR-AUC is undefined with only one class present. This usually "
        "means the test split is too small or too narrow (e.g. a temporal "
        "cutoff that leaves almost no rows after it) rather than a real bug; "
        "check the split before re-running."
    )

    model_point = float(average_precision_score(y, model_s))
    baseline_point = float(average_precision_score(y, baseline_s))

    rng = np.random.default_rng(seed)
    alpha = 1 - confidence
    lower_pct, upper_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)

    model_boot = _bootstrap_pr_auc(y, model_s, artists, n_bootstrap, rng)
    baseline_boot = _bootstrap_pr_auc(y, baseline_s, artists, n_bootstrap, rng)

    model_ci = tuple(np.nanpercentile(model_boot, [lower_pct, upper_pct]))
    baseline_ci = tuple(np.nanpercentile(baseline_boot, [lower_pct, upper_pct]))

    model_beats_baseline = model_ci[0] > baseline_ci[1]

    return {
        "model_pr_auc": model_point,
        "baseline_pr_auc": baseline_point,
        "model_pr_auc_ci": model_ci,
        "baseline_pr_auc_ci": baseline_ci,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "model_beats_baseline": bool(model_beats_baseline),
    }


def print_kill_gate_result(name: str, result: dict) -> None:
    """Human-readable one-block printout of an evaluate_kill_gate() result,
    meant for a notebook cell, not for logging/parsing.
    """
    verdict = "PASSES: keep this target" if result["model_beats_baseline"] else "FAILS: kill this target"
    pct = int(result["confidence"] * 100)
    print(f"Kill gate: {name}")
    print(f"  Model    PR-AUC: {result['model_pr_auc']:.3f}  "
          f"[{pct}% CI {result['model_pr_auc_ci'][0]:.3f} - {result['model_pr_auc_ci'][1]:.3f}]")
    print(f"  Baseline PR-AUC: {result['baseline_pr_auc']:.3f}  "
          f"[{pct}% CI {result['baseline_pr_auc_ci'][0]:.3f} - {result['baseline_pr_auc_ci'][1]:.3f}]")
    print(f"  Verdict: {verdict}")
