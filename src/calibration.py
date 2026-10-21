"""Read-only calibration check of the final models on the final holdout: reliability table, Brier score and ECE."""

import numpy as np  # array maths
import pandas as pd  # tables
from scipy.stats import norm  # normal quantile for the Wilson interval

from config import CONFIG  # project settings from config.yaml

CAL = CONFIG["calibration"]  # calibration settings


def wilson_interval(successes: np.ndarray, n: np.ndarray, confidence: float = CAL["confidence"]) -> tuple:
    """Wilson score interval of a rate. Inputs: successes and trials per bin. Output: (low, high) arrays."""
    z = norm.ppf(1 - (1 - confidence) / 2)  # two-sided normal quantile
    p = successes / n  # observed rate
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)  # shrunk centre
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)  # half width
    return centre - half, centre + half  # interval limits


def reliability_table(scores: np.ndarray, labels: np.ndarray, bins: int = CAL["bins"]) -> pd.DataFrame:
    """Equal-frequency reliability table. Output: one row per bin with mean score, observed rate, n and Wilson interval."""
    frame = pd.DataFrame({"score": scores, "label": labels})  # one row per scored row
    frame["bin"] = pd.qcut(frame["score"].rank(method="first"), bins, labels=False) + 1  # equal-frequency bins 1..bins
    table = frame.groupby("bin").agg(mean_score=("score", "mean"), observed_rate=("label", "mean"), n=("label", "size"), positives=("label", "sum")).reset_index()  # per-bin figures
    table["ci_low"], table["ci_high"] = wilson_interval(table["positives"].to_numpy(), table["n"].to_numpy())  # 95% Wilson interval
    return table.drop(columns="positives")  # positives only needed for the interval


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = CAL["bins"]) -> float:
    """Expected calibration error with equal-frequency bins: size-weighted mean of |observed rate - mean score|."""
    table = reliability_table(scores, labels, bins)  # reliability bins
    return float((table["n"] * (table["observed_rate"] - table["mean_score"]).abs()).sum() / table["n"].sum())  # weighted gap


def summarize(target: str, scores: np.ndarray, labels: np.ndarray, years: np.ndarray) -> dict:
    """Summary row of a target: rows, rates, Brier, Brier skill against the constant base rate, ECE and ECE per year."""
    base = labels.mean()  # constant base rate, the holdout positive rate
    brier = float(np.mean((scores - labels) ** 2))  # Brier score of the model
    brier_base = float(np.mean((base - labels) ** 2))  # Brier score of the constant base rate
    by_year = {int(y): round(ece(scores[years == y], labels[years == y]), 5) for y in sorted(np.unique(years))}  # ECE per holdout year
    return {"target": target, "rows": len(labels), "positive_rate": float(base), "mean_score": float(scores.mean()), "brier": brier, "brier_base": brier_base,
            "brier_skill": 1 - brier / brier_base, "ece": ece(scores, labels), "ece_by_year": str(by_year)}  # one summary row
