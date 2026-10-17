"""WHEN scores of the demo artists: reproduction control against the stored holdout, then per-work scores."""

import json  # stored result files
import time  # seconds per fit

import numpy as np  # score arrays
import pandas as pd  # dataframes
from sklearn.metrics import average_precision_score  # PR-AUC, same metric as evaluate_holdout

from app_data import load_demo_config  # demo settings, artist list
from config import CONFIG, ROOT  # project settings and root
from models import FINAL_XGB_PARAMS, make_xgboost  # locked final configuration

HOLDOUT_START = CONFIG["years"]["holdout_start"]  # first holdout year, first year of the output
HOLDOUT_END = CONFIG["years"]["holdout_end"]  # last holdout year, labels of later years are "after holdout"
TARGETS = ["next_1y", "next_2y"]  # WHEN targets
DECIMALS = 4  # decimals of the control comparison


def demo_artist_ids() -> list:
    """Discogs ids of the demo artists as ints, read from app/demo_config.yaml through the app loader."""
    return [int(a["discogs_id"]) for a in load_demo_config()["artists"]]  # ids may be float elsewhere, ints here


def fit_with_control(target: str, final_train: pd.DataFrame, holdout: pd.DataFrame) -> tuple:
    """Fit the final configuration once and recompute the stored holdout aggregate.

    Returns (score function, fitted model, control row). The control row holds the stored and recomputed
    PR-AUC, train rows and test rows, the fit seconds and whether the PR-AUC matches at 4 decimals.
    """
    stored = json.loads((ROOT / f"results/holdout/xgboost_final_{target}.json").read_text())  # stored holdout result
    fit_fn, score_fn = make_xgboost(target, **FINAL_XGB_PARAMS[target])  # locked model, same seed
    test = holdout[holdout[target].notna()]  # holdout rows with a resolved label, as in evaluate_holdout
    start = time.time()  # clock of the fit
    fitted = fit_fn(final_train)  # fit on the training rows of the holdout run
    seconds = time.time() - start  # fit seconds
    pr_auc = float(average_precision_score(test[target].to_numpy(), np.asarray(score_fn(test, fitted))))  # aggregate PR-AUC
    row = {
        "target": target,  # target of the control
        "stored_pr_auc": stored["pr_auc"],  # stored value
        "recomputed_pr_auc": pr_auc,  # recomputed value
        "difference": pr_auc - stored["pr_auc"],  # recomputed minus stored
        "stored_train_rows": stored["train_rows"],  # stored count
        "recomputed_train_rows": len(final_train),  # recomputed count
        "stored_test_rows": stored["test_rows"],  # stored count
        "recomputed_test_rows": len(test),  # recomputed count
        "fit_seconds": seconds,  # wall-clock of the fit
        "match": round(pr_auc, DECIMALS) == round(stored["pr_auc"], DECIMALS)  # match at 4 decimals
        and len(final_train) == stored["train_rows"] and len(test) == stored["test_rows"],  # and the row counts
    }
    return score_fn, fitted, row  # model pieces and the control row


def demo_scores(score_fn, fitted, target: str, rows: pd.DataFrame, train_counts: pd.Series) -> pd.DataFrame:
    """Per-work scores of the demo rows for one target, in the output layout.

    `rows` holds the demo snapshots from the holdout start on. `train_counts` is the number of training
    rows per artist_id. The label is empty (NA) when unresolved.
    """
    label = rows[target].astype("Int64")  # NA stays NA, resolved labels are 0 or 1
    return pd.DataFrame({
        "artist_id": rows["artist_id"].astype("int64").to_numpy(),  # whole-number artist id
        "snapshot_year": rows["year"].to_numpy(),  # snapshot year
        "work_id": rows["work_id"].to_numpy(),  # work of the snapshot
        "target": target,  # next_1y or next_2y
        "score": np.asarray(score_fn(rows, fitted)),  # probability of the label
        "label": label.to_numpy(),  # resolved label or empty
        "is_projection": label.isna().to_numpy(),  # true when the label is unresolved
        "label_after_holdout": (rows["year"] > HOLDOUT_END).to_numpy(),  # snapshot year after the last holdout year
        "artist_train_rows": rows["artist_id"].map(train_counts).fillna(0).astype("int64").to_numpy(),  # training rows of the artist, 0 if unseen
    })


def label_counts(table: pd.DataFrame) -> pd.DataFrame:
    """Rows, resolved-label rows and projection rows per target and snapshot year."""
    grouped = table.groupby(["target", "snapshot_year"])  # one group per target and year
    return pd.DataFrame({
        "rows": grouped.size(),  # all rows
        "labelled": grouped["is_projection"].apply(lambda x: int((~x).sum())),  # rows with a resolved label
        "projection": grouped["is_projection"].sum().astype(int),  # rows with no label
    })
