"""Per-work WHEN scores for the demo artists, with a reproduction control of the stored holdout aggregate.

The control refits the final configuration on the training rows of the holdout run and must match the stored
PR-AUC at 4 decimals before any file is written. The output is created once and never overwritten.

Usage:
    venv/bin/python3 scripts/run_when_demo_scores.py
"""

import sys  # module search path and exit
from pathlib import Path  # project root

import pandas as pd  # dataframes

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from backtesting import holdout_gap, split_holdout  # holdout split and label gap
from when_demo import HOLDOUT_END, HOLDOUT_START, TARGETS, demo_artist_ids, demo_scores, fit_with_control, label_counts  # demo pieces

OUT_PATH = ROOT / "results/demo/when_demo_scores.csv"  # new file, never overwritten

if OUT_PATH.exists():  # never overwrite
    sys.exit(f"{OUT_PATH.relative_to(ROOT)} already exists, nothing was run")
pd.set_option("display.width", 250)  # readable printed tables

snapshots = pd.read_parquet(ROOT / "data/processed/snapshots.parquet")  # cached snapshots
parts = split_holdout(snapshots)  # development, holdout, excluded_recent, demo
final_train, _ = holdout_gap(parts["development"])  # training rows of the holdout run
ids = demo_artist_ids()  # demo artist ids from app/demo_config.yaml

# Step 2: control, one fit per target, nothing written yet.
print("== STEP 2: control ==")
fitted, controls = {}, []  # (score function, model) per target, control rows
for target in TARGETS:  # one fit per target
    score_fn, model, row = fit_with_control(target, final_train, parts["holdout"])  # fit and recompute the aggregate
    fitted[target] = (score_fn, model)  # kept for the demo scores
    controls.append(row)  # control row
control = pd.DataFrame(controls).set_index("target")  # control table
print(control.round(6).T.to_string(), flush=True)
if not control["match"].all():  # stop before any file is written
    sys.exit("the control does not match the stored values at 4 decimals, nothing was written")

# Step 3: scores of the demo artists, every snapshot from the holdout start on.
print("== STEP 3: demo scores ==")
rows = snapshots[(snapshots["year"] >= HOLDOUT_START) & snapshots["artist_id"].isin(ids)]  # demo rows, labelled or not
train_counts = final_train.assign(artist_id=final_train["artist_id"].astype("int64")).groupby("artist_id").size()  # training rows per artist
rows = rows.assign(artist_id=rows["artist_id"].astype("int64"))  # whole-number ids before matching
table = pd.concat([demo_scores(*fitted[t], t, rows, train_counts) for t in TARGETS], ignore_index=True)  # both targets
print(label_counts(table).to_string())
print("labelled rows after the holdout (year >", HOLDOUT_END, ")")
after = table[table["label_after_holdout"] & ~table["is_projection"]]  # resolved labels after the holdout
print(after.groupby(["target", "snapshot_year"]).size().rename("labelled_after_holdout").to_string())
per_artist = table.groupby("artist_id")["artist_train_rows"].first().reindex(ids)  # training rows per demo artist
print("demo artists with 0 training rows:", int((per_artist.fillna(0) == 0).sum()), sorted(per_artist.index[per_artist.fillna(0) == 0].tolist()))
print("demo artists with no snapshot from", HOLDOUT_START, ":", sorted(set(ids) - set(table["artist_id"])))

# Step 4: checks, then write.
print("== STEP 4: checks ==")
key = ["artist_id", "snapshot_year", "work_id", "target"]  # row identity
print("duplicate keys:", int(table.duplicated(key).sum()))
print("scores in [0, 1]:", bool(table["score"].between(0, 1).all()))
print("rows before", HOLDOUT_START, ":", int((table["snapshot_year"] < HOLDOUT_START).sum()))
print("fit seconds:", control["fit_seconds"].round(1).to_dict())
if table.duplicated(key).any() or not table["score"].between(0, 1).all() or (table["snapshot_year"] < HOLDOUT_START).any():  # stop before writing
    sys.exit("a check failed, nothing was written")
table.to_csv(OUT_PATH, mode="x", index=False)  # exclusive creation
print("written", OUT_PATH.relative_to(ROOT), len(table), "rows")
