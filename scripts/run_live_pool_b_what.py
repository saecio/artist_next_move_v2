"""Build the WHAT assets of the pool B artists (features, labels, control) from the saved model.

Refuses to run when any of the three new files exists and writes with exclusive creation.

Usage:
    venv/bin/python3 scripts/run_live_pool_b_what.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from app_data import load_demo_config  # demo settings
from live_what_extended import run  # whole step, parametrised

cfg = load_demo_config()  # settings
ids = [int(a["discogs_id"]) for a in cfg["artists_b"]]  # the 21 pool B ids
paths = cfg["pool_b"]["paths"]  # pool B file paths
files = {k: ROOT / paths[k] for k in ("what_features", "what_labels", "what_control")}  # output files, keys as in pool_b.paths

run(ids, files)  # score, control, write
