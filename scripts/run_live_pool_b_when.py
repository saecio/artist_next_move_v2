"""Build the WHEN assets of the pool B artists (features, scores with labels and flags, control) from the saved models.

Refuses to run when any of the three new files exists and writes with exclusive creation.

Usage:
    venv/bin/python3 scripts/run_live_pool_b_when.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from live_pool_b import run  # whole step

run()  # score, control, write
