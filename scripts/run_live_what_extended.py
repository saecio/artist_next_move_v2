"""Build the extended WHAT live assets (features, labels, control) for every eligible demo artist-year from 2018.

Refuses to run when any of the three new files exists and writes with exclusive creation.

Usage:
    venv/bin/python3 scripts/run_live_what_extended.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from live_what_extended import run  # whole step

run()  # build, control, write
