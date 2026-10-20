"""Build the WHAT assets for artist-years without a dated Work (features, labels, scores, controls).

Refuses to run when any of the four new files exists and writes with exclusive creation.

Usage:
    venv/bin/python3 scripts/run_what_year_end.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from what_year_end import run  # whole step

run()  # build, control, write, print
