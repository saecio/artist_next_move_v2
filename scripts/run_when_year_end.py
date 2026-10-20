"""Build the WHEN assets for artist-years without a dated Work (features, scores, extended route table, controls).

Refuses to run when any of the four new files exists and writes with exclusive creation.

Usage:
    venv/bin/python3 scripts/run_when_year_end.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from when_year_end import run  # whole step

run()  # build, control, write, print
