"""Build results/demo/what_new_styles.csv (new styles of the next active year per demo artist and year).

Refuses to run when the output file exists and writes with exclusive creation.

Usage:
    venv/bin/python3 scripts/run_what_new_styles.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from what_new_styles import run  # whole step

run()  # build, control, write
