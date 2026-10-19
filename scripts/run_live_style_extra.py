"""Build the extra live style assets (Random Forest of target year 2023, panel rows, percentiles, actuals, control).

Refuses to run when any of the new files exists and writes with exclusive creation.

Usage:
    venv/bin/python3 scripts/run_live_style_extra.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from live_style_extra import run  # whole step

run()  # refit, control, write
