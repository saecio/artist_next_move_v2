"""Build the live-scoring assets of the style forecast (Random Forest per test year, percentiles, panel rows, control).

Refuses to run when the output folder exists and writes with exclusive creation.

Usage:
    venv/bin/python3 scripts/run_live_style_assets.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from live_style_assets import run  # whole step

run()  # refit, control, write, reload check
