"""Refit the final WHEN and WHAT models, build the demo feature rows, control against the stored scores, then save.

Nothing is written when the control differs by more than 0.00005. Refuses to run when the output folder exists.

Usage:
    venv/bin/python3 scripts/run_live_assets.py
"""

import sys  # module search path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # make src/ importable

from live_assets import run  # the whole step

run()  # refit, control, write
