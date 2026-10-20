"""Write results/demo/release_pick.csv: the release title shown in the tab 6 reality cells, one Work per artist-year (rule B).

Refuses to run when the file exists.

Usage:
    venv/bin/python3 scripts/run_release_pick.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from release_pick import run  # whole step

run()  # build, check, write, print
