"""Write results/demo/route_table.csv: WHEN and WHAT verdicts next to their realities for every playable artist-year of both pools.

Read-only table for choosing a disclosed demo route, not an evaluation. Refuses to run when the file exists.

Usage:
    venv/bin/python3 scripts/run_demo_route_table.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from route_table import run  # whole step

run()  # build, write, print
