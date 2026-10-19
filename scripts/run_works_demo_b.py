"""Build results/demo/works_demo_b.csv (Works of the pool B artists for the timeline cards).

Refuses to run when the output file exists and writes with exclusive creation.

Usage:
    venv/bin/python3 scripts/run_works_demo_b.py
"""

import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from demo_works_b import run  # whole step

run()  # build, controls, write
