"""Build results/demo/static/models_compared.csv from the saved notebook outputs (exclusive creation, nothing is retrained).

Usage:
    venv/bin/python3 scripts/run_models_compared_assets.py
"""

import sys  # module path and exit
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

from app_data import load_demo_config  # demo settings
from models_compared import build_models_compared  # reader of the notebook outputs
from static_assets import write_new  # exclusive CSV writer

cfg = load_demo_config()  # settings
sc = cfg["static_tabs"]  # static tabs section
out = ROOT / sc["paths"]["dir"] / sc["paths"]["models_compared"]  # output file
if out.exists():  # never overwrite
    sys.exit(f"{out.relative_to(ROOT)} already exists, nothing written")  # stop
table, missing = build_models_compared(cfg)  # rows read from the notebooks
write_new(out, table)  # exclusive creation
print(f"written {out.relative_to(ROOT)}: {len(table)} rows")  # confirmation
print(table.round(4).to_string(index=False))  # what was written
print("left out (scores not readable from a file):", missing or "none")  # omissions
