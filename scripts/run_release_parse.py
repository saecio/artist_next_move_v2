"""Full run of parse_releases on the MusicBrainz release dump, fed through stdin from the compressed archive.

Usage from the project root:
    nohup sh -c 'tar -xJO mbdump/release < ~/Downloads/release.tar.xz | venv/bin/python3 -u scripts/run_release_parse.py' > scripts/release_parse.log 2>&1 &
"""

import sys  # extend the import path to src/
from pathlib import Path  # build paths relative to the project root

# Project root is the parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
# Make src/ importable without installing anything
sys.path.insert(0, str(ROOT / "src"))

from snapshots import CONFIG  # noqa: E402  (project settings read from config.yaml)
from year_imputation import parse_releases  # noqa: E402  (needs the path set above)

# Read the stream from stdin and write the reference parquet; progress every 500k releases
df, counts = parse_releases(
    path="-",
    out_path=str(ROOT / CONFIG["paths"]["processed"] / "musicbrainz_releases.parquet"),  # processed folder from config
    log_every=500_000,
)
# Final tally: every outcome, including the catalog-number-with-year split
print("DONE", dict(counts), flush=True)
