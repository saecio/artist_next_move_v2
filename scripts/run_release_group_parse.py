"""Full run of parse_release_groups on the MusicBrainz dump, meant to be launched from the terminal."""

import sys  # extend the import path to src/
from pathlib import Path  # build paths relative to the project root

# Project root is the parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
# Make src/ importable without installing anything
sys.path.insert(0, str(ROOT / "src"))

from snapshots import CONFIG  # noqa: E402  (project settings read from config.yaml)
from year_imputation import parse_release_groups  # noqa: E402  (needs the path set above)

# Run the parser on the full dump and write the reference parquet
df, counts = parse_release_groups(
    path=str(ROOT / CONFIG["paths"]["raw"] / "release-group"),  # raw dump folder from config
    out_path=str(ROOT / CONFIG["paths"]["processed"] / "musicbrainz_release_groups.parquet"),  # processed folder from config
)
# Final tally: kept rows and every skip reason
print("DONE", dict(counts), flush=True)
# Usable share of the dump
print(f"usable share: {counts['kept'] / counts['total']:.1%}", flush=True)
