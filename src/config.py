"""Project settings: reads config.yaml once and exposes the project root."""

from pathlib import Path  # project root and config path

import yaml  # reads config.yaml

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory


def load_config() -> dict:
    """Read config.yaml from the project root and return it as a dict."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:  # the single settings file of the project
        return yaml.safe_load(f)  # plain nested dict


CONFIG = load_config()  # read once at import, shared by the modules that import it
assert CONFIG["years"]["development_last"] == CONFIG["years"]["holdout_start"] - 1, "years.development_last must equal years.holdout_start - 1"  # one development boundary for WHEN, WHAT and the style forecast
