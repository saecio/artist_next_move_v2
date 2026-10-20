"""Small data files of the static tabs: the Hypothesis 1 and 6 buckets, the data figures and the hypotheses table."""

from pathlib import Path  # output paths

import pandas as pd  # tables
import yaml  # reads the demo config

from config import CONFIG, ROOT  # project settings and root
from eda import (  # existing EDA functions, nothing is recomputed by hand
    artist_style_transitions, gap_bucket_chi_square, load_when_gap_rows,  # transitions table and the Hypothesis 6 rows
    new_style_rate_by_prior_work_bucket, next_year_rate_by_gap_bucket,  # the two bucket tables
)


def static_config() -> dict:  # static_tabs section of the demo config
    """Read the static_tabs section of app/demo_config.yaml (no Streamlit import)."""
    with open(ROOT / "app" / "demo_config.yaml", encoding="utf-8") as f:  # the single settings file of the app
        return yaml.safe_load(f)["static_tabs"]  # only the static tabs section


def output_paths(sc: dict) -> dict:  # absolute path of each new file
    """Map each output key to its path under the static folder."""
    folder = ROOT / sc["paths"]["dir"]  # output folder
    return {k: folder / sc["paths"][k] for k in ("h1_buckets", "h6_buckets", "data_figures", "hypotheses", "works_per_year", "data_quality")}  # six files


def build_h1(works: pd.DataFrame) -> pd.DataFrame:  # Hypothesis 1 buckets
    """Five prior-work buckets: bucket, transitions, new_style_rate (same call as notebook 02 cell 50)."""
    transitions = artist_style_transitions(works)  # one row per consecutive known-year transition
    rates = new_style_rate_by_prior_work_bucket(transitions, CONFIG["what"]["min_previous_works"])  # rate and size per bucket
    out = rates.reset_index()[["prior_work_bucket", "n_transitions", "new_style_rate"]]  # keep the three columns
    out.columns = ["bucket", "transitions", "new_style_rate"]  # file column names
    out["bucket"] = out["bucket"].astype(str)  # plain text labels
    return out  # five rows


def build_h6() -> pd.DataFrame:  # Hypothesis 6 buckets
    """Five gap buckets: bucket, artist_years, release_next_year_rate (same calls as the Hypothesis 6 cells)."""
    rows = load_when_gap_rows(str(ROOT / CONFIG["paths"]["snapshots"]))  # artist-year rows with a resolved label
    gap_bucket_chi_square(rows)  # same test call as the notebook, keeps the rows identical to the tested ones
    rates = next_year_rate_by_gap_bucket(rows)  # count and rate per gap bucket
    out = rates.reset_index()[["gap_bucket", "n_rows", "rate"]]  # keep the three columns
    out.columns = ["bucket", "artist_years", "release_next_year_rate"]  # file column names
    out["bucket"] = out["bucket"].astype(str)  # plain text labels
    return out  # five rows


def build_figures(works: pd.DataFrame) -> pd.DataFrame:  # cleaned Works and artists
    """Number of cleaned Works and of artists, read from the cleaned Works table."""
    return pd.DataFrame({"metric": ["works", "artists"], "value": [len(works), works["artist_id"].nunique()]})  # two rows


def build_works_per_year(works: pd.DataFrame) -> pd.DataFrame:  # dated Works per year
    """Number of cleaned Works per year, counting only Works with a known year."""
    counts = works["year"].dropna().astype(int).value_counts().sort_index()  # Works per year
    return pd.DataFrame({"year": counts.index, "works": counts.values})  # two columns

def build_source_counts(works: pd.DataFrame, rows: dict) -> pd.DataFrame:  # Works per source
    """Works that come from a Master and from a standalone release, split by the work_source column."""
    by_source = works["work_source"].value_counts()  # master and synthetic counts
    return pd.DataFrame({"metric": [rows["metric_master"], rows["metric_standalone"]], "value": [int(by_source[rows["source_master"]]), int(by_source[rows["source_standalone"]])]})  # two rows

def check_new_controls(works: pd.DataFrame, per_year: pd.DataFrame, sources: pd.DataFrame, figures: pd.DataFrame) -> list:  # failed controls of the new files
    """The yearly counts sum to the dated Works of data_figures; the two sources sum to its total."""
    fig = dict(zip(figures["metric"], figures["value"]))  # metric -> value
    undated = int(works["year"].isna().sum())  # Works without a year
    checks = {  # name -> (found, expected)
        "works_per_year_sum": (int(per_year["works"].sum()), int(fig["works"]) - undated),  # dated share of the total
        "source_sum": (int(sources["value"].sum()), int(fig["works"])),  # master plus standalone
    }
    failed = []  # failures
    for name, (found, expected) in checks.items():  # compare each control
        print(f"control {name}: found {found}, expected {expected}, {'OK' if found == expected else 'FAIL'}")  # printed result
        if found != expected:  # control does not hold
            failed.append(name)  # remember it
    return failed  # empty when all hold

def rate_range(table: pd.DataFrame, column: str) -> dict:  # first and last rate of a bucket table
    """First and last bucket rate as percent text with one decimal."""
    return {"first": f"{table[column].iloc[0] * 100:.1f}%", "last": f"{table[column].iloc[-1] * 100:.1f}%"}  # placeholders of the result phrase


def build_summary(sc: dict, h1: pd.DataFrame, h6: pd.DataFrame) -> pd.DataFrame:  # one row per hypothesis
    """Hypotheses table from the config entries, with the rate ranges of H1 and H6 filled from the bucket files."""
    ranges = {"H1": rate_range(h1, "new_style_rate"), "H6": rate_range(h6, "release_next_year_rate")}  # the two quantitative results
    rows = []  # one dict per hypothesis
    for hid, h in sc["hypotheses"].items():  # in config order
        rows.append({"id": hid, "question": h["question"], "result": h["result"].format(**ranges.get(hid, {})), "test": h["test"], "h0_rejected": h["h0_rejected"], "source": h["source"]})  # one row
    return pd.DataFrame(rows)  # six rows


def check_controls(sc: dict, h1: pd.DataFrame, h6: pd.DataFrame, figures: pd.DataFrame) -> list:  # failed controls
    """Return the list of controls that do not hold (empty when all hold)."""
    c, failed = sc["controls"], []  # expected values and the failures
    fig = dict(zip(figures["metric"], figures["value"]))  # metric -> value
    checks = {  # name -> (found, expected)
        "h1_transitions": (int(h1["transitions"].sum()), c["h1_transitions"]),  # sum of the buckets
        "h1_rates": ([round(x * 100, 2) for x in h1["new_style_rate"]], c["h1_rates"]),  # rates in percent
        "h6_artist_years": (int(h6["artist_years"].sum()), c["h6_artist_years"]),  # sum of the buckets
        "h6_rates": ([round(x * 100, 2) for x in h6["release_next_year_rate"]], c["h6_rates"]),  # rates in percent
        "works": (int(fig["works"]), c["works"]),  # cleaned Works
        "artists": (int(fig["artists"]), c["artists"]),  # artists
    }
    for name, (found, expected) in checks.items():  # compare each control
        print(f"control {name}: found {found}, expected {expected}, {'OK' if found == expected else 'FAIL'}")  # printed result
        if found != expected:  # control does not hold
            failed.append(name)  # remember it
    return failed  # empty when all hold


def write_new(path: Path, table: pd.DataFrame) -> None:  # exclusive creation
    """Write a CSV that must not exist yet (mode x raises if it does)."""
    with open(path, "x", encoding="utf-8", newline="") as f:  # exclusive creation
        table.to_csv(f, index=False)  # no index column


def build_quality(works: pd.DataFrame) -> pd.DataFrame:  # data quality figures
    """One row: the number of Works without a year."""
    return pd.DataFrame({"metric": ["works_undated"], "value": [int(works["year"].isna().sum())]})  # metric, value


def check_quality_controls(sc: dict, quality: pd.DataFrame, figures: pd.DataFrame, per_year: pd.DataFrame) -> list:  # failed controls
    """Undated count must equal the configured value and Works minus the dated yearly counts."""
    fig = dict(zip(figures["metric"], figures["value"]))  # metric -> value
    found = int(quality["value"].iloc[0])  # undated count from the Works table
    checks = {"undated": (found, sc["controls"]["undated"]), "undated_vs_yearly": (found, int(fig["works"]) - int(per_year["works"].sum()))}  # name -> (found, expected)
    failed = []  # failures
    for name, (f, e) in checks.items():  # compare each control
        print(f"control {name}: found {f}, expected {e}, {'OK' if f == e else 'FAIL'}")  # printed result
        if f != e:  # control does not hold
            failed.append(name)  # remember it
    return failed  # empty when all hold
