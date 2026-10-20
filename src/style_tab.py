"""Tab 7 logic: the Random Forest style forecast per target year, reality, marks and summary. Pure functions, no Streamlit code."""

import numpy as np  # arrays
import pandas as pd  # tables

from app_live import load_tables, style_live  # live rescoring and the stored files, as the app reads them
from poc_view import badge_kind  # same marks as mode 2 of tab 6
from style_forecast import top_hits  # registered hits definition


def year_kind(year: int, cfg: dict) -> str:  # counted, flagged or none
    """Kind of a target year from style_tab.years of the config."""
    for kind, years in cfg["style_tab"]["years"].items():  # each kind
        if year in years:  # found
            return kind  # counted, flagged or none
    raise KeyError(year)  # a year the config does not list


def style_years(cfg: dict) -> list:  # every target year of the selector
    """All target years of the config, sorted; none is hidden."""
    return sorted(y for ys in cfg["style_tab"]["years"].values() for y in ys)  # every year of every kind


def year_status(year: int, cfg: dict) -> dict:  # kind and the line shown for the year
    """Kind of the year and its status text (none for a counted year's reason, the reason for a flagged year, the no-forecast text for a year without forecast)."""
    st = cfg["style_tab"]  # tab settings
    kind = year_kind(year, cfg)  # kind
    text = {"counted": st["texts"]["flag_counted"], "flagged": st["reasons"].get(year), "none": st["none_text"]}[kind]  # status text
    return {"kind": kind, "text": text}  # status


def forecast_table(year: int, top_n: int) -> pd.DataFrame:  # top N styles of the forecast
    """Random Forest rows of the year with rank at most top_n, in rank order: style, expected growth over the base, expected Works, range and rank."""
    rf = style_live(year)  # live rows of the year
    rf = rf[rf["method"] == "random_forest"]  # Random Forest only, repeat_last stays out
    return rf[rf["rank"] <= top_n].sort_values("rank")[["rank", "style", "expected_growth", "expected_works", "range_low", "range_high"]].reset_index(drop=True)  # top N, ordered by the growth that defines the rank


def with_reality(table: pd.DataFrame, year: int) -> pd.DataFrame:  # actual Works and the in-range flag
    """Adds actual_works, actual_growth (growth over the base, the units of expected_growth) and in_range (True, False or None when the actual is missing) from actuals.csv."""
    act = load_tables()["actuals"]  # stored actual counts
    a = act[act["target_year"] == year][["style", "actual_works"]]  # actual Works of the year
    base = load_tables()["panel"]  # panel rows with the base, the mean of the three latest years
    base = base[base["origin"] == year - 1][["style", "base"]]  # base of the styles eligible at the origin
    out = table.merge(a, on="style", how="left").merge(base, on="style", how="left")  # actual and base next to the forecast
    out["actual_growth"] = out["actual_works"].clip(lower=1) / out["base"] - 1  # same units as expected_growth: exp(label) - 1, label = log(max(actual, 1) / base)
    inside = (out["actual_works"] >= out["range_low"]) & (out["actual_works"] <= out["range_high"])  # inside the range
    out["in_range"] = [None if pd.isna(w) else bool(i) for w, i in zip(out["actual_works"], inside)]  # None when no actual
    return out  # table with reality


def mark_for(in_range, flagged: bool) -> tuple:  # color and mark key of a row
    """Grey dot without reality; green check or red cross when counted; amber with an asterisk when the year is flagged."""
    if in_range is None:  # no actual count
        return "grey", "mark_dot"  # grey dot
    return badge_kind(True, "yes" if in_range else "no", not flagged, "yes")  # same rule as mode 2


def style_summary(year: int, top_n: int, cfg: dict) -> dict | None:  # counted years only
    """Top n hits (styles of the real top n by actual growth among the eligible styles present in the forecast top n) and the in-range count of the forecast top n; None for any year that is not counted."""
    if year_kind(year, cfg) != "counted":  # flagged and no-forecast years have no summary
        return None  # never for those years
    rf = style_live(year)  # live rows of the year
    rf = rf[rf["method"] == "random_forest"]  # Random Forest only
    act = load_tables()["actuals"]  # stored actuals and labels
    m = rf.merge(act[act["target_year"] == year][["style", "label"]], on="style", how="left")  # label of every eligible style
    hits = int(top_hits(m["expected_growth"].to_numpy(), m["label"].to_numpy(dtype=float), top_n))  # registered definition
    shown = with_reality(forecast_table(year, top_n), year)  # forecast top n with reality
    known = shown[shown["in_range"].notna()]  # rows with an actual
    return {"hits": hits, "n": top_n, "in_range": int(known["in_range"].sum()), "m": len(known)}  # counts
