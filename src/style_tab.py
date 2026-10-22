"""Tab 7 logic: the Random Forest style forecast per target year, reality, marks and summary. Pure functions, no Streamlit code."""

import numpy as np  # arrays
import pandas as pd  # tables

from app_live import load_tables, style_live  # live rescoring and the stored files, as the app reads them
from poc_view import badge_kind  # same marks as mode 2 of tab 6
from config import ROOT  # project root
from style_forecast import FINAL_MODEL, top_hits  # registered hits definition


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
    reason = st["reasons"].get(year)  # reason of a flagged year, with a {year} placeholder
    text = {"counted": st["texts"]["flag_counted"], "flagged": reason and reason.format(year=year), "none": st["none_text"]}[kind]  # status text
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
    """Grey dot without reality; green check or red cross, followed by an asterisk when the year is flagged."""
    if in_range is None:  # no actual count
        return "grey", "mark_dot"  # grey dot
    return badge_kind(True, "yes" if in_range else "no", not flagged, "yes")  # same rule as mode 2


def hits_of(year: int, top_n: int, method: str) -> int:  # top n hits of one method
    """Top n hits of the method's ranking: styles of the real top n by actual growth among the eligible styles present in its top n."""
    rf = style_live(year)  # live rows of the year
    rf = rf[rf["method"] == method]  # one method only
    act = load_tables()["actuals"]  # stored actuals and labels
    m = rf.merge(act[act["target_year"] == year][["style", "label"]], on="style", how="left")  # label of every eligible style
    return int(top_hits(m["expected_growth"].to_numpy(), m["label"].to_numpy(dtype=float), top_n))  # registered definition


def style_summary(year: int, top_n: int, cfg: dict) -> dict | None:  # counted years only
    """Top n hits (styles of the real top n by actual growth among the eligible styles present in the forecast top n) and the in-range count of the forecast top n; None for any year that is not counted."""
    if year_kind(year, cfg) != "counted":  # flagged and no-forecast years have no summary
        return None  # never for those years
    hits = hits_of(year, top_n, "random_forest")  # registered definition
    shown = with_reality(forecast_table(year, top_n), year)  # forecast top n with reality
    known = shown[shown["in_range"].notna()]  # rows with an actual
    return {"hits": hits, "n": top_n, "in_range": int(known["in_range"].sum()), "m": len(known)}  # counts


def stored_in_range(year: int, top_n: int) -> int:  # in-range count of a stored year
    """Number of the stored Random Forest top n styles of the year whose actual Works fall inside the stored range (styles without an actual are not counted)."""
    tables = load_tables()  # stored files
    pred = tables["predictions"]  # stored forecast
    top = pred[(pred["method"] == "random_forest") & (pred["target_year"] == year) & (pred["rank"] <= top_n)]  # stored top n of the year
    act = tables["actuals"]  # stored actuals
    m = top.merge(act[act["target_year"] == year][["style", "actual_works"]], on="style", how="inner")  # actual Works of the top styles
    return int(((m["actual_works"] >= m["range_low"]) & (m["actual_works"] <= m["range_high"])).sum())  # inside the range


def mean_in_range(cfg: dict) -> float:  # mean in-range count over the counted years
    """Mean over the counted test years of the stored top n in-range count (the figure behind 'about 5 of the 10 top styles')."""
    years = cfg["style_tab"]["years"]["counted"]  # counted test years
    return float(np.mean([stored_in_range(y, cfg["style_tab"]["top_n"]) for y in years]))  # mean of the yearly counts


def forecast_beats_repeat(cfg: dict) -> bool:  # verdict of the forecast against repeating last year
    """True when the lower bound of the 95% paired bootstrap interval (final forecast model minus repeat last year, hits at top_n, test years) is above zero; False when the file or the row is missing."""
    path = ROOT / cfg["style_tab"]["paired_file"]  # stored paired differences of the test years
    if not path.exists():  # no stored interval
        return False  # the claim cannot be supported
    table = pd.read_csv(path, comment="#")  # model, metric, mean, ci_low, ci_high
    row = table[(table["model"] == FINAL_MODEL) & (table["metric"] == f"hits@{cfg['style_tab']['top_n']}")]  # forecast model, hits row
    return len(row) == 1 and bool(row["ci_low"].iloc[0] > 0)  # clearly better only when the whole interval is above zero

def style_subtitle(cfg: dict) -> str:  # subtitle of the tab 7 headline
    """The subtitle with the model hits, the tested years and the hits of picking at random (one decimal); the older tested-years line when the random file is missing."""
    st, figs = cfg["style_tab"], validation_figures(cfg)  # tab settings and figures
    try:  # the random mean comes from the stored file
        rand = pd.read_csv(ROOT / st["random_file"], comment="#").set_index("method").loc["random", "mean"]  # mean random hits
    except (OSError, KeyError, ValueError):  # no file or no row: fall back to the line without the random sentence
        return st["headline"].format(**figs)  # tested-years line
    return st["texts"]["t7_sub"].format(n=figs["n"], hits=f"{float(figs['k']):g}", y_from=figs["first"], y_to=figs["last"], rand=f"{rand:.1f}")  # new subtitle

def validation_figures(cfg: dict) -> dict:  # figures of the headline and the validation text
    """First and last counted year, mean top n hits of the Random Forest (k) and of repeating last year (b), n and the number of counted years."""
    st = cfg["style_tab"]  # tab settings
    years, n = st["years"]["counted"], st["top_n"]  # counted test years and the top size
    k, b = (f"{np.mean([hits_of(y, n, m) for y in years]):.1f}" for m in ("random_forest", "repeat_last"))  # mean hits per method, one decimal
    return {"first": min(years), "last": max(years), "k": k, "b": b, "n": n, "years": len(years)}  # placeholders of the texts
