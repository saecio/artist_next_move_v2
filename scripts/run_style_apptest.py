"""AppTest run of tab 7 (Global style forecast). Prints only, writes no files.

Usage:
    venv/bin/python3 scripts/run_style_apptest.py
"""

import html  # unescape of table text
import re  # extraction of rows from the HTML
import sys  # module path
import time  # timings
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

import numpy as np  # plain top-k, independent of the project function
import pandas as pd  # tables
from streamlit.testing.v1 import AppTest  # headless app runner

from app_data import load_demo_config  # demo settings
from app_live import style_live  # live scoring, only for the timing
from config import CONFIG  # project settings, first style column
from style_forecast import build_panel  # panel builder that defines the base
from style_tab import mean_in_range, stored_in_range, style_summary, validation_figures  # figures behind the validation text

cfg = load_demo_config()  # settings
st_cfg = cfg["style_tab"]  # tab 7 settings
k, tx = st_cfg["keys"], st_cfg["texts"]  # widget keys and texts
N = st_cfg["top_n"]  # rows in the table
T0 = time.time()  # total clock
pred = pd.read_csv(ROOT / cfg["live"]["style_predictions"])  # stored forecast table
pred = pred[pred["method"] == "random_forest"]  # Random Forest only
act = pd.read_csv(ROOT / cfg["live"]["style_folder"] / cfg["live"]["style_actuals"])  # stored actuals
LOG = (ROOT / "results/style_forecast/test_20261015_173215.out").read_text(encoding="utf-8").split("\n")  # stored test log


def stored_hits() -> dict:  # hits per test year from the log
    """Random Forest hits at 10 per test year from the stored log."""
    i = next(n for n, l in enumerate(LOG) if l.startswith("== hits@10 per target year"))  # start of the table
    col = LOG[i + 1].split().index("random_forest")  # column of the Random Forest
    return {int(l.split()[0]): int(float(l.split()[1 + col])) for l in LOG[i + 2:i + 7]}  # year -> hits


def new_app() -> AppTest:  # first run
    """Fresh app, first run, no exception."""
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # first run
    assert not at.exception, [e.value for e in at.exception]  # no exception
    return at  # app


def tab6_many(at: AppTest) -> AppTest:  # check the opening view of tab 6, then select Several artists
    """Tab 6 opens on One artist; select the Several artists view, which these checks need."""
    mk = cfg["game"]["keys"]["mode"]  # mode radio key
    assert cfg["game"]["default_mode"] == "one" and at.radio(key=mk).value == "one", "tab 6 must open on One artist"  # opening view
    at.radio(key=mk).set_value("many").run()  # select mode 2
    assert not at.exception, [e.value for e in at.exception]  # no exception after the switch
    return at  # app


def go(at: AppTest, stage: str) -> AppTest:  # press the buttons up to a stage
    """Press Predict and then Show reality as far as the stage needs, from the stage the page is in."""
    order = ["placeholder", "predict", "reveal"]  # stages in order
    now = at.session_state[k["stage"]]["stage"] if k["stage"] in at.session_state else "placeholder"  # current stage
    if order.index(now) < 1 <= order.index(stage):  # step 1 is needed
        at.button(key=k["predict"]).click().run()  # Predict
    now = at.session_state[k["stage"]]["stage"]  # stage after step 1
    if order.index(now) < 2 <= order.index(stage):  # step 2 is needed
        at.button(key=k["reveal"]).click().run()  # Show reality
    assert not at.exception, [e.value for e in at.exception]  # no exception
    return at  # app


def set_year(at: AppTest, year: int) -> AppTest:  # choose a target year
    """Select a year in the tab 7 selector."""
    at.selectbox(key=k["year"]).select(year).run()  # year
    assert not at.exception, [e.value for e in at.exception]  # no exception
    return at  # app


def table(at: AppTest) -> str:  # tab 7 table without style blocks
    """HTML of the tab 7 table, style blocks removed."""
    h = next(x.value for x in at.get("html") if "poc-table" in x.value and tx["col_rank"] in x.value)  # the tab 7 element
    return re.sub(r"<style>.*?</style>", "", h, flags=re.S)  # table only


def rows(at: AppTest) -> list:  # rows as cell texts
    """Cell texts of every body row."""
    out = []  # rows
    for r in re.findall(r"<tr>(.*?)</tr>", table(at)):  # every row
        c = [html.unescape(re.sub(r"<[^>]+>", "", x)) for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r)]  # cell texts
        if c and c[0].isdigit():  # body row, header left out
            out.append(c)  # keep
    return out  # rows


def marks(at: AppTest) -> list:  # mark colors
    """Mark color classes in the table."""
    return re.findall(r"<span class='poc-mark (\w+)'[^>]*>", table(at))  # colors


def ref_rows(year: int) -> pd.DataFrame:  # stored top N of the year
    """Stored Random Forest top N rows of the year, in rank order."""
    return pred[(pred["target_year"] == year) & (pred["rank"] <= N)].sort_values("rank")  # stored forecast


def check_default() -> None:  # test 1
    """Every target year in the selector, the validation text with its figures, the status line, no exception."""
    at = new_app()  # app
    sel = at.selectbox(key=k["year"])  # year selector
    assert list(sel.options) == [str(y) for y in range(2018, 2027)], f"every year must be listed: {list(sel.options)}"  # nothing hidden
    figs = validation_figures(cfg)  # figures computed from the data
    text = st_cfg["validation_text"].format(**figs)  # expected full text
    assert not any(text in w.value for w in at.warning), "the full text is no longer a permanent warning"  # no warning
    assert st_cfg["texts"]["t7_sub"].format(n=figs["n"], hits=f"{float(figs['k']):g}", y_from=figs["first"], y_to=figs["last"]) in "".join(h.value for h in at.get("html")), "the tested-years sentence with the random baseline wording is the subtitle of the headline"  # subtitle
    ex = next(e for e in at.expander if e.label == tx["validation_label"])  # the validation expander, there before Predict
    assert not ex.proto.expanded and [c.value for c in ex.caption] == [text], "closed expander with the validation text only"  # contents
    assert cfg["notices"]["style_notice"] in "".join(h.value for h in at.get("html")) and cfg["notices"]["style_notice"] not in [w.value for w in at.warning], "the question is the headline, no banner"  # headline instead of the notice
    years = st_cfg["years"]["counted"]  # counted test years
    mean_range = mean_in_range(cfg)  # mean in-range count from the stored actuals and predictions
    assert [stored_in_range(y, st_cfg["top_n"]) for y in years] == [style_summary(y, st_cfg["top_n"], cfg)["in_range"] for y in years], "the stored counts must equal the live summary"  # cross-check
    mean_hits = sum(style_summary(y, st_cfg["top_n"], cfg)["hits"] for y in years) / len(years)  # mean top 10 hits
    assert "{:.1f}".format(mean_hits) == figs["k"] and f"gets {figs['k']} of the top" in text, "the hits wording must follow the data"  # hits figure
    print(f"1 default: years {list(sel.options)}, validation text in the closed expander, mean in range {mean_range} (about {round(mean_range)}), mean hits {mean_hits}, no exception")  # result


def check_stages() -> None:  # test 2
    """Placeholders before Predict, forecast without reality after it, everything after Show reality."""
    at = set_year(new_app(), 2019)  # a counted year
    h0 = table(at)  # table
    assert at.button(key=k["predict"]) and not any(b.key == k["reveal"] for b in at.button), "only Predict first"  # buttons
    first = ref_rows(2019).iloc[0]["style"]  # best forecast style
    assert first not in html.unescape(h0) and h0.count("class='poc-ph'") == 7 * N, "no style name before Predict, seven placeholders per row"  # placeholders
    at = go(at, "predict")  # Predict
    r1 = rows(at)  # rows
    assert len(r1) == N and r1[0][1] == first, "the forecast shows after Predict"  # forecast
    assert "poc-mark" not in table(at) and not any(b.key == k["predict"] for b in at.button) and at.button(key=k["reveal"]), "no mark yet, Show reality offered"  # buttons and marks
    growth = st_cfg["growth_note"].format(n=st_cfg["base_years"])  # growth note text, only in the tab 7 expander
    how = next(e for e in at.expander if e.label == tx["how_label"] and growth in [c.value for c in e.caption])  # the tab 7 expander after Predict
    assert st_cfg["growth_note"].format(n=st_cfg["base_years"]) in [c.value for c in how.caption], "the expander explains the order by expected growth"  # growth note
    at = go(at, "reveal")  # Show reality
    assert marks(at) and not any(b.key in (k["predict"], k["reveal"]) for b in at.button), "marks after Show reality, no step button"  # full
    print(f"2 stages: {7 * N} placeholders -> forecast of {len(r1)} styles without marks -> actuals and marks")  # result


def check_order_and_reality() -> None:  # tests 3 and 4
    """Order, ranks and expected Works equal the stored forecast; actuals and marks equal a recount; hits and in-range summaries equal the references."""
    want_hits = stored_hits()  # hits from the log
    for year in range(2018, 2023):  # final test years
        at = go(set_year(new_app(), year), "reveal")  # full table
        got, ref = rows(at), ref_rows(year)  # rendered and stored
        assert [g[1] for g in got] == ref["style"].tolist() and [int(g[0]) for g in got] == ref["rank"].tolist(), f"{year}: order and ranks differ from predictions.csv"  # order
        assert [g[2] for g in got] == [tx["percent_format"].format(v * 100) for v in ref["expected_growth"]], f"{year}: expected growth differs from predictions.csv"  # expected growth, percent format
        eg = ref["expected_growth"].tolist()  # stored expected growth in rank order
        assert all(x >= y for x, y in zip(eg, eg[1:])), f"{year}: stored rows must be ordered by expected growth descending"  # reference order
        assert [float(g[2].rstrip("%")) for g in got] == sorted([float(g[2].rstrip("%")) for g in got], reverse=True), f"{year}: rendered rows must be ordered by expected growth descending"  # rendered order
        assert [int(g[3].replace(",", "")) for g in got] == [int(round(v)) for v in ref["expected_works"]], f"{year}: expected Works differ"  # expected
        a = act[act["target_year"] == year].set_index("style")  # actuals of the year
        assert [int(g[5].replace(",", "")) for g in got] == [int(a.loc[s, "actual_works"]) for s in ref["style"]], f"{year}: actual Works differ"  # actual
        assert [g[6] for g in got] == [tx["percent_format"].format((np.exp(a.loc[s, "label"]) - 1) * 100) for s in ref["style"]], f"{year}: actual growth differs from exp(label) - 1 of actuals.csv"  # actual growth, independent of the app code
        inside = [bool(r["range_low"] <= a.loc[r["style"], "actual_works"] <= r["range_high"]) for _, r in ref.iterrows()]  # independent in-range
        cols = marks(at)  # mark colors
        assert cols == ["green" if i else "red" for i in inside], f"{year}: marks differ from the recount"  # marks
        e = pred[pred["target_year"] == year]  # all eligible styles of the year
        m = e.merge(a[["label"]].reset_index(), on="style")  # label of every style
        s = np.where(np.isnan(m["expected_growth"].to_numpy()), -np.inf, m["expected_growth"].to_numpy())  # scores
        hits = len(set(np.argsort(-s, kind="stable")[:N]) & set(np.argsort(-m["label"].to_numpy(dtype=float), kind="stable")[:N]))  # independent hits
        assert hits == want_hits[year], f"{year}: hits {hits} differ from the log {want_hits[year]}"  # hits equal the log
        text = html.unescape(table(at))  # table text
        assert tx["summary_hits"].format(k=hits, n=N) in text and tx["summary_range"].format(k=sum(inside), n=N) in text, f"{year}: summary lines differ"  # both summary lines
        print(f"3-4 {year}: order, ranks, expected, actuals and marks equal the references; hits {hits} of {N} (log {want_hits[year]}), in range {sum(inside)} of {N}")  # result


def check_flagged_and_none() -> None:  # tests 5 and 6
    """2023 and 2024 have asterisk marks in green or red and no summary; 2025 and 2026 show the text and no table."""
    for year in (2023, 2024):  # flagged years
        at = go(set_year(new_app(), year), "reveal")  # full table
        assert st_cfg["reasons"][year].format(year=year) in [i.value for i in at.info], f"{year}: the reason must be shown"  # reason
        cols = set(marks(at))  # mark colors
        assert cols and cols <= {"green", "red"} and "amber" not in table(at), f"{year}: marks must be green or red, no amber left: {cols}"  # colors
        every = re.findall(r"<span class='poc-mark (\w+)'[^>]*>(.*?)</span>", table(at))  # every mark with its glyph
        assert len(every) == N and all(g in ("✓*", "✗*") and (c == "green") == (g == "✓*") for c, g in every), f"{year}: every flagged mark has the asterisk and the matching color"  # asterisk and color
        text = html.unescape(table(at))  # table text
        assert tx["summary_hits"].split("{")[0] not in text and tx["summary_range"].split("{")[0] not in text, f"{year}: no summary for a flagged year"  # no summary
        assert "✓*" in text or "✗*" in text, f"{year}: marks carry the asterisk"  # asterisk
        assert "class='poc-cell" not in table(at), f"{year}: the actual cells use the normal cell style"  # no cell fill
        g24 = [x[6] for x in rows(at)]  # actual growth cells
        assert all("%" in c for c in g24), f"{year}: actual growth is shown (computed from actual Works and the base)"  # shown
        r = rows(at)  # rows
        assert len(r) == N and [int(x[0]) for x in r] == list(range(1, N + 1)), f"{year}: {N} ranked rows"  # rows
        if year == 2024:  # stored forecast exists for 2024
            assert [x[1] for x in r] == ref_rows(2024)["style"].tolist(), "2024 order equals predictions.csv"  # order
        print(f"5 {year}: reason shown, {len(r)} rows, marks {sorted(cols)} all with the asterisk, no amber, no summary")  # result
    for year in (2025, 2026):  # years without forecast
        at = set_year(new_app(), year)  # year
        assert st_cfg["none_text"] in [w.value for w in at.warning], f"{year}: the no-forecast text must be shown"  # text
        assert not any(b.key in (k["predict"], k["reveal"]) for b in at.button) and not any("poc-table" in x.value and tx["col_rank"] in x.value for x in at.get("html")), f"{year}: no button and no table"  # nothing to predict
        print(f"6 {year}: no-forecast text shown, no button, no table")  # result


def check_reset_and_tab6() -> None:  # test 7
    """The state resets on a year change; tab 7 does not touch the tab 6 state, and tab 6 keeps its mode."""
    at = tab6_many(new_app())  # tab 6 opens on One artist, Several artists is selected
    at = go(set_year(at, 2018), "reveal")  # full table
    assert at.session_state[k["stage"]]["stage"] == "reveal"  # revealed
    route_before = at.session_state[cfg["route"]["keys"]["stage"]]["stage"]  # tab 6 mode 2 stage
    set_year(at, 2019)  # year change
    assert at.session_state[k["stage"]]["stage"] == "placeholder" and not marks(at), "a year change must reset to placeholders"  # reset
    assert at.session_state[cfg["route"]["keys"]["stage"]]["stage"] == route_before, "tab 7 must not touch the tab 6 stage"  # tab 6 untouched
    assert at.radio(key=cfg["game"]["keys"]["mode"]).value == "many"  # tab 6 view unchanged by tab 7
    print(f"7 reset: year change returns tab 7 to placeholders; tab 6 stage ({route_before}) and mode untouched")  # result


def check_tab6_table() -> None:  # test 8
    """Tab 6 mode 2 shares the table styles with tab 7: after tab 7 is used, the tab 6 table has no Release column, the year-end tag sits under the artist and the reality cells carry titles."""
    route = cfg["route"]["texts"]  # tab 6 texts
    at = go(set_year(tab6_many(new_app()), 2019), "reveal")  # tab 6 on Several artists, tab 7 used first
    at.button(key=cfg["route"]["keys"]["predict"]).click().run()  # tab 6 Predict
    at.button(key=cfg["route"]["keys"]["reveal"]).click().run()  # tab 6 Show reality
    page = next(h.value for h in at.get("html") if route["col_artist"] in h.value and route["col_timing"] in h.value)  # the tab 6 table element
    head = re.findall(r"<th[^>]*>(.*?)</th>", " ".join(re.findall(r"<tr>(.*?)</tr>", page)[:2]))  # cells of the two header rows
    assert "Release" not in head and len(head) == 8, f"the tab 6 header must have 4 + 4 header cells and no Release: {head}"  # no Release column
    assert html.escape(route["year_end_tag"].format(year=cfg["route"]["year"])) in page and page.count("poc-sub'>" + html.escape(route["year_end_tag"].format(year=cfg["route"]["year"]))) >= 1, "the year-end tag sits under the artist"  # tag
    assert "Yes: " in page, "the reality cells carry the release title"  # titles
    print(f"8 tab 6 table after using tab 7: header {len(head)} columns without Release, year-end tag under the artist, titles in the reality cells")  # result


def check_base_years() -> None:  # config and model code must agree on the base length
    """The base of src/style_forecast.py equals the mean of the last style_tab.base_years counts, on a toy series with distinct values."""
    first = CONFIG["years"]["style_first_column"]  # first calendar column of the style table
    last = first + 8  # a few origins after the five warm-up columns
    counts = {y: 150 + 17 * (y - first) ** 2 + 5 * (y - first) for y in range(first, last + 1)}  # distinct count per year, all above the eligibility floor
    kept = pd.DataFrame([{"year": y, "styles": ["Toy"], "genres": ["Rock"]} for y, n in counts.items() for _ in range(n)])  # one Work row per count
    panel = build_panel(kept, last)  # panel with the base of every origin year
    n = st_cfg["base_years"]  # years in the base according to the config
    assert len(panel) > 0, "the toy series must produce panel rows"  # series is eligible
    for _, r in panel.iterrows():  # every origin year
        want = np.mean([counts[int(r["origin"]) - i] for i in range(n)])  # mean of the last n counts
        assert abs(r["base"] - want) < 1e-9, f"origin {int(r['origin'])}: base {r['base']} differs from the mean of the last {n} years {want}; style_tab.base_years and src/style_forecast.py disagree"  # drift guard
    print(f"9 base years: the panel base equals the mean of the last {n} counts on {len(panel)} toy origins")  # result


def timing() -> None:  # live scoring per year
    """Time of the live scoring of every forecast year, first call after the models are loaded."""
    style_live(2018)  # warm the caches
    parts = []  # timings
    for y in range(2018, 2025):  # every year with a forecast
        t = time.time()  # clock
        style_live(y)  # live rescoring
        parts.append(f"{y} {time.time() - t:.3f} s")  # timing
    print("live scoring per year:", ", ".join(parts))  # result


for fn in (check_default, check_stages, check_order_and_reality, check_flagged_and_none, check_reset_and_tab6, check_tab6_table, check_base_years, timing):  # all checks
    t = time.time()  # clock
    fn()  # run the check
    print(f"   {fn.__name__} {time.time() - t:.2f} s")  # timing
print(f"\ntotal time {time.time() - T0:.2f} s")  # total
