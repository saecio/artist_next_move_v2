"""AppTest run of the static tabs 1 to 5 and 8. Prints only, writes no files.

Usage:
    venv/bin/python3 scripts/run_static_apptest.py
"""

import html  # unescape of the tab text
import json  # result files
import re  # text checks
import sys  # module path
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

import pandas as pd  # files
import pyarrow as arrow  # reads the chart data sent to the browser
from streamlit.testing.v1 import AppTest  # headless app runner

from app_data import load_demo_config  # demo settings
from config import CONFIG  # project settings

cfg = load_demo_config()  # settings
sc = cfg["static_tabs"]  # static tabs section
pa, t, poc = sc["paths"], sc["texts"], cfg["poc"]["texts"]  # paths and texts
folder = ROOT / pa["dir"]  # static data folder

at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()  # first run
assert not at.exception, [e.value for e in at.exception]  # no exception


def tab_text(i: int) -> str:  # text of one tab, tags removed
    """All html elements of tab i as plain text with the raw markup kept in a second string."""
    return "".join(x.value for x in at.tabs[i].get("html"))  # raw html of the tab


def plain(raw: str) -> str:  # tags and style blocks removed
    """Visible text of an html string."""
    return html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<style>.*?</style>", "", raw)))  # visible text


# files read independently of src/static_tabs.py
fig = pd.read_csv(folder / pa["data_figures"]).set_index("metric")["value"]  # Works and artists
h1 = pd.read_csv(folder / pa["h1_buckets"])  # Hypothesis 1 buckets
hyp = pd.read_csv(folder / pa["hypotheses"])  # hypotheses table
wm, wb = json.load(open(ROOT / pa["when_model"])), json.load(open(ROOT / pa["when_baseline"]))  # Timing files
hl = pd.read_csv(ROOT / pa["what_headline"]).set_index("model")  # Style headline
top = pd.read_csv(ROOT / pa["what_top10"]).set_index("model").loc[sc["rows"]["what_model"]]  # Style top group
y = CONFIG["years"]  # split years

raws = {i: tab_text(i) for i in (0, 1, 2, 3, 4, 7)}  # raw html of every static tab
texts = {i: plain(r) for i, r in raws.items()}  # visible text of every static tab

for i, txt in texts.items():  # all six tabs
    assert txt.strip(), f"tab {i + 1} is empty"  # something is drawn
    assert "Pending." not in txt, f"tab {i + 1} still pending"  # placeholder gone
    assert not re.search(r"\bWHEN\b|\bWHAT\b", txt), f"tab {i + 1} shows WHEN or WHAT"  # words not on screen
assert not any("Pending." in x.value for x in at.info), "no info box says Pending."  # no notice left
print("all six tabs: not empty, no Pending., no WHEN or WHAT")  # result

norm = lambda x: re.sub(r"\s+", " ", x)  # whitespace collapsed, wrapped SVG labels read as one line
n1, n2 = norm(texts[0]), norm(texts[1])  # visible text of tabs 1 and 2
works_file = pd.read_parquet(ROOT / CONFIG["paths"]["works"], columns=["year"])  # year column of the cleaned Works
undated = int(works_file["year"].isna().sum())  # Works without a year
tests_rows = json.load(open(ROOT / pa["when_tests"]))["test_rows"]  # test snapshots from the Timing result file
assert t["t1_headline"] in n1 and t["t1_use"] in n1, "tab 1 headline and use sentence"  # tab 1 sentences
assert "Timing and Style" not in n1 and len(raws[0]) > 0, "tab 1 old long sentence gone"  # old text removed
for number, label in ((f"{int(fig['artists']):,}", t["t1_kpi_artists"]), (f"{int(fig['works']):,}", t["t1_kpi_works"]), (str(len(sc["models_compared"])), t["t1_kpi_models"]), (f"{int(tests_rows):,}", t["t1_kpi_tests"])):  # four numbers
    assert number in n1 and label in n1, f"tab 1 KPI {label}"  # equal to the files and the config list
assert len(sc["models_compared"]) == 5 and raws[0].count("class='ui-card'") == 5, "five models, one schema card and four KPI cards"  # card count
for lab in (t["t1_node_source"], t["t1_node_clean"], t["t1_node_past"], t["t1_node_check"], f"{poc['label_timing']} {poc['q_timing']}", f"{poc['label_style']} {poc['q_style']}"):  # schema labels
    assert lab in n1, f"tab 1 schema label {lab}"  # labels from the config
assert "<svg viewBox=" in raws[0] and "px'" not in re.search(r"<svg[^>]*>", raws[0]).group(0), "schema scales through the view box"  # no fixed pixel width
print("tab 1 ok")  # result

assert t["t2_headline"] in n2 and t["t2_source"].format(date=sc["dump_date"]) in n2, "tab 2 headline and dump date"  # headline and date from the config
assert f"{int(fig['works']):,}" in n2 and f"{int(fig['artists']):,}" in n2, "tab 2 numbers equal data_figures.csv"  # file values
assert fig[sc["rows"]["metric_master"]] + fig[sc["rows"]["metric_standalone"]] == fig["works"], "source counts sum to the total"  # control
for number in (fig[sc["rows"]["metric_master"]], fig[sc["rows"]["metric_standalone"]]):  # both counts on screen
    assert f"{int(number):,}" in n2, "tab 2 schema count"  # from the file
for lab in (t["t2_in_master"], t["t2_in_standalone"].format(n=CONFIG["works"]["min_standalone_releases"]), t["t2_out"], t["t2_chart_title"]):  # labels
    assert lab in n2, f"tab 2 label {lab}"  # threshold from config.yaml
print("tab 2 ok")  # result

def spec_rows(i: int) -> list:  # data rows of every Altair chart of one tab
    """One list of rows per vega chart of tab i, read from the spec the app sent."""
    return [[r for d in e.proto.datasets for r in arrow.ipc.open_stream(d.data.data).read_all().to_pylist()] for e in at.tabs[i].get("vega_lite_chart")]  # all datasets of each chart (one per panel), Arrow bytes to rows


charts3 = spec_rows(2)  # the two tab 3 charts
assert len(charts3) == 2, "tab 3 has two charts"  # Style and Timing
h6 = pd.read_csv(folder / pa["h6_buckets"])  # Hypothesis 6 buckets
for rows, table, rate_col, ctrl in ((charts3[0], h1, "new_style_rate", sc["controls"]["h1_rates"]), (charts3[1], h6, "release_next_year_rate", sc["controls"]["h6_rates"])):  # Style then Timing
    assert len(rows) == 5, "five bars"  # one per bucket
    assert [r["bucket"] for r in rows] == list(table["bucket"]), "buckets equal the file"  # same order
    assert [round(r["rate_pct"], 2) for r in rows] == [round(v * 100, 2) for v in table[rate_col]] == ctrl, "rates equal the file and the controls"  # file and controls
    assert [r["rate_pct"] for r in rows] == sorted((r["rate_pct"] for r in rows), reverse=True), "bars descend"  # descending
chips = ["H1", "H6"]  # hypotheses behind the two H0 chips
h0 = hyp.set_index("id")  # by id
assert raws[2].count("ui-chip'") == 2 + len(sc["chip_hypotheses"]), "two H0 chips and four hypothesis chips"  # chip count
assert all(h0.loc[i, "h0_rejected"] == "yes" for i in chips) and texts[2].count(t["t3_col_h0"]) == 2, "H0 chips come from the file"  # file driven
for i in sc["chip_hypotheses"]:  # H2 to H5
    assert f"{h0.loc[i, 'question']} {h0.loc[i, 'result']}" in texts[2], f"chip {i}"  # question and result from the file
assert t["t3_headline"] in texts[2] and "<table" not in raws[2], "tab 3 headline shown, old table gone"  # headline true for the data
print("tab 3 ok")  # result

gap, first, last = y["label_gap"], y["holdout_start"], y["holdout_end"]  # split years
for s in (t["t4_train"].format(last=first - gap - 1), t["t4_gap"].format(n=gap), t["t4_test"].format(first=first, last=last)):  # three segments
    assert s in texts[3], f"tab 4 segment {s}"  # years from config.yaml
assert poc["q_timing"] in texts[3] and poc["q_style"] in texts[3] and t["t4_rule"] in texts[3], "tab 4 lines"  # questions and rule
print("tab 4 ok")  # result

m, b = hl.loc[sc["rows"]["what_model"]], hl.loc[sc["rows"]["what_rule"]]  # Style rows
expected = [(wm["positive_rate"], None, None), (wb["pr_auc"], wb["pr_auc_lo"], wb["pr_auc_hi"]), (wm["pr_auc"], wm["pr_auc_lo"], wm["pr_auc_hi"]), (m["positive_rate"], None, None), (b["pr_auc"], b["pr_auc_lo"], b["pr_auc_hi"]), (m["pr_auc"], m["pr_auc_lo"], m["pr_auc_hi"])]  # chart rows from the files
rows5, rows_dots = spec_rows(4)  # the tab 5 results chart and dot chart
got = [(r["value"], *(None if r[k] is None or r[k] != r[k] else r[k] for k in ("lo", "hi"))) for r in rows5]  # chart rows, a missing interval (null or NaN) is None
assert got == expected, "tab 5 chart data equals the files"  # exact file values
assert t["t5_headline"] in texts[4] and t["t5_sub"].format(first=first, last=last) in texts[4], "tab 5 headline and subtitle"  # true case
n_dots = sc["sizes"]["dots_side"] ** 2  # circles per grid
ks = [round(n_dots * top["holdout_positive_rate"]), round(n_dots * top["precision_top"])]  # expected filled circles, left and right
grids = [r for r in rows_dots if "state" in r]  # circle rows of both grids
assert len(grids) == 2 * n_dots, "two grids of circles"  # 2 x n
for g, k, rate in zip((grids[:n_dots], grids[n_dots:]), ks, (top["holdout_positive_rate"], top["precision_top"])):  # left then right
    assert sum(r["state"] == t["t5_dot_yes"] for r in g) == k, "filled circles equal round(n x rate)"  # from the files
    assert {r["rate"] for r in g} == {f"{rate * 100:.1f}%"}, "exact rate in the tooltip"  # from the files
assert t["t5_dot_head"].format(n=n_dots, b=ks[0], k=ks[1]) in norm(texts[4]), "line above the dot card"  # computed line
assert t["t5_col_rule"] == "History only" and t["t5_col_rule"] in texts[4], "History only label on screen"  # renamed
assert "Simple rule" not in texts[4] and all(r["label"] != "Simple rule" for r in rows5) and "Simple rule" not in open(ROOT / "app/demo_config.yaml").read(), "Simple rule is gone"  # no trace
for key in ("t5_tip_chance", "t5_tip_rule_timing", "t5_tip_rule_style", "t5_tip_model"):  # the three column tooltips
    assert html.escape(t[key], quote=True) in raws[4], f"tooltip {key}"  # on screen
import static_tabs  # the module the app uses
data5 = static_tabs.results_data(cfg)  # chart table
assert static_tabs.models_beat_rule(data5, cfg), "true case"  # data supports it
low = data5.copy()  # monkeypatched interval
low.loc[low["label"] == t["t5_col_model"], "lo"] = 0.0  # model lower bound below the rule upper bound
assert not static_tabs.models_beat_rule(low, cfg), "fallback case"  # headline condition off
orig = static_tabs.results_data  # keep the real function
static_tabs.results_data = lambda c: low  # app run with the patched table
try:  # always restore
    at2 = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()  # second run
finally:  # restore
    static_tabs.results_data = orig  # restore
assert t["t5_headline_neutral"] in "".join(x.value for x in at2.tabs[4].get("html")) and t["t5_headline"] not in "".join(x.value for x in at2.tabs[4].get("html")), "neutral headline when the interval check fails"  # fallback on screen
print("tab 5 ok")  # result

assert raws[7].count("class='poc-st-block'") == 3, "tab 8 has exactly three blocks"  # three blocks
assert t["t8_data"] in texts[7] and t["t8_next"].format(first=first, last=last) in texts[7], "tab 8 texts"  # fixed blocks
assert re.search(r"Tested on \d+ years \(\d{4} to \d{4}\)", texts[7]), "tab 8 style block shows years from the data"  # computed years
print("tab 8 ok")  # result

(chart2,) = spec_rows(1)  # the tab 2 chart
main = [r for r in chart2 if "works" in r and "label" not in r]  # yearly rows
peak_rows = [r for r in chart2 if "label" in r and "works" in r]  # the peak point
band_rows = [r for r in chart2 if "x0" in r]  # the shaded band
wpy = pd.read_csv(folder / pa["works_per_year"])  # yearly file
assert sum(int(r["works"]) for r in main) == int(wpy["works"].sum()) == int(fig["works"]) - undated, "yearly counts equal the dated share of the Works"  # control
top_row = wpy.loc[wpy["works"].idxmax()]  # maximum of the file
assert len(peak_rows) == 1 and int(peak_rows[0]["year"]) == int(top_row["year"]) and int(peak_rows[0]["works"]) == int(top_row["works"]) == max(int(r["works"]) for r in main), "peak annotation equals the data maximum"  # peak
assert peak_rows[0]["label"] == t["t2_peak"].format(year=int(top_row["year"]), works=f"{int(top_row['works']):,}"), "peak label"  # text from the config
assert len(band_rows) == 1 and int(band_rows[0]["x0"]) == cfg["design"]["incomplete_from_year"] and int(band_rows[0]["x1"]) == int(wpy["year"].max()), "band starts at design.incomplete_from_year"  # band
assert band_rows[0]["label"] == t["t2_incomplete"], "band label"  # text from the config
print("tab 2 chart ok")  # result

tag = cfg["header"]["tagline"]  # header tagline
assert any(cfg["header"]["title"] in e.value and tag in e.value for e in at.get("html")), "header with name and tagline"  # on the page
assert all(tag not in tab_text(i) for i in range(8)), "header sits above the tabs, so it shows on every tab"  # not inside a tab
css_text = "".join(e.value for e in at.get("html") if "<style>" in e.value)  # shared css
pal = cfg["design"]["palette"]  # palette
bg, tm = pal["background"], pal["timing"]  # button colors
assert f"[data-testid='stBaseButton-primary']{{background:{tm};border-color:{tm};color:{bg}}}" in css_text and "font-weight:" in css_text and ":hover" in css_text, "primary button: dark text, bold, hover"  # shared css
for f in list((ROOT / "app").glob("*.py")) + list((ROOT / "src").glob("*.py")):  # every python file of the app
    src = f.read_text()  # file text
    for line in src.splitlines():  # every line
        if "st.button(" in line:  # a button call
            assert "style" not in line and "unsafe" not in line, f"button bypasses the shared css in {f.name}"  # no inline style
            assert 'type="primary"' in line or "type=" not in line, f"button type in {f.name}"  # primary or default only
    if f.name != "ui_components.py":  # only the shared module may carry button css
        assert "stBaseButton" not in src and "stButton" not in src, f"button css outside ui_components in {f.name}"  # one place
print("header and buttons ok")  # result
assert all(not re.search(r"\bWHEN\b|\bWHAT\b", tab_text(i)) for i in range(8)), "no WHEN or WHAT in any html, tooltips included"  # raw html
print("no WHEN or WHAT in the raw html")  # result

import tomllib  # toml reader
pal = cfg["design"]["palette"]  # palette
theme = tomllib.load(open(ROOT / ".streamlit/config.toml", "rb"))["theme"]  # theme section
for key, name in (("backgroundColor", "background"), ("secondaryBackgroundColor", "card"), ("textColor", "text"), ("borderColor", "line"), ("primaryColor", "timing")):  # keys the toml repeats
    assert theme[key].lower() == pal[name].lower(), f"toml {key} equals the yaml palette {name}"  # same values
print("toml equals yaml palette")  # result
hexre = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b|rgba?\(")  # color literal
notebook_only = {"eda.py", "style_report.py"}  # notebook plotting modules, not used by the app
for f in sorted(list((ROOT / "app").glob("*.py")) + list((ROOT / "src").glob("*.py"))):  # every python file of the app and src
    if f.name not in notebook_only:  # the app does not import these two
        assert not hexre.search(f.read_text()), f"no color literal in {f.name}"  # colors only in the config
print("no color literal in the design files")  # result
placeholder = re.compile(r"\{[^}]*\}")  # a placeholder in braces
bad = {k: v for k, v in t.items() if re.search(r"\d", placeholder.sub("", v).replace("H0", ""))}  # texts with a digit outside a placeholder, the term H0 is allowed (t3_col_h0, t3_note)
assert not bad, f"digits outside placeholders: {bad}"  # none, no exception was needed
print("static_tabs texts: no digit outside a placeholder (only the term H0 allowed, in t3_col_h0)")  # result
print("run_static_apptest OK")  # done
