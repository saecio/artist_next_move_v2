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

import numpy as np  # sums check
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

def spec_rows(i: int) -> list:  # data rows of every Altair chart of one tab
    """One list of rows per vega chart of tab i, read from the spec the app sent."""
    return [[r for d in e.proto.datasets for r in arrow.ipc.open_stream(d.data.data).read_all().to_pylist()] for e in at.tabs[i].get("vega_lite_chart")]  # all datasets of each chart (one per panel), Arrow bytes to rows


norm = lambda x: re.sub(r"\s+", " ", x)  # whitespace collapsed, wrapped SVG labels read as one line
n1, n2 = norm(texts[0]), norm(texts[1])  # visible text of tabs 1 and 2
works_file = pd.read_parquet(ROOT / CONFIG["paths"]["works"], columns=["year"])  # year column of the cleaned Works
undated = int(works_file["year"].isna().sum())  # Works without a year
tests_rows = json.load(open(ROOT / pa["when_tests"]))["test_rows"]  # test snapshots from the Timing result file
st_ = sc["story"]  # story settings
cutoff = st_["cutoff_year"]  # year the reader is in
assert t["t1_headline"].format(year=cutoff) in n1 and t["t1_use"] in n1, "tab 1 headline with the configured year and the use sentence"  # tab 1 sentences
for name, qkey in ((st_["timing_artist"], "t1_q_timing"), (st_["style_artist"], "t1_q_style")):  # both story artists
    assert name in n1 and t[qkey].format(artist=name) in n1, f"tab 1 artist and question {name}"  # name and question on screen
route = pd.read_csv(ROOT / pa["route_table"])  # artist names and ids
works_all = pd.concat([pd.read_csv(ROOT / f) for f in pa["works"]])  # Works of both pools
picks = pd.read_csv(ROOT / pa["release_pick"])  # picked titles
(chart_a, chart_b) = spec_rows(0)  # the two timelines
for chart, name in ((chart_a, st_["timing_artist"]), (chart_b, st_["style_artist"])):  # Timing then Style card
    aid = int(route.loc[route["artist"] == name, "artist_id"].iloc[0])  # Discogs id from the route table
    w = works_all[(works_all["artist_id"] == aid) & (works_all["year"] <= cutoff) & ~works_all["is_rerecording"]]  # dated up to the cutoff, no re-recordings
    expected_counts = w.groupby("year").size().to_dict()  # releases per year from the files
    dots = [r for r in chart if "releases" in r]  # timeline dots
    assert {int(r["year"]): int(r["releases"]) for r in dots} == expected_counts and len(dots) == len(expected_counts), f"timeline of {name} equals the demo files"  # same years and counts
    assert all(int(r["year"]) <= cutoff for r in dots) and max(int(r["year"]) for r in dots) <= cutoff, "no point after the cutoff year"  # no reveal
    pk = picks[picks["artist_id"] == aid].set_index("year")["title"]  # picked titles of this artist
    assert all(r["title"] == pk[int(r["year"])] for r in dots if int(r["year"]) in pk.index), "picked titles from release_pick.csv"  # tooltip titles
    assert any(r.get("text") == t["t1_here"] for r in chart) and any(r.get("text") == "?" for r in chart), "cutoff label and question mark"  # annotations
    year_labels = sorted(int(r["text"]) for r in chart if r.get("text", "").isdigit())  # muted year labels under the line
    assert year_labels == sorted({min(expected_counts), max(expected_counts), cutoff}), f"year labels of {name}: first dot, last dot and cutoff come from the data"  # equal to the files
kpi_values = [str(len(hyp)), str(len(sc["models_compared"])), f"{int(tests_rows):,}", t["t1_years_value"].format(first=y["holdout_start"], last=y["holdout_end"])]  # four numbers from the files and config
for number, label in zip(kpi_values, (t["t1_kpi_hyp"], t["t1_kpi_models"], t["t1_kpi_tests"], t["t1_kpi_years"])):  # four numbers
    assert number in n1 and label in n1, f"tab 1 KPI {label}"  # equal to the files and the config list
assert not {f"{int(fig['works']):,}", f"{int(fig['artists']):,}"} & set(kpi_values) and f"{int(fig['works']):,}" not in n1 and f"{int(fig['artists']):,}" not in n1, "tab 1 repeats no Works or artist count"  # no figure shared with tab 2
assert raws[0].count("class='ui-card'") == 4, "four KPI cards, no schema card"  # card count
assert "<svg" not in raws[0] and len(at.tabs[0].get("vega_lite_chart")) == 2, "tab 1 has no inline svg and two timelines"  # no svg
print("tab 1 ok")  # result

undated_file = int(pd.read_csv(folder / pa["data_quality"]).set_index("metric").loc["works_undated", "value"])  # undated Works from the new file
pct = f"{undated_file / int(fig['works']) * 100:.1f}"  # share in percent
assert t["t2_headline"] in n2 and t["t2_source"].format(date=sc["dump_date"]) in n2, "tab 2 headline and dump date"  # headline and date from the config
for key in ("works", "artists"):  # flow numbers
    assert f"{int(fig[key]):,}" in n2, f"tab 2 flow {key} equals data_figures.csv"  # file values
assert fig[sc["rows"]["metric_master"]] + fig[sc["rows"]["metric_standalone"]] == fig["works"], "source counts sum to the total"  # control
for number in (fig[sc["rows"]["metric_master"]], fig[sc["rows"]["metric_standalone"]]):  # both counts on screen
    assert f"{int(number):,}" in n2, "tab 2 flow count"  # from the file
for lab in (t["t2_in_master"], t["t2_in_standalone"].format(n=CONFIG["works"]["min_standalone_releases"]), t["t2_out"], t["t2_artists"], t["t2_chart_title"]):  # labels
    assert lab in n2, f"tab 2 label {lab}"  # threshold from config.yaml
assert "<svg" not in raws[1] and "ui-flow-box" in raws[1] and raws[1].count("ui-flow-box") == 4, "tab 2 flow is HTML and CSS with four boxes"  # no svg
boxes = re.findall(r"<div class='ui-flow-box'.*?</div></div>", raws[1])  # flow boxes
assert len(boxes) == 4 and all(plain(b_).strip() for b_ in boxes), "every flow box has text"  # non-empty cards
assert t["t2_undated"].format(pct=pct) in n2 and html.escape(t["t2_tip_undated"], quote=True) in raws[1], "undated chip equals the file value, with its tooltip"  # chip
assert undated_file == undated, "undated count equals the Works table"  # control
assert not any("<svg" in r for r in raws.values()), "no inline svg in any static tab"  # all tabs
print("tab 2 ok")  # result

charts3 = spec_rows(2)  # the two tab 3 charts
assert len(charts3) == 2, "tab 3 has two charts"  # Style and Timing
h6 = pd.read_csv(folder / pa["h6_buckets"])  # Hypothesis 6 buckets
for rows, table, rate_col, ctrl in ((charts3[0], h1, "new_style_rate", sc["controls"]["h1_rates"]), (charts3[1], h6, "release_next_year_rate", sc["controls"]["h6_rates"])):  # Style then Timing
    assert len(rows) == 5, "five bars"  # one per bucket
    assert [r["bucket"] for r in rows] == list(table["bucket"]), "buckets equal the file"  # same order
    assert [round(r["rate_pct"], 2) for r in rows] == [round(v * 100, 2) for v in table[rate_col]] == ctrl, "rates equal the file and the controls"  # file and controls
    assert [r["rate_pct"] for r in rows] == sorted((r["rate_pct"] for r in rows), reverse=True), "bars descend"  # descending
assert raws[2].count("ui-chip'") == 0, "Two patterns view: no chips under the charts"  # H0 chips and hypothesis chips are gone
assert t["t3_line_style"] in texts[2] and t["t3_line_timing"] in texts[2], "Two patterns view: muted one-liners under the charts"  # config texts
assert t["t3_headline"] in texts[2] and "<table" not in raws[2], "tab 3 headline shown, old table gone"  # headline true for the data
print("tab 3 ok")  # result

gap, first, last = y["label_gap"], y["holdout_start"], y["holdout_end"]  # split years
from backtesting import make_folds, method_spans  # fold function and the year spans drawn by tab 4
from style_tab import validation_figures  # tab 7 test years, the source of the tab 8 numbers
frame = pd.DataFrame({"year": range(1980, first), "artist_id": 1.0, "next_1y": 1.0, "history_count": 1})  # one row per development year
folds = make_folds(frame)  # the real fold function
spans = method_spans()  # years the tab draws
for f, s in zip(folds, spans[:-1]):  # practice runs
    assert f["train_years"].endswith(f"-{s['train_end']}") and f["gap_years"] == f"{s['gap_start']}-{s['test_start'] - 1}" and f["test_years"] == f"{s['test_start']}-{s['test_end']}", "spans equal make_folds"  # fold outputs
assert len(folds) == 5 and spans[-1] == {"train_end": first - gap - 1, "gap_start": first - gap, "test_start": first, "test_end": last}, "final split from config.yaml"  # five folds and the final exam
rows4 = re.findall(r"<div class='ui-m-row( final)?' data-train-end='(\d+)' data-gap='(\d+)-(\d+)' data-test='(\d+)-(\d+)'", raws[3])  # the six rows
assert len(rows4) == 6 and [bool(r[0]) for r in rows4] == [False] * 5 + [True], "tab 4 has six rows, the last one accented"  # five practice runs and the final exam
assert [tuple(map(int, r[1:])) for r in rows4] == [(s["train_end"], s["gap_start"], s["test_start"] - 1, s["test_start"], s["test_end"]) for s in spans], "row years equal the fold functions and the config"  # years
for i, s in enumerate(spans):  # visible test years and labels
    assert t["t4_test"].format(first=s["test_start"], last=s["test_end"]) in texts[3], f"test years of row {i + 1}"  # test block text
    assert (t["t4_final"] if i == 5 else t["t4_run"].format(n=i + 1)) in texts[3], f"label of row {i + 1}"  # row label
assert t["t4_sealed"] in texts[3] and raws[3].count("ui-m-lock'") == 1 and t["t4_train"] in texts[3], "Graded once with one lock, train label"  # final row only
assert t["t4_train_final"].format(year=first - gap - 1) in texts[3] and raws[3].count(t["t4_train"]) == 5, "final exam text uses the last training year of config.yaml, practice runs keep their label"  # year from config.yaml
assert all(t[k] in texts[3] for k in ("t4_legend_train", "t4_legend_gap", "t4_legend_test")) and raws[3].count("class='ui-m-sw'") == 3 and html.escape(t["t4_tip_gap"], quote=True) in raws[3], "legend with three swatches and the gap tooltip"  # legend
assert t["t4_headline"] in texts[3] and t["t4_caption"] in texts[3] and html.escape(t["t4_chips_tip"], quote=True) in raws[3] and html.escape(t["t4_caption_tip"], quote=True) in raws[3] and "never used to choose or tune" not in texts[3] and "never touched" not in texts[3] + texts[4], "headline, caption, caption tooltip, chip tooltip, old claims absent"  # config texts
assert re.findall(r"class='ui-m-tick'[^>]*>(\d+)<", raws[3]) == [str(v) for v in range(1990, last + 1, 10)] and cfg["design"]["method_axis_start"] == 1985, "ticks every ten years from the axis start"  # axis
chips4 = [html.unescape(c) for c in re.findall(r"class='ui-chip'[^>]*>([^<]*)", raws[3])]  # chip texts of the row
chips4, tags4 = chips4[:-3], chips4[-3:]  # model chips, then the three final-model tags
assert chips4 == [t["t5_col_rule"]] + sc["models_compared"] and raws[3].count("ui-m-arrow'") == len(chips4) - 1 and t["t4_final_flag"] in texts[3], "chips equal the config list plus past habits, arrows between"  # model row
assert re.search(r"ui-m-inline'><span class='ui-chip'[^>]*>XGBoost</span><span class='ui-info'", raws[3]), "info icon inline right after the final chip"  # attached to the chip
from models import make_what_xgboost, make_xgboost  # builders of the Timing and Style final models
from style_forecast import FINAL_MODEL, make_model  # final model of the forecast and its builder
assert (make_xgboost.__name__, make_what_xgboost.__name__, FINAL_MODEL, type(make_model(FINAL_MODEL)).__name__) == ("make_xgboost", "make_what_xgboost", "random_forest", "RandomForestRegressor"), "final model sources: XGBoost builders and the forest"  # sources
assert tags4 == [t["t4_final_timing"].format(model="XGBoost"), t["t4_final_style"].format(model="XGBoost"), t["t4_final_forecast"].format(model="Random Forest")] and t["t4_final_models"] in texts[3], "three final-model tags equal their sources"  # tags
assert "<svg" not in raws[3] and not re.search(r"\bWHEN\b|\bWHAT\b", texts[3]), "tab 4: no svg, no WHEN or WHAT"  # clean tab
print("tab 4 ok")  # result

m, b = hl.loc[sc["rows"]["what_model"]], hl.loc[sc["rows"]["what_rule"]]  # Style rows
expected = [(wm["positive_rate"], None, None), (wb["pr_auc"], wb["pr_auc_lo"], wb["pr_auc_hi"]), (wm["pr_auc"], wm["pr_auc_lo"], wm["pr_auc_hi"]), (m["positive_rate"], None, None), (b["pr_auc"], b["pr_auc_lo"], b["pr_auc_hi"]), (m["pr_auc"], m["pr_auc_lo"], m["pr_auc_hi"])]  # chart rows from the files
rows_timing, rows_style, rows_dots = spec_rows(4)  # the two tab 5 panels and the dot chart
rows5 = rows_timing + rows_style  # all six bars
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
assert t["t5_col_rule"] == "Past habits" and t["t5_col_rule"] in texts[4], "Past habits label on screen"  # renamed
cfg_text = open(ROOT / "app/demo_config.yaml").read()  # config file text
assert all(w_ not in texts[4] and w_ not in cfg_text and all(r["label"] != w_ for r in rows5) for w_ in ("Rule of thumb", "rule of thumb", "History only", "Simple rule")), "old names are gone"  # no trace
for key in ("t5_tip_chance", "t5_tip_rule_timing", "t5_tip_rule_style", "t5_tip_model"):  # the three column tooltips
    assert html.escape(t[key], quote=True) in raws[4], f"tooltip {key}"  # on screen
tags = {r["panel"]: r["tag"] for r in rows5 if r.get("tag")}  # advantage tags by panel
assert tags == {poc["label_timing"]: t["t5_advantage"].format(d=round(100 * (wm["pr_auc"] - wb["pr_auc"]))), poc["label_style"]: t["t5_advantage"].format(d=round(100 * (m["pr_auc"] - b["pr_auc"])))}, "advantage tags are whole points, model minus past habits from the files"  # computed
spec5 = "".join(e.proto.spec for e in at.tabs[4].get("vega_lite_chart")[:2])  # the two panel specs
assert len(at.tabs[4].get("vega_lite_chart")) == 3 and re.search(r'"width":\s*"container"', spec5) and all(f in spec5 for f in ("score_txt", "range_txt", "tag_tip")), "two panels fill their equal columns, friendly tooltip fields"  # layout and tooltips
assert all(x in spec5 for x in (t["t5_tt_name"], t["t5_tt_score"], t["t5_tt_range"].format(ci=round(CONFIG["bootstrap"]["confidence"] * 100)))) and not re.search(r'"title": "(value|lo|hi)"', spec5) and t["t5_tip_advantage"] in json.dumps(rows5), "tooltip titles come from the config, no internal field names"  # tooltips
for r_ in rows5:  # tooltip texts of every bar
    assert r_["score_txt"] == f"{r_['value']:.3f}", "score with three decimals"  # score
    assert (r_["range_txt"] is None) == (r_["label"] == t["t5_col_chance"]) and (r_["range_txt"] is None or r_["range_txt"] == t["t5_range_fmt"].format(lo=f"{r_['lo']:.3f}", hi=f"{r_['hi']:.3f}")), "95% range only on past habits and model bars"  # range
chart5 = json.dumps([e.proto.spec for e in at.tabs[4].get("vega_lite_chart")])  # specs sent to the browser
assert t["t5_line_timing"] in chart5 and t["t5_line_style"] in chart5, "visible past habits line under each panel title"  # chart subtitles
assert t["t5_caption"] in raws[4] and t["t5_headline"] in raws[4], "caption and headline"  # on screen
old_dot = ("the 100 the model ranks highest", "ranks highest", "artists picked at random", "Picking at random", "Picking with our model")  # removed wording
assert not any(w_ in texts[4] or w_ in cfg_text for w_ in old_dot), "old dot chart wording is gone"  # no trace
share = float(top["selected_share"])  # share of the top group from the file
s_ = round(100 * share)  # whole percent
assert s_ == round(100 * share) and t["t5_dot_head"].format(s=s_, n=n_dots, b=ks[0], k=ks[1]) in norm(texts[4]), "headline with s, k and b from the files"  # computed line
assert t["t5_dot_left"] in chart5 and t["t5_dot_right"].format(s=s_) in chart5, "grid titles"  # titles in the specs
assert t["t5_strip_top"].format(s=s_) in texts[4] and t["t5_strip_most"] in texts[4] and t["t5_strip_least"] in texts[4] and t["t5_dot_sort"] in texts[4], "strip labels and sort line"  # strip texts
wd = re.search(r"class='ui-strip-top' style='width:([\d.]+)%", raws[4])  # width of the highlighted segment
assert wd and abs(float(wd.group(1)) - share * 100) < 0.01 and "<svg" not in raws[4], "strip segment width equals selected_share, no svg"  # width from the file
assert ks[1] > ks[0], "right grid is the top segment, better than blind"  # statement stays true
assert sum(r.get("text") == t["t5_dot_caption"].format(n=n_dots) for r in rows_dots) >= 1, "caption under each big number"  # captions
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

tags = [r["tag"] for r in rows5 if r.get("tag")]  # the two advantage tags of tab 5, Timing then Style
d1, d2 = (re.search(r"\+(\d+) points", g).group(1) for g in tags)  # whole points inside the tags
v8 = validation_figures(cfg)  # tab 7 tested years
vals = {"d1": d1, "d2": d2, "ci": round(CONFIG["bootstrap"]["confidence"] * 100), "n": v8["years"], "first": v8["first"], "last": v8["last"], "test_first": first, "test_last": last}  # placeholders
assert raws[7].count("class='ui-card'") == 3 and raws[7].count("ui-close-dot'") == 3 and raws[7].count("class='ui-close-item'") == 9, "tab 8: three cards, a marker each, three items each"  # structure
for key in ("works", "mind", "next"):  # each card
    assert t[f"t8_{key}_title"] in texts[7], f"card title {key}"  # title
    for i in (1, 2, 3):  # items
        assert t[f"t8_{key}_{i}"].format(**vals) in texts[7], f"item {key} {i}"  # item with its numbers
assert t["t8_headline"] in texts[7] and "poc-st" not in raws[7], "tab 8 headline, old blocks gone"  # clean
assert "<svg" not in raws[7] and not re.search(r"\bWHEN\b|\bWHAT\b", texts[7]), "tab 8: no svg, no WHEN or WHAT"  # clean tab
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
assert band_rows[0]["tip"] == t["t2_band_tip"].format(year=int(top_row["year"])), "band tooltip names the data peak year"  # sentence from the config
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

# views of tab 2 and tab 3, selected with the pills (widget interaction)
vk = sc["view_keys"]  # pill keys
pill = lambda key: at.segmented_control(key=key)  # a pill by key
assert pill(vk["tab2"]).options == [t[k] for k in sc["views"]["tab2"]] and pill(vk["tab3"]).options == [t[k] for k in sc["views"]["tab3"]], "two pills on tab 2 and on tab 3"  # options from the config
assert pill(vk["tab2"]).value == t["t2_view_volume"] and pill(vk["tab3"]).value == t["t3_view_patterns"], "defaults are Volume and Two patterns"  # first view of each
assert t["t2_headline"] in texts[1] and len(spec_rows(1)) == 1, "Volume view is the current tab 2"  # unchanged default
print("pills ok")  # result

pill(vk["tab2"]).set_value(t["t2_view_genres"]).run()  # open the Genres view
assert not at.exception, [e.value for e in at.exception]  # no exception
gen = pd.read_csv(folder / pa["genre_share"])  # stored genre shares
g_raw = tab_text(1)  # raw html of tab 2
g_txt = plain(g_raw)  # visible text of tab 2
g_rows = spec_rows(1)  # chart data of tab 2
assert len(g_rows) == 1 and sorted((int(r["year"]), r["genre"], round(r["share"], 10)) for r in g_rows[0]) == sorted((int(a_), g_, round(s_, 10)) for a_, g_, s_ in zip(gen["year"], gen["genre"], gen["share"])), "Genres chart data equals genre_share_by_year.csv"  # exact file values
assert np.allclose(gen.groupby("year")["share"].sum(), 1.0), "shares sum to 1 per year"  # control
y0, y1 = cfg["design"]["genre_story_start_year"], int(gen["year"].max())  # story years
wide = gen.pivot(index="genre", columns="year", values="share")  # genre x year
best = (wide[y1] - wide[y0]).idxmax()  # genre with the largest absolute increase
line = t["t2g_headline"].format(genre=best, a=f"{wide.loc[best, y0] * 100:.0f}", b=f"{wide.loc[best, y1] * 100:.0f}", y0=y0, y1=y1)  # expected headline
assert line in g_txt and t["t2g_sub"] in g_txt, "Genres headline names the genre of largest increase, numbers equal the file"  # headline and subtitle
assert t["t2g_note"].format(year=cfg["design"]["incomplete_from_year"] - 1) in g_txt and t["t2g_title"] in g_txt, "years note and card title"  # note and title
assert t["t2_headline"] not in g_txt and "<svg" not in g_raw and not re.search(r"\bWHEN\b|\bWHAT\b", g_raw), "Genres view: Volume content gone, no svg, no WHEN or WHAT"  # clean view
spec = json.loads(at.tabs[1].get("vega_lite_chart")[0].proto.spec)  # chart spec
enc = spec["encoding"]  # encodings
assert spec["params"][0]["bind"] == "legend" and enc["color"]["legend"]["orient"] == "right", "legend on the right, clickable"  # legend selection
assert dict(zip(enc["color"]["scale"]["domain"], enc["color"]["scale"]["range"])) == cfg["design"]["genre_palette"] and set(gen["genre"]) == set(cfg["design"]["genre_palette"]), "one palette color per genre"  # config palette
assert [x["title"] for x in enc["tooltip"]] == [t["t2g_tip_year"], t["t2g_tip_genre"], t["t2g_tip_share"]], "tooltip: year, genre, share"  # tooltip titles
assert spec["height"] == cfg["design"]["chart"]["height_genres"], "chart height from the config"  # height
print("tab 2 genres ok")  # result
pill(vk["tab2"]).set_value(t["t2_view_volume"]).run()  # back to Volume
assert t["t2_headline"] in plain(tab_text(1)) and len(spec_rows(1)) == 1, "Volume view returns unchanged"  # switch back
print("tab 2 views ok")  # result

pill(vk["tab3"]).set_value(t["t3_view_tests"]).run()  # open the Six tests view
assert not at.exception, [e.value for e in at.exception]  # no exception
c_raw = tab_text(2)  # raw html of tab 3
c_txt = plain(c_raw)  # visible text of tab 3
cards = pd.read_csv(folder / pa["hyp_cards"]).merge(pd.read_csv(folder / pa["hyp_labels"]), on="id")  # stored cards with short labels
order = cards.sort_values("effect_size", ascending=False).reset_index(drop=True)  # strongest first
assert list(order["id"]) == ["H6", "H1", "H3", "H2", "H5", "H4"], "bars sorted by effect size"  # today's order
assert list(pd.read_csv(folder / pa["hyp_labels"])["short_label"]) == [v["short_label"] for v in sc["cards"]["items"].values()], "labels file equals the config texts"  # one source
assert t["t3s_headline"].format(label=order.loc[0, "short_label"]) in c_txt and order.loc[0, "id"] == "H6", "headline names the short label of the strongest effect"  # headline
assert t["t3s_line"] in c_txt and html.escape(t["t3s_tip_honesty"], quote=True) in c_raw, "muted line and honesty tooltip"  # muted line
size = cards.set_index("id")["effect_size"]  # by id
assert all(abs(size[i] - e) <= sc["controls"]["effect_tolerance"] for i, e in sc["controls"]["effect_sizes"].items()), "effect sizes equal the controls"  # controls
assert "ui-qtext" not in c_raw and "ui-meter" not in c_raw and "ui-answer" not in c_raw and "ui-chip'" not in c_raw and "hyp-card" not in c_raw, "no card grid, strength bars or chips"  # grid gone
charts = at.tabs[2].get("vega_lite_chart")  # the charts of the view
assert len(charts) == 1, "one chart"  # no mini charts
got_rows = [r for d in charts[0].proto.datasets for r in arrow.ipc.open_stream(d.data.data).read_all().to_pylist()]  # chart data
assert len(got_rows) == 6 and [r["short_label"] for r in got_rows] == list(order["short_label"]), "six bars in descending order of strength"  # rows
assert [round(r["effect_size"], 9) for r in got_rows] == [round(v, 9) for v in order["effect_size"]], "bar values equal the effect sizes of the file"  # values
cs = json.loads(charts[0].proto.spec)  # chart spec
assert cs["height"] == cfg["design"]["chart"]["height_strength"] and cs["layer"][0]["mark"]["type"] == "bar", "chart height and bar mark"  # style
assert cs["layer"][0]["encoding"]["x"]["scale"]["domain"] == [0, cfg["design"]["chart"]["strength_axis_max"]] and cfg["design"]["chart"]["strength_axis_max"] > cards["effect_size"].max(), "axis maximum is above every effect size"  # axis
ticks = cs["layer"][0]["encoding"]["x"]["axis"]["values"]  # explicit tick values of the Six tests axis
labels = [f"{v:.1f}" for v in ticks]  # the one-decimal labels shown on the axis
assert len(set(labels)) == len(labels) and ticks[0] == 0 and ticks[-1] == cfg["design"]["chart"]["strength_axis_max"] and cs["layer"][0]["encoding"]["x"]["axis"]["format"] == ".1f", "Six tests axis ticks are unique, from 0 to the maximum, one decimal"  # no repeated label
assert [r["color"] for r in got_rows] == [cfg["design"]["palette"]["timing" if i == "H6" else "style"] for i in order["id"]], "timing color for H6, style color for the others"  # colors
for r, (_, o) in zip(got_rows, order.iterrows()):  # bar by bar
    assert r["question"] == o["question"] and r["answer_text"] == o["answer"].format(**json.loads(o["values"])), f"tooltip question and answer of {o['id']} equal the file"  # tooltip text
    assert (t["t3s_tip_h3"] in r["strength_text"]) == (o["id"] == "H3") and o["effect_measure"] in r["strength_text"] and f"{o['effect_size']:.2f}" in r["strength_text"], f"tooltip strength of {o['id']}"  # strength and rank test line
assert "<svg" not in c_raw and not re.search(r"\bWHEN\b|\bWHAT\b", c_raw) and t["t3_headline"] not in c_txt, "Six tests view: no svg, no WHEN or WHAT, Two patterns content gone"  # clean view
print("tab 3 six tests ok")  # result
pill(vk["tab3"]).set_value(t["t3_view_patterns"]).run()  # back to Two patterns
back = spec_rows(2)  # the two charts again
assert [[r["bucket"] for r in rows] for rows in back] == [list(h1["bucket"]), list(h6["bucket"])] and t["t3_headline"] in plain(tab_text(2)), "Two patterns view returns with the file bars"  # switch back
print("tab 3 views ok")  # result
placeholder = re.compile(r"\{[^}]*\}")  # a placeholder in braces
bad = {k: v for k, v in t.items() if k not in ("t3s_line", "t4_final_forecast", "t5_tip_advantage") and re.search(r"\d", placeholder.sub("", v))}  # texts with a digit outside a placeholder, except the 0 to 1 scale of the muted line, the tab number of the forecast tag and the factor 100 of the advantage tooltip
assert not bad, f"digits outside placeholders: {bad}"  # none, no exception was needed
print("static_tabs texts: no digit outside a placeholder (only the 0 to 1 scale in t3s_line)")  # result
print("run_static_apptest OK")  # done
