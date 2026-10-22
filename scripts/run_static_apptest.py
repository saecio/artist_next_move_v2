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
from ui_components import base_css, header_html  # design css and header
from depth_views import models_data, trees_tie  # models compared table and the tie condition
from static_tabs import story_halves, tab2_headline_text  # tab 2 headline built from the data

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
    w = works_all[(works_all["artist_id"] == aid) & (works_all["year"] <= st_["last_release_year"]) & ~works_all["is_rerecording"]]  # dated up to the last year drawn, no re-recordings
    expected_counts = w.groupby("year").size().to_dict()  # releases per year from the files
    dots = [r for r in chart if "releases" in r]  # timeline dots
    assert {int(r["year"]): int(r["releases"]) for r in dots} == expected_counts and len(dots) == len(expected_counts), f"timeline of {name} equals the demo files"  # same years and counts
    assert max(int(r["year"]) for r in dots) <= st_["last_release_year"] < cutoff, "no point from the cutoff year on"  # no reveal
    pk = picks[picks["artist_id"] == aid].set_index("year")["title"]  # picked titles of this artist
    assert all(r["title"] == pk[int(r["year"])] for r in dots if int(r["year"]) in pk.index), "picked titles from release_pick.csv"  # tooltip titles
    assert any(r.get("text") == t["t1_here"] for r in chart) and any(r.get("text") == "?" for r in chart), "cutoff label and question mark"  # annotations
    year_labels = sorted(int(r["text"]) for r in chart if r.get("text", "").isdigit())  # muted year labels under the line
    assert year_labels == sorted({min(expected_counts), cutoff}) and str(max(expected_counts)) not in [r.get("text") for r in chart], f"year labels of {name}: first data year and the marker year only, no last-data-year label"  # equal to the files
wpy = pd.read_csv(folder / pa["works_per_year"])["year"]  # years of the Works chart file
import static_tabs  # module whose results_data the strip calls
nn = sc["sizes"]["dots_side"] ** 2  # artists per group
rel_t = pd.read_csv(ROOT / pa["calibration_timing"]).sort_values("bin")  # Timing reliability bins
assert rel_t["n"].max() - rel_t["n"].min() <= 1, "the ten Timing bins hold equal row counts"  # equal bins, so the last one is the top 10%
rel_s = pd.read_csv(ROOT / pa["calibration_style"])  # Style reliability bins
assert rel_s["n"].max() - rel_s["n"].min() <= 1, "the ten Style bins hold equal row counts"  # equal bins
k_t, b_t = round(nn * rel_t["observed_rate"].iloc[-1]), round(nn * json.load(open(ROOT / pa["when_model"]))["positive_rate"])  # Timing top bin rate and By chance rate
k_s, b_s = round(nn * top["precision_top"]), round(nn * top["holdout_positive_rate"])  # Style top group rate and positive rate, as the dot chart
pl = cfg["poc"]["texts"]  # Timing and Style labels
story = [  # the three cards: big number, label, muted line
    (t["t1_s1_big"].format(millions=f"{int(fig['works']) / 1e6:.1f}"), t["t1_s1_label"], t["t1_s1_sub"].format(artists=f"{int(fig['artists']):,}", y0=wpy.min(), y1=wpy.max())),  # Works
    (str(len(sc["models_compared"])), t["t1_s2_label"], t["t1_s2_sub"].format(tests=f"{int(tests_rows):,}")),  # models
    (t["t1_s3_big"].format(k=k_t, n=nn), pl["label_timing"], t["t1_s3_blind"].format(b=b_t, n=nn)),  # Timing half
    (t["t1_s3_big"].format(k=k_s, n=nn), pl["label_style"], t["t1_s3_blind"].format(b=b_s, n=nn)),  # Style half
]
assert k_t > b_t and k_s > b_s and t["t1_s3_cap_timing"].format(s=round(100 * top["selected_share"])) in n1 and t["t1_s3_cap_style"].format(s=round(100 * top["selected_share"])) in n1, "both halves above random, captions shown"  # card 3 shown
for number, label, line in story:  # three cards
    assert number in n1 and label in n1 and line in n1, f"tab 1 story card {label}"  # equal to the files and the config
assert t["t1_s1_tip"].format(date=sc["dump_date"]) in html.unescape(raws[0]) and t["t1_tip_models"].format(names=", ".join(sc["models_compared"])) in html.unescape(raws[0]) and t["t1_s3_tip"].format(s=round(100 * top["selected_share"]), y_from=y["holdout_start"], y_to=y["holdout_end"]) in html.unescape(raws[0]), "tab 1 tooltips"  # tooltips from the config
assert raws[0].count("class='ui-card'") == 3 and pl["label_timing"] in n1 and pl["label_style"] in n1, "three story cards"  # card count
for gone in ("t1_kpi_hyp", "t1_kpi_years", "t1_tip_hyp", "t1_tip_years", "t1_s3_plus", "t1_s3_label"):  # dropped keys
    assert gone not in t, f"{gone} removed from the config"  # no leftover
orig_res = static_tabs.story_halves  # real function
static_tabs.story_halves = lambda c: [(*h[:3], h[4], h[4]) if i else h for i, h in enumerate(orig_res(c))]  # Style k equal to b, not above random
try:  # always restore
    two_cards = plain("".join(x.value for x in AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run().tabs[0].get("html")))  # fresh run, tab 1 html
finally:  # restore
    static_tabs.story_halves = orig_res  # restore
assert t["t1_s3_cap_style"].format(s=round(100 * top["selected_share"])) not in norm(two_cards) and t["t1_s3_cap_timing"].format(s=round(100 * top["selected_share"])) not in norm(two_cards) and t["t1_s1_label"] in norm(two_cards) and t["t1_s2_label"] in norm(two_cards), "fallback: only cards 1 and 2"  # two cards
assert "<svg" not in raws[0] and len(at.tabs[0].get("vega_lite_chart")) == 2, "tab 1 has no inline svg and two timelines"  # no svg
print("tab 1 ok")  # result

undated_file = int(pd.read_csv(folder / pa["data_quality"]).set_index("metric").loc["works_undated", "value"])  # undated Works from the new file
pct = f"{undated_file / int(fig['works']) * 100:.1f}"  # share in percent
assert tab2_headline_text(cfg) in n2 and t["t2_source"].format(date=sc["dump_date"]) in n2, "tab 2 headline and dump date"  # headline and date from the config
for key in ("works", "artists"):  # flow numbers
    assert f"{int(fig[key]):,}" in n2, f"tab 2 flow {key} equals data_figures.csv"  # file values
assert fig[sc["rows"]["metric_master"]] + fig[sc["rows"]["metric_standalone"]] == fig["works"], "source counts sum to the total"  # control
for number in (fig[sc["rows"]["metric_master"]], fig[sc["rows"]["metric_standalone"]]):  # both counts on screen
    assert f"{int(number):,}" in n2, "tab 2 flow count"  # from the file
for lab in (t["t2_in_master"], t["t2_in_standalone"].format(n=CONFIG["works"]["min_standalone_releases"]), t["t2_out"], t["t2_artists"], t["t2_chart_title"]):  # labels
    assert lab in n2, f"tab 2 label {lab}"  # threshold from config.yaml
assert t["t2_by"] in n2 and t["t2_tip_out"] in raws[1] and "Works" not in n2, "tab 2: connector word, releases tooltip and the word releases only"  # one word for one concept
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
old_dot = ("the 100 the model ranks highest", "ranks highest", "artists picked at random", "Picking with our model", "Guessing blindly", "guessing blindly")  # removed wording
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
summ = pd.read_csv(ROOT / pa["calibration_summary"]).set_index("target")  # calibration summary, read independently of the app
gap_pts = {k: round(100 * float(v)) for k, v in summ["ece"].items()}  # round(100 x ECE) per target
pts_text = lambda n: cfg["ui"]["point_one" if n == 1 else "point_many"].format(n=n)  # singular for one point, plural otherwise
vals = {"d1": d1, "d2": d2, "ci": round(CONFIG["bootstrap"]["confidence"] * 100), "n": v8["years"], "first": v8["first"], "last": v8["last"], "test_first": first, "test_last": last, "gap_style": pts_text(gap_pts["what"]), "gap_timing": pts_text(gap_pts["next_1y"]), "k": story_halves(cfg)[1][3], "n": sc["sizes"]["dots_side"] ** 2}  # placeholders, the gaps are read from the calibration summary
assert raws[7].count("class='ui-card'") == 3 and raws[7].count("ui-close-dot'") == 3 and raws[7].count("class='ui-close-item'") == 11, "tab 8: three cards, a marker each, three items in What works, four in What to keep in mind, three in What comes next and the closing sentence"  # structure
for key in ("works", "mind", "next"):  # each card
    assert t[f"t8_{key}_title"] in texts[7], f"card title {key}"  # title
    for i in {"works": (1, 2, 3), "mind": (1, 2, 3, 4), "next": (1, 2, 3)}[key]:  # items
        want = t[f"t8_{key}_{i}"].format(**vals)  # item with its numbers
        if (key, i) == ("next", 1) and not trees_tie(models_data(cfg), cfg):  # the tie clause is not supported by the data
            want = t["t8_next_1_plain"]  # first item without the tie clause
        assert want in texts[7], f"item {key} {i}"  # item on screen
import depth_views  # module whose tie check the closing slide calls
import static_tabs  # builder of the closing slide
for tie, want in ((True, t["t8_next_1"]), (False, t["t8_next_1_plain"])):  # the tie claim both ways
    depth_views.trees_tie = lambda d, c, tie=tie: tie  # forced condition (restored below)
    assert want in html.unescape(static_tabs.tab8_html(cfg)), f"tab 8 first next item with tie {tie}"  # item follows the condition
depth_views.trees_tie = trees_tie  # real condition back
chip_text = lambda q, st_: t["t8_close_chip"].format(question=q, status=st_)  # closing chip as shown
real = (static_tabs.panel_beats_rule, static_tabs.forecast_beats_repeat)  # real conditions, restored below
for ok in (True, False):  # the closing block both ways
    static_tabs.panel_beats_rule = lambda d, c, ok=ok: pd.Series({poc["label_timing"]: ok, poc["label_style"]: ok})  # forced advantage flags
    static_tabs.forecast_beats_repeat = lambda c, ok=ok: ok  # forced forecast verdict
    page = html.unescape(static_tabs.tab8_html(cfg))  # closing slide with forced conditions
    sure = t["t8_close_beats"] if ok else t["t8_close_tested"]  # Timing and Style status
    assert chip_text(t["t8_close_timing_q"], sure) in page and chip_text(t["t8_close_style_q"], sure) in page, f"closing chips Timing and Style with {ok}"  # statuses
    assert chip_text(t["t8_close_trends_q"], t["t8_close_trend_beats"] if ok else t["t8_close_first"]) in page, f"closing chip Trends with {ok}"  # trends status
    assert t["t8_close_line"] in page and page.count("class='ui-chip'") == 3, f"closing sentence and three chips with {ok}"  # sentence and chip count
static_tabs.panel_beats_rule, static_tabs.forecast_beats_repeat = real  # real conditions back
import copy  # copy of the config for the missing-file case
import tempfile  # temporary interval files
paired = pd.read_csv(ROOT / cfg["style_tab"]["paired_file"], comment="#")  # stored interval, read independently of the app
low = paired[(paired["model"] == FINAL_MODEL) & (paired["metric"] == f"hits@{cfg['style_tab']['top_n']}")]["ci_low"].iloc[0]  # lower bound of the forecast model
want_trend = t["t8_close_trend_beats"] if low > 0 else t["t8_close_first"]  # chip text follows the sign of ci_low
assert static_tabs.forecast_beats_repeat(cfg) == (low > 0) and chip_text(t["t8_close_trends_q"], want_trend) in html.unescape(static_tabs.tab8_html(cfg)), "Trends chip follows the sign of ci_low in the stored file"  # real data
for lo, expected in ((0.3, True), (-0.4, False), (0.0, False)):  # lower bound above, below and at zero
    with tempfile.TemporaryDirectory() as tmp:  # scratch folder
        fake = Path(tmp) / "paired.csv"  # interval file with one chosen lower bound
        fake.write_text(f"model,metric,mean,ci_low,ci_high\n{FINAL_MODEL},hits@{cfg['style_tab']['top_n']},0.5,{lo},1.5\n", encoding="utf-8")  # forecast model row
        c2 = copy.deepcopy(cfg)  # config with the fake file
        c2["style_tab"]["paired_file"] = str(fake)  # absolute path replaces the root-relative one
        assert static_tabs.forecast_beats_repeat(c2) is expected, f"ci_low {lo} gives {expected}"  # sign rule
c3 = copy.deepcopy(cfg)  # config with a missing file
c3["style_tab"]["paired_file"] = "results/style_forecast/does_not_exist.csv"  # no such file
assert static_tabs.forecast_beats_repeat(c3) is False, "missing file gives False"  # chip falls back to A first step
for pal_key, q in (("timing", "t8_close_timing_q"), ("style", "t8_close_style_q"), ("highlight", "t8_close_trends_q")):  # chip colors on the real page
    assert t[q] in texts[7] and f"border-color:{cfg['design']['palette'][pal_key]}" in raws[7], f"closing chip {q} on screen in its color"  # question and color
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
bg, tm = pal["background"], pal["text"]  # button colors, ink fill with paper text
assert f"[data-testid='stBaseButton-primary']{{background:{tm};border-color:{tm};color:{bg}}}" in css_text and "font-weight:" in css_text and ":hover" in css_text, "primary button: paper text on ink, bold, hover"  # shared css
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

fo = cfg["design"]["font"]  # typefaces of the design
assert fo["body"] in theme["font"] and fo["heading"] in theme["headingFont"], "toml names both typefaces"  # theme fonts
faces = tomllib.load(open(ROOT / ".streamlit/config.toml", "rb"))["theme"]["fontFaces"]  # declared font files
assert {f["family"] for f in faces} == {fo["body"], fo["heading"]} and sorted({f["weight"] for f in faces}) == ["400", "500", "600", "700"], "four weights of both families"  # families and weights
assert all((ROOT / f["url"]).is_file() and (ROOT / f["url"]).stat().st_size > 1000 for f in faces), "every font file exists"  # files on disk
assert tomllib.load(open(ROOT / ".streamlit/config.toml", "rb"))["server"]["enableStaticServing"] is True, "static serving is on"  # fonts are served
assert all(list((ROOT / "app/static/fonts").glob(f"LICENSE-{n}.txt")) for n in ("inter", "fraunces")), "license texts sit next to the fonts"  # OFL texts
print("fonts declared and present")  # result
lum = lambda h: sum(w * (lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4)(int(h.lstrip("#")[i:i + 2], 16) / 255) for w, i in ((0.2126, 0), (0.7152, 2), (0.0722, 4)))  # relative luminance of a hex color
ratio = lambda a, b: (max(lum(a), lum(b)) + 0.05) / (min(lum(a), lum(b)) + 0.05)  # WCAG contrast ratio
floors = {"text": 4.5, "muted": 4.5, "timing": 4.5, "style": 4.0, "highlight": 2.9}  # style and highlight are used for big numbers, marks and borders
for name, floor in floors.items():  # colors used on cards
    print(f"contrast {name} on card: {ratio(pal[name], pal['card']):.2f}")  # value for the record
    assert ratio(pal[name], pal["card"]) >= floor, f"{name} over the card background is at least {floor}:1"  # contrast floor
assert cfg["design"]["shadow"] == "none", "cards have no shadow"  # border only
css = base_css(cfg)  # the style block
assert "gradient" not in css and "ui-rule" not in css and "ui-rule" not in header_html(cfg) and "rule" not in cfg["design"] and "box-shadow" not in css.replace("box-shadow:none", ""), "no gradient, no header rule, no shadow"  # editorial identity
assert "@media (max-width:1023px)" in css and "@media (max-width:639px)" in css, "two breakpoints"  # responsive
print("design pass 11 ok")  # result
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
assert tab2_headline_text(cfg) in texts[1] and len(spec_rows(1)) == 1, "Volume view is the current tab 2"  # unchanged default
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
assert tab2_headline_text(cfg) not in g_txt and "<svg" not in g_raw and not re.search(r"\bWHEN\b|\bWHAT\b", g_raw), "Genres view: Volume content gone, no svg, no WHEN or WHAT"  # clean view
spec = json.loads(at.tabs[1].get("vega_lite_chart")[0].proto.spec)  # chart spec
enc = spec["encoding"]  # encodings
assert spec["params"][0]["bind"] == "legend" and enc["color"]["legend"]["orient"] == "right", "legend on the right, clickable"  # legend selection
assert dict(zip(enc["color"]["scale"]["domain"], enc["color"]["scale"]["range"])) == cfg["design"]["genre_palette"] and set(gen["genre"]) == set(cfg["design"]["genre_palette"]), "one palette color per genre"  # config palette
assert [x["title"] for x in enc["tooltip"]] == [t["t2g_tip_year"], t["t2g_tip_genre"], t["t2g_tip_share"]], "tooltip: year, genre, share"  # tooltip titles
assert spec["height"] == cfg["design"]["chart"]["height_genres"], "chart height from the config"  # height
print("tab 2 genres ok")  # result
pill(vk["tab2"]).set_value(t["t2_view_volume"]).run()  # back to Volume
assert tab2_headline_text(cfg) in plain(tab_text(1)) and len(spec_rows(1)) == 1, "Volume view returns unchanged"  # switch back
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
# tab 5 depth views (pill, data equal to the files, headline conditions both ways)
import app_data  # module whose loader the app calls, patched below
import depth_views  # the module the app uses
from streamlit.testing.v1 import AppTest as _AppTest  # second runs


def find_key(obj, key):  # every value stored under `key` in a nested spec
    """All values of `key` anywhere in a nested dict or list."""
    found = []  # values seen
    if isinstance(obj, dict):  # a dict
        found += [obj[key]] if key in obj else []  # this level
        for v in obj.values():  # children
            found += find_key(v, key)  # deeper
    elif isinstance(obj, list):  # a list
        for v in obj:  # children
            found += find_key(v, key)  # deeper
    return found  # all values


def patched_run(owner, name, repl, view: str):  # a fresh app run with one function replaced, then the view opened
    """Run the app with owner.name replaced, open the tab 5 view, restore the function."""
    orig = getattr(owner, name)  # real function
    setattr(owner, name, repl)  # patch
    try:  # always restore
        run = _AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()  # fresh run
        run.segmented_control(key=vk["tab5"]).set_value(view).run()  # open the view
    finally:  # restore
        setattr(owner, name, orig)  # restore
    assert not run.exception, [e.value for e in run.exception]  # no exception
    return run  # the app


def tab5_html(app) -> str:  # raw html of tab 5
    """All html elements of tab 5 of an app run."""
    return "".join(x.value for x in app.tabs[4].get("html"))  # raw html


assert pill(vk["tab5"]).options == [t[k] for k in sc["views"]["tab5"]] and len(pill(vk["tab5"]).options) == 4, "tab 5 pill has four views"  # options from the config
assert pill(vk["tab5"]).value == t["t5_view_results"] and len(at.tabs[4].get("vega_lite_chart")) == 3, "default is the Results view, unchanged"  # default
q_t, q_s = poc["label_timing"], poc["label_style"]  # question labels
pal_ = cfg["design"]["palette"]  # palette

pill(vk["tab5"]).set_value(t["t5_view_years"]).run()  # Year by year
assert not at.exception, [e.value for e in at.exception]  # no exception
y_raw = tab_text(4)  # raw html
ch_y = spec_rows(4)  # the two line charts
byy = pd.read_csv(ROOT / pa["what_by_year"])  # Style per year
want_t = {(t["t5_col_model"], int(k), round(v["pr_auc"], 10)) for k, v in wm["by_year"].items()} | {(t["t5_col_rule"], int(k), round(v["pr_auc"], 10)) for k, v in wb["by_year"].items()}  # Timing points from the JSON files
want_s = {(t["t5_col_model"] if m == sc["rows"]["what_model"] else t["t5_col_rule"], int(yr), round(v, 10)) for m, yr, v in zip(byy["model"], byy["year"], byy["pr_auc"]) if m in (sc["rows"]["what_model"], sc["rows"]["what_rule"])}  # Style points from by_year.csv
got = [{(r["line"], int(r["year"]), round(r["score"], 10)) for r in rows} for rows in ch_y]  # points sent to the browser
assert len(ch_y) == 2 and got[0] == want_t and got[1] == want_s, "Year by year chart data equals the files"  # data
assert max(yr for _, yr, _ in want_t) == 2023 and max(yr for _, yr, _ in want_s) == 2022 and len(want_s) == 10, "Style stops one year earlier, no 2023 point is invented"  # missing year
assert all(f in "".join(e.proto.spec for e in at.tabs[4].get("vega_lite_chart")) for f in ("score_txt", t["t5y_tip_year"], t["t5y_tip_score"])), "friendly tooltips"  # tooltip rows
ahead = all(m > r for want in (want_t, want_s) for (ln, yr, m) in want if ln == t["t5_col_model"] for (ln2, yr2, r) in want if ln2 == t["t5_col_rule"] and yr2 == yr)  # model above past habits in every year of each question
assert ahead, "the files support the Year by year headline"  # true case
assert t["t5y_headline"] in plain(y_raw) and t["t5y_line"] in plain(y_raw), "headline and muted line shown"  # true case on screen
dip = depth_views.year_data(cfg).copy()  # real table
dip.loc[(dip["question"] == q_s) & (dip["line"] == t["t5_col_model"]) & (dip["year"] == 2020), "score"] = 0.0  # one Style year below past habits
fb = patched_run(depth_views, "year_data", lambda c: dip, t["t5_view_years"])  # patched run
assert t["t5y_headline_neutral"] in plain(tab5_html(fb)) and t["t5y_headline"] not in plain(tab5_html(fb)), "fallback headline when the model dips below past habits"  # false case
assert "<svg" not in y_raw and not re.search(r"\bWHEN\b|\bWHAT\b", y_raw), "Year by year: no svg, no WHEN or WHAT"  # clean view
print("tab 5 year by year ok")  # result

pill(vk["tab5"]).set_value(t["t5_view_models"]).run()  # Models compared
assert not at.exception, [e.value for e in at.exception]  # no exception
m_raw = tab_text(4)  # raw html
mc = pd.read_csv(folder / pa["models_compared"])  # stored table
ch_m = spec_rows(4)  # the two dot plots
assert len(ch_m) == 2, "two dot plots"  # charts
for rows, q, col in zip(ch_m, (q_t, q_s), (pal_["timing"], pal_["style"])):  # each question
    part = mc[mc["question"] == q]  # stored rows
    dots = [r for r in rows if "color" in r]  # dot rows
    assert {(r["model"], round(r["mean_score"], 10)) for r in dots} == {(m, round(v, 10)) for m, v in zip(part["model"], part["mean_score"]) if m != t["t5_col_rule"]}, f"{q}: dot data equals models_compared.csv"  # data
    assert all(r["color"] == (col if r["model"] == sc["final_model"] else pal_["rule"]) for r in dots), f"{q}: only the final model is in the question color"  # highlight
    rule_row = [r for r in rows if r.get("model") == t["t5_col_rule"]]  # past habits row
    assert len(rule_row) == 1 and abs(rule_row[0]["mean_score"] - float(part.loc[part["model"] == t["t5_col_rule"], "mean_score"].iloc[0])) < 1e-9, f"{q}: dashed line equals the stored past habits score"  # line
    order = [m for m in part.sort_values("mean_score", ascending=False)["model"] if m != t["t5_col_rule"]]  # best first
    assert order in find_key(json.loads(ch_m and at.tabs[4].get("vega_lite_chart")[(q_t, q_s).index(q)].proto.spec), "sort"), f"{q}: rows sorted from best to worst"  # sort order sent to the chart
assert set(mc["question"]) == {q_t, q_s} and (mc["source"].str.len() > 0).all() and {"Logistic Regression", "Naive Bayes", "Random Forest", "HistGradientBoosting", "XGBoost"} <= set(mc["model"]), "every model has a row and a source"  # file content
half = lambda q: ((lambda p: ((p["ci_high"] - p["ci_low"]) / 2).min())(mc[(mc["question"] == q) & mc["model"].isin(sc["tree_models"])]))  # smallest half-width of the trees
ties = all((lambda p: p["mean_score"].max() - p["mean_score"].min() < half(q) and mc[(mc["question"] == q) & (mc["model"] == sc["simple_model"])]["mean_score"].iloc[0] < p["mean_score"].min())(mc[(mc["question"] == q) & mc["model"].isin(sc["tree_models"])]) for q in (q_t, q_s))  # independent condition
assert ties and t["t5m_headline"] in plain(m_raw), "the files support the tie headline and it is shown"  # true case
assert t["t5m_tip"] in html.unescape(m_raw) and m_raw.count("data-tip=") >= 2, "info tooltip on each card title"  # tooltip
big = depth_views.models_data(cfg).copy()  # real table
big.loc[(big["question"] == q_t) & (big["model"] == "XGBoost"), "mean_score"] += 0.1  # XGBoost far above the other trees
fb = patched_run(depth_views, "models_data", lambda c: big, t["t5_view_models"])  # patched run
assert t["t5m_headline_neutral"] in plain(tab5_html(fb)) and t["t5m_headline"] not in plain(tab5_html(fb)), "fallback headline when the trees do not tie"  # false case
assert "<svg" not in m_raw and not re.search(r"\bWHEN\b|\bWHAT\b", m_raw) and not re.search(r"fold|practice|final test", plain(m_raw), re.I), "Models compared: no svg, no WHEN or WHAT, no folds, practice runs or final test"  # clean view
print("tab 5 models compared ok")  # result

pill(vk["tab5"]).set_value(t["t5_view_trust"]).run()  # Can you trust the score?
assert not at.exception, [e.value for e in at.exception]  # no exception
c_raw5 = tab_text(4)  # raw html
(ch_c,) = spec_rows(4)  # the reliability chart
rel_t, rel_s = (pd.read_csv(ROOT / pa[k]) for k in ("calibration_timing", "calibration_style"))  # reliability files
pts = [r for r in ch_c if "bin" in r]  # point rows
assert len(pts) == 20 and all(sum(r["question"] == q for r in pts) == 10 for q in (q_t, q_s)), "ten points per question"  # count
for q, f in ((q_t, rel_t), (q_s, rel_s)):  # each question
    assert sorted((int(r["bin"]), round(r["mean_score"], 10), round(r["observed_rate"], 10), round(r["ci_low"], 10), round(r["ci_high"], 10)) for r in pts if r["question"] == q) == sorted((int(b), round(a, 10), round(o, 10), round(lo, 10), round(hi, 10)) for b, a, o, lo, hi in zip(f["bin"], f["mean_score"], f["observed_rate"], f["ci_low"], f["ci_high"])), f"{q}: points and whiskers equal the reliability file"  # data
assert sorted(r["a"] for r in ch_c if "a" in r) == [0.0, 1.0], "the diagonal exists"  # perfect line
spec_c = at.tabs[4].get("vega_lite_chart")[0].proto.spec  # chart spec
assert spec_c.count('"domain": [0, 1]') + spec_c.count('"domain":[0,1]') >= 2 and "strokeDash" in spec_c, "equal axes from 0 to 1 and a dashed diagonal"  # axes and dash
tip_s = [r for r in pts if r["question"] == q_s and int(r["bin"]) == 10][0]["tip"]  # tooltip of the top Style point
top_s = rel_s.loc[rel_s["bin"].idxmax()]  # highest Style bin
assert tip_s == t["t5t_tip_point"].format(score=round(100 * top_s["mean_score"]), observed=round(100 * top_s["observed_rate"])), "tooltip sentence"  # sentence
assert abs(top_s["mean_score"] - top_s["observed_rate"]) < cfg["design"]["calibration"]["max_headline_gap"], "the file supports the headline"  # true case
c_txt5 = plain(c_raw5)  # visible text
assert t["t5t_headline"] in c_txt5 and t["t5t_sub"].format(n=cfg["design"]["calibration"]["group_size"], s=round(100 * top_s["mean_score"]), o=round(cfg["design"]["calibration"]["group_size"] * top_s["observed_rate"])) in c_txt5, "forecast headline and a subtitle whose numbers equal the highest Style bin"  # headline and subtitle
assert t["t5t_line"].format(style=pts_text(gap_pts["what"]), timing=pts_text(gap_pts["next_1y"])) in c_txt5, "muted line values equal round(100 x ECE), with the singular for one point"  # line
top_t = rel_t.loc[rel_t["bin"].idxmax()]  # highest Timing bin
assert (abs(top_t["mean_score"] - top_t["observed_rate"]) > cfg["design"]["calibration"]["note_gap"]) == (t["t5t_note"] in c_txt5), "the Timing note appears exactly when the top Timing bin gap is above note_gap"  # second line
assert t["t5t_tip"] in html.unescape(c_raw5), "info tooltip on the title"  # tooltip


def tight(c: dict) -> dict:  # config with strict thresholds
    """Copy of the loaded config with both calibration thresholds below any gap."""
    c["design"]["calibration"].update({"max_headline_gap": 0.0001, "note_gap": 1.0})  # headline off, note off
    return c  # patched config


real_load = app_data.load_demo_config  # real loader
fb = patched_run(app_data, "load_demo_config", lambda: tight(real_load()), t["t5_view_trust"])  # patched run
fb_txt = plain(tab5_html(fb))  # visible text
assert t["t5t_headline_neutral"] in fb_txt and t["t5t_headline"] not in fb_txt and "rated around" not in fb_txt and t["t5t_note"] not in fb_txt, "fallback headline and no note when the thresholds are below the gaps"  # false case
assert "<svg" not in c_raw5 and not re.search(r"\bWHEN\b|\bWHAT\b", c_raw5), "Can you trust the score: no svg, no WHEN or WHAT"  # clean view
print("tab 5 calibration ok")  # result

assert cfg["poc"]["gap_points"] == gap_pts, "the loader's gaps equal round(100 x ECE) of the summary"  # tab 6 and tab 8 numbers
assert "{gap}" in cfg["poc"]["texts"]["score_tip"] and "not a probability" not in cfg["poc"]["texts"]["score_tip"], "tab 6 tooltip uses the gap and no longer denies calibration"  # tooltip pattern
assert not any(re.search(r"not a probability|not calibrated|not been calibrated|calibration was not", v, re.I) for v in list(t.values()) + list(cfg["poc"]["texts"].values()) + list(cfg["notices"].values()) if isinstance(v, str)), "no on-screen text denies the calibration"  # texts
pill(vk["tab5"]).set_value(t["t5_view_results"]).run()  # back to Results
assert len(at.tabs[4].get("vega_lite_chart")) == 3 and t["t5_headline"] in plain(tab_text(4)), "Results view returns unchanged"  # switch back
print("tab 5 views ok")  # result
# design pass 8: alignment, gap wording, zoom caption, reading card, plural, tab 1 hook and photos
def find_columns(node):  # the container whose children are the two columns of the Results view
    """Depth-first search for the first container that holds columns, through the fragment and the slot."""
    kids = list(node.children.values())  # direct children
    if kids and all(k.type == "column" for k in kids):  # the row of columns
        return kids  # found
    for k in kids:  # go deeper
        if hasattr(k, "children"):  # containers only
            hit = find_columns(k)  # search below
            if hit:  # found below
                return hit  # pass it up
    return []  # none here


cols5 = find_columns(at.tabs[4])  # chart column and dot column of the Results view
cards5 = [next(iter(c.children.values())) for c in cols5]  # the two bordered cards
assert [int(re.search(r"pixel_height: (\d+)", str(c.proto)).group(1)) for c in cards5] == [sc["sizes"]["tab5_card_height"]] * 2, "both Results cards have the configured height"  # same top and bottom
dot_line_html = [x.value for x in cards5[1].children.values() if x.type == "html"][0]  # first html element of the dot card
assert "ui-card-title" in dot_line_html and plain(dot_line_html).strip().startswith(plain(static_tabs.dots_panels(cfg)[0]).strip()[:20]), "the dot line is the first line inside the dot card"  # moved inside
assert not any(x.type == "html" and "ui-card-title" in x.value for x in cols5[1].children.values()), "no title line outside the dot card"  # nothing above the card
print("tab 5 alignment ok")  # result

assert t["t4_legend_gap"] == "Gap" and t["t4_tip_gap"] == "We leave a gap of two years. Each answer looks ahead one to two years, so training right up to the test would let the model peek at what it is graded on.", "gap legend wording"  # exact texts
assert "Skipped" not in raws[3] and html.escape(t["t4_tip_gap"], quote=True) in raws[3], "tab 4 shows Gap and the new tooltip, no Skipped"  # on screen
print("tab 4 gap wording ok")  # result

pill(vk["tab5"]).set_value(t["t5_view_models"]).run()  # Models compared
dom_min = lambda q: (lambda v: v.min() - 0.15 * (v.max() - v.min()))(mc.loc[mc["question"] == q, "mean_score"])  # axis minimum of a question
assert cfg["design"]["chart"]["axis_pad"] == 0.15 and all(dom_min(q) > 0 for q in (q_t, q_s)), "both axes start above zero"  # premise
assert plain(tab_text(4)).count(t["t5m_zoom"]) == 2, "one zoom caption under each dot plot"  # captions
low0 = depth_views.models_data(cfg).copy()  # real table
low0.loc[(low0["question"] == q_t) & (low0["model"] == "Naive Bayes"), "mean_score"] = 0.0  # Timing axis now reaches zero
fb = patched_run(depth_views, "models_data", lambda c: low0, t["t5_view_models"])  # patched run
assert plain(tab5_html(fb)).count(t["t5m_zoom"]) == 1, "no caption for the axis that starts at zero"  # only Style
print("tab 5 zoom caption ok")  # result

pill(vk["tab5"]).set_value(t["t5_view_trust"]).run()  # Can you trust the score?
cal_ = cfg["design"]["calibration"]  # example numbers
read_txt = plain(tab_text(4))  # visible text
assert t["t5t_read"].format(pct=cal_["example_pct"], days=cal_["example_days"], total=cal_["example_total"]) in read_txt and t["t5t_read_title"] in read_txt, "reading card text from the config numbers"  # card
assert cal_["example_pct"] / 100 * cal_["example_total"] == cal_["example_days"], "the example is arithmetically right"  # 60% of 10 days is 6
assert app_data.point_text(cfg, 1) == "1 point" and app_data.point_text(cfg, 2) == "2 points" and "1 points" not in read_txt, "singular for one point, plural otherwise"  # plural rule
print("tab 5 reading card and plural ok")  # result

h1_txt = texts[0]  # tab 1 visible text
assert t["t1_headline"].format(year=2026) in h1_txt and t["t1_use"] in h1_txt and sc["story"]["cutoff_year"] == 2026, "tab 1 hook texts"  # new framing
assert not re.search(r"correct|right|predicted|accura|result|score", h1_txt, re.I), "the hook claims no prediction result"  # no claim
photo = cfg["design"]["photo"]  # photo settings
cr = pd.read_csv(ROOT / pa["artist_credits"]).set_index("artist")  # credits file
ts_ = sc["story"]["timing_artist"]  # Taylor Swift
assert (ROOT / pa["artist_photos"] / cr.loc[ts_, "file"]).is_file(), "premise: the Taylor Swift photo exists"  # file
assert re.search(rf"<img class='ui-photo' src='data:image/jpeg;base64,[^']+' alt='{ts_}' style='width:100%;max-width:{photo['width']}px;height:{photo['height']}px;border-radius:{photo['radius']}px", raws[0]), "photo element for Taylor Swift with the configured size"  # photo
assert "size" not in photo and f"object-position:{photo['focus']}" in raws[0] and f"font-size:{photo['name_font']}px" in raws[0] and f"font-size:{photo['question_font']}px" in raws[0], "old 56 px key gone, focus, name and question sizes from the config"  # card styling
assert photo["card_height"] <= 300 and cfg["design"]["chart"]["height_story"] == 100, "one-screen proxy: the card height stays small and the timeline is the short one"  # height proxy
have = [a_ for a_ in cr.index if (ROOT / pa["artist_photos"] / cr.loc[a_, "file"]).is_file()]  # artists whose photo file exists
wraps = re.findall(r"<div class='ui-photo-wrap'.*?</div></div>", raws[0], flags=re.S)  # photo columns with their overlay
assert len(wraps) == len(have) and all(w_.index("<img") < w_.index("class='ui-credit'") for w_ in wraps) and not any("position:absolute" in w_ for w_ in wraps), "the credit is a line under the photo inside the photo column, not an overlay"  # placement
assert raws[0].count("class='ui-credit'") == len(have) and not re.search(r"</div>\s*<div class='ui-caption'>Photo:", raws[0]) and "linear-gradient" not in raws[0] and f"font-size:{photo['credit_font']}px" in raws[0] and f"opacity:{photo['credit_opacity']}" in raws[0] and "text-overflow:ellipsis" in raws[0] and f"margin-top:{photo['credit_margin']}px" in raws[0], "credit sized from the config, no gradient, not after the question"  # styling
want_credit = lambda a_: (t["t1_credit"].format(author=cr.loc[a_, "author"], license=cr.loc[a_, "license"]) if isinstance(cr.loc[a_, "author"], str) and cr.loc[a_, "author"].strip() else t["t1_credit_noauthor"].format(license=cr.loc[a_, "license"]))  # text by the two patterns
for a_ in have:  # each artist with a photo
    assert plain(static_tabs.credit_html(cfg, a_)).strip() == want_credit(a_), f"credit text of {a_} matches credits.csv"  # file text
assert all(want_credit(a_) in h1_txt for a_ in have) and h1_txt.count("Photo:") == len(have), "one credit per photo that exists, none for a monogram"  # count
assert raws[0].count(f"padding-bottom:{photo['bottom_pad']}px") == len(have), "bottom padding sits under the credit inside the photo column"  # padding
st_q = {n_: t[k_].format(artist=n_) for n_, k_ in ((sc["story"]["timing_artist"], "t1_q_timing"), (sc["story"]["style_artist"], "t1_q_style"))}  # the two questions
assert all(html.escape(q_) in html.unescape(raws[0]) or q_ in html.unescape(raws[0]) for q_ in st_q.values()), "both question lines are in the rendered tab 1 html"  # question visible
gap_px = 16  # Streamlit vertical gap between two elements of a column
need = photo["name_font"] * 1.4 + cfg["design"]["box"]["gap"] + cfg["design"]["chart"]["height_story"] + photo["question_font"] * 1.5 + 2 * gap_px  # name block, timeline and question line
assert need <= photo["height"], f"name, timeline and question fit the photo height ({need} px of {photo['height']} px)"  # config proxy
assert photo["height"] + photo["credit_margin"] + photo["credit_line"] + photo["bottom_pad"] <= photo["card_height"] and all(sum(f"pixel_height: {photo['height']}" in str(c.proto) for c in col.children.values()) >= 0 for col in at.tabs[0].columns), "photo column plus credit and padding fits the card; the right column is a container of the photo height"  # proxy
assert raws[0].count(f"padding-bottom:{photo['bottom_pad']}px") == len(have) and photo["card_height"] == 262 and photo["height"] + photo["credit_line"] + photo["credit_margin"] == 224, "bottom padding on the question, photo shortened by the same amount"  # padding
marker = re.findall(r'"y2Offset":\s*(\d+)', "".join(e.proto.spec for e in at.tabs[0].get("vega_lite_chart")))  # marker offset sent to the browser
assert marker and all(int(v) == cfg["design"]["story"]["marker_below"] for v in marker), "marker ends a configured number of pixels below the timeline line"  # marker height proxy
labels_dy = re.findall(r'"dy":\s*(\d+)', "".join(e.proto.spec for e in at.tabs[0].get("vega_lite_chart")))  # label offset sent to the browser
assert str(cfg["design"]["story"]["label_dy"]) in labels_dy, "year labels lowered by the configured offset"  # label offset
bb_ = sc["story"]["style_artist"]  # Bad Bunny
assert ((ROOT / pa["artist_photos"] / cr.loc[bb_, "file"]).is_file() and f"alt='{bb_}'" in raws[0]) or (">BB<" in raws[0] and "ui-mono" in raws[0]), "Bad Bunny shows its photo when the file exists, else the monogram"
orig_cr = static_tabs.credit_row  # keep the real function
try:  # always restore
    static_tabs.credit_row = lambda c, n: {**orig_cr(c, n), "author": float("nan")}  # empty author
    assert plain(static_tabs.credit_html(cfg, ts_)).strip() == t["t1_credit_noauthor"].format(license=cr.loc[ts_, "license"]), "an empty author gives the pattern without the author"  # new pattern
    static_tabs.credit_row = lambda c, n: {**orig_cr(c, n), "license": float("nan")}  # empty license
    assert static_tabs.credit_html(cfg, ts_) == "", "no credit without a license"  # blank license
    static_tabs.credit_row = lambda c, n: {**orig_cr(c, n), "license": "  "}  # blank license text
    assert static_tabs.credit_html(cfg, ts_) == "", "no credit with a blank license"  # blank text
finally:  # restore
    static_tabs.credit_row = orig_cr  # restore
assert static_tabs.credit_html(cfg, ts_) != "" and (static_tabs.credit_html(cfg, bb_) == "") == (bb_ not in have), "credit only when the license is filled and the photo exists"  # filled case


def no_photos(c: dict) -> dict:  # config pointing at an empty folder
    """Copy of the loaded config whose photo folder has no files."""
    c["static_tabs"]["paths"]["artist_photos"] = "app/assets/artists_missing"  # folder that does not exist
    return c  # patched config


orig_load = app_data.load_demo_config  # real loader
app_data.load_demo_config = lambda: no_photos(orig_load())  # patch
try:  # always restore
    miss = _AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()  # fresh run
finally:  # restore
    app_data.load_demo_config = orig_load  # restore
m_raw0 = "".join(x.value for x in miss.tabs[0].get("html"))  # tab 1 html without photos
assert not miss.exception and "<img" not in m_raw0 and m_raw0.count("ui-mono") == 2 and ">TS<" in m_raw0 and "Photo:" not in plain(m_raw0), "monograms and no credit when the photo file is missing"  # fallback
assert "<svg" not in raws[0] and not re.search(r"\bWHEN\b|\bWHAT\b", raws[0]), "tab 1: no svg, no WHEN or WHAT"  # clean
print("tab 1 hook and photos ok")  # result
fresh = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=120).run()  # first load, fragments in place
assert not fresh.exception and [fresh.segmented_control(key=vk[k]).value for k in ("tab2", "tab3", "tab5")] == [t["t2_view_volume"], t["t3_view_patterns"], t["t5_view_results"]], "fragments keep the defaults Volume, Two patterns and Results on first load"  # defaults
for k_, v_ in (("tab2", "t2_view_genres"), ("tab3", "t3_view_tests"), ("tab5", "t5_view_years")):  # switch each pill
    fresh.segmented_control(key=vk[k_]).set_value(t[v_]).run()  # click inside the fragment
    assert not fresh.exception and fresh.segmented_control(key=vk[k_]).value == t[v_], f"{k_}: the pill switches the view without an exception"  # view chosen
print("fragments ok")  # result
placeholder = re.compile(r"\{[^}]*\}")  # a placeholder in braces
bad = {k: v for k, v in t.items() if k not in ("t3s_line", "t4_final_forecast", "t5_tip_advantage") and re.search(r"\d", placeholder.sub("", v))}  # texts with a digit outside a placeholder, except the 0 to 1 scale of the muted line, the tab number of the forecast tag and the factor 100 of the advantage tooltip
assert not bad, f"digits outside placeholders: {bad}"  # none, no exception was needed
print("static_tabs texts: no digit outside a placeholder (only the 0 to 1 scale in t3s_line)")  # result
print("run_static_apptest OK")  # done
