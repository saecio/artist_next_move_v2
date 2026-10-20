"""AppTest run of mode 2 of tab 6 (several artists in one year). Prints only, writes no files.

Usage:
    venv/bin/python3 scripts/run_multi_apptest.py
"""

import html  # unescape of table text
import re  # extraction of rows from the HTML
import sys  # module path
import time  # timings
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

import pandas as pd  # tables
from streamlit.testing.v1 import AppTest  # headless app runner

from app_data import all_artists, load_demo_config  # demo settings
from app_live import artist_timeline, load_tables  # live data layer, used only to cross-check
from config import CONFIG  # project settings
from demo_multi import multi_rows, multi_summary  # logic under test

cfg = load_demo_config()  # settings
rt, g = cfg["route"], cfg["game"]  # route and game settings
k, tx = rt["keys"], rt["texts"]  # widget keys and texts
pb = cfg["pool_b"]["paths"]  # pool B file paths
names = {int(a["discogs_id"]): a["name"] for a in all_artists(cfg)}  # id -> name
by_name = {v: i for i, v in names.items()}  # name -> id
T0 = time.time()  # total clock


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


def new_app(stage: str = "reveal") -> AppTest:  # first run, mode 2 opens first, then the buttons
    """Fresh app, first run, with no exception, advanced to a stage (the full table by default)."""
    at = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=60).run()  # first run
    assert not at.exception, [e.value for e in at.exception]  # no exception
    return go(at, stage)  # buttons up to the stage


def table_html(at: AppTest) -> str:  # HTML of the mode 2 table
    """HTML of the mode 2 table element."""
    return re.sub(r"<style>.*?</style>", "", next(h.value for h in at.get("html") if "poc-table" in h.value), flags=re.S)  # the table element without its style blocks


def rows_of(at: AppTest) -> list:  # parsed rows
    """(artist, row HTML) of every table row, in display order."""
    return [(html.unescape(a), r) for a, r in re.findall(r"<tr><th>(.*?)</th>(.*?)</tr>", table_html(at)) if a != tx["col_artist"]]  # artist cells and the rest, header row left out


def stored_when(ids: list, year: int) -> pd.DataFrame:  # independent WHEN reference from the stored files
    """Stored WHEN score and label of the representative Work of the year, read from the csv files only (no app function)."""
    works = pd.concat([pd.read_csv(ROOT / "results/demo/works_demo.csv"), pd.read_csv(ROOT / pb["works"])])  # Works of both pools
    rep = works[(works["year"] == year) & works["is_representative"] & works["artist_id"].isin(ids)][["artist_id", "work_id"]]  # representative Work per artist
    sc = pd.concat([pd.read_csv(ROOT / "results/demo/when_demo_scores.csv"), pd.read_csv(ROOT / pb["when_scores"])])  # stored scores
    sc = sc[sc["target"] == "next_1y"]  # WHEN next_1y
    return rep.merge(sc, on=["artist_id", "work_id"])  # score, label and flags


def check_default() -> None:  # test 1
    """Default view: mode 2, year 2018, the 8 prepared artists, the selection note."""
    at = new_app("placeholder")  # app, nothing pressed
    assert at.radio(key=g["keys"]["mode"]).value == g["default_mode"], "mode 2 must open first"  # mode
    assert at.selectbox(key=k["year"]).value == rt["year"], "the default year"  # year
    assert rt["selection_note"] in [c.value for c in at.caption], "the selection note must be on screen"  # disclosure line
    shown = [a for a, _ in rows_of(at)]  # artists in the table
    assert sorted(shown) == sorted(names[i] for i in rt["artists"]), "the table must hold the prepared artists"  # prepared list
    assert shown == [names[i] for i in rt["artists"]], "placeholder stage keeps the selection order"  # no score-based order yet
    print(f"1 default: mode {at.radio(key=g['keys']['mode']).value}, year {rt['year']}, {len(shown)} artists in selection order, note on screen, no exception")  # result


def check_order_and_summary() -> None:  # tests 2 and 3
    """Rows sorted by WHEN score descending against the stored files; the summary equals an independent recount."""
    at = new_app()  # app
    ref = stored_when(rt["artists"], rt["year"])  # stored reference
    ref["name"] = ref["artist_id"].map(names)  # names
    want = ref.sort_values(["score", "name"], ascending=[False, True])["name"].tolist()  # independent order
    got = [a for a, _ in rows_of(at) if a in want]  # table order of the artists that have a row
    assert got == want, f"the order differs: {got} vs {want}"  # sorted by score descending
    ok = ref[ref["label"].notna() & ~ref["is_projection"] & ~ref["label_after_holdout"]]  # counted rows
    k_when, n_when = int(((ok["score"] >= g["threshold"]) == (ok["label"] == 1)).sum()), len(ok)  # independent WHEN count
    text = table_html(at)  # table HTML
    assert tx["summary_when"].format(k=k_when, n=n_when) in html.unescape(text), "the WHEN summary differs from the recount"  # summary line
    rows = multi_rows(cfg, rt["artists"], rt["year"], g["threshold"], load_tables()["what_top10"]["score_cutoff"], rt["max_artists"])  # rows from the logic
    labs = pd.concat([pd.read_csv(ROOT / "results/demo/live/what_labels.csv"), pd.read_csv(ROOT / pb["what_labels"])])  # stored WHAT labels
    lab = labs[(labs["year"] == rt["year"]) & labs["artist_id"].isin(rt["artists"]) & labs["label"].notna() & ~labs["label_after_holdout"]]  # counted WHAT labels
    right = 0  # right WHAT verdicts
    for _, r in lab.iterrows():  # every counted WHAT label
        row = rows[rows["artist_id"] == r["artist_id"]].iloc[0]  # same artist in the table rows
        right += int(row["what_verdict"] == ("yes" if r["label"] == 1 else "no"))  # verdict against the stored label
    s = multi_summary(rows)  # summary from the logic
    assert (s["what_right"], s["what_n"]) == (right, len(lab)), "the WHAT summary differs from the recount"  # WHAT count
    print(f"2 order equals the stored-file order ({len(want)} rows with a score); 3 summary equals the recount: WHEN {k_when} of {n_when}, WHAT {right} of {len(lab)}")  # result


def check_boygenius() -> None:  # test 4
    """boygenius shows the missing-score text over two columns and stays out of the WHAT count."""
    at = new_app()  # app
    row = dict(rows_of(at))["boygenius"]  # its row
    want = html.escape(tx["what_missing"].format(n=CONFIG["what"]["min_previous_works"]))  # expected text
    assert "colspan='2'" in row and want in row, "the WHAT cells must be one colspan text"  # colspan text
    assert all(r["artist_id"] != by_name["boygenius"] for _, r in multi_rows(cfg, rt["artists"], rt["year"], g["threshold"], 0.4943, 8).iterrows() if r.get("what_available") is True), "boygenius has no WHAT score"  # not available
    print("4 boygenius: WHAT cells are one colspan=2 text:", html.unescape(want))  # result


def check_years() -> None:  # test 5
    """2019 has a row without Work; 2024 has asterisk marks and is uncounted; 2026 has no reality."""
    at = new_app()  # app
    at.selectbox(key=k["year"]).select(2019).run()  # 2019
    go(at, "reveal")  # the state reset, press again
    assert not at.exception  # no exception
    assert tx["no_work"] in dict(rows_of(at))["boygenius"], "boygenius has no Work in 2019"  # gap row
    at.selectbox(key=k["year"]).select(2024).run()  # 2024
    go(at, "reveal")  # the state reset, press again
    h24 = table_html(at)  # table in 2024
    marks = re.findall(r"<span class='poc-mark (\w+)'>", h24)  # mark colors
    glyphs = re.findall(r"<span class='poc-mark (\w+)'>(.*?)</span>", h24)  # every mark with its glyph
    assert glyphs and "amber" not in h24 and all((c == "grey" and g == "●") or (c in ("green", "red") and g in ("✓*", "✗*") and (c == "green") == (g == "✓*")) for c, g in glyphs), f"2024 marks must be green or red with the asterisk, or the grey dot: {glyphs}"  # no amber, asterisk and color match
    assert tx["summary_when"].format(k=0, n=0) in html.unescape(h24), "2024 rows are not counted"  # not counted
    at.selectbox(key=k["year"]).select(2026).run()  # 2026
    go(at, "reveal")  # the state reset, press again
    h26 = table_html(at)  # table in 2026
    assert tx["no_reality"] in h26 and set(re.findall(r"<span class='poc-mark (\w+)'>", h26)) <= {"grey"}, "2026 has no reality, grey marks only"  # last data year
    print(f"5 years: 2019 gap row, 2024 marks {sorted(set(glyphs))} and 0 counted, 2026 no reality with grey marks, no exception")  # result


def check_manual_and_cap() -> None:  # test 6
    """Genre filter and manual picks from both pools; the cap is the widget limit and the function limit."""
    at = new_app()  # app
    at.selectbox(key=k["genre"]).select("Country").run()  # genre filter
    ms = at.multiselect(key=k["artists"])  # picker
    ms.unselect(by_name["BTS"]).unselect(by_name["BLACKPINK"]).run()  # make room: the widget refuses more than the cap
    ms = at.multiselect(key=k["artists"])  # picker after the rerun
    ms.select(by_name["Luke Combs"]).select(by_name["Kacey Musgraves"]).run()  # one artist of each pool
    assert not at.exception  # no exception
    go(at, "reveal")  # the state reset after the artist change, press again
    shown = [a for a, _ in rows_of(at)]  # table artists
    assert "Luke Combs" in shown and "Kacey Musgraves" in shown, "manual picks from both pools must be in the table"  # both pools
    assert ms.proto.max_selections == rt["max_artists"], "the widget must carry the cap"  # widget cap
    big = multi_rows(cfg, [by_name[n] for n in list(by_name)[:rt["max_artists"] + 3]], 2018, g["threshold"], 0.4943, rt["max_artists"])  # more ids than the cap
    assert len(big) <= rt["max_artists"], "the function must cap the artists"  # function cap
    print(f"6 manual picks: Luke Combs (A) and Kacey Musgraves (B) added with genre Country, widget cap {ms.proto.max_selections}, function cap {len(big)} rows for {rt['max_artists'] + 3} ids")  # result


def check_switch() -> None:  # test 7
    """Mode 1 history survives a switch to mode 2 and back, and mode 2 creates no game state."""
    at = new_app()  # app
    assert g["keys"]["state"] not in at.session_state or not at.session_state[g["keys"]["state"]], "mode 2 must not create game state"  # no state
    at.radio(key=g["keys"]["mode"]).set_value("one").run()  # mode 1
    at.selectbox(key=g["keys"]["genre"]).select("Pop").run()  # genre
    opts = sorted((a["name"] for a in all_artists(cfg) if a["genre"] == "Pop"), key=str.lower)  # Pop artists
    at.selectbox(key=g["keys"]["artist"]).select_index(opts.index("Taylor Swift")).run()  # artist
    aid = by_name["Taylor Swift"]  # id
    at.button(key=g["keys"]["reveal"].format(id=aid)).click().run()  # reveal one year
    before = list(at.session_state[g["keys"]["state"]][aid]["history"])  # history
    at.radio(key=g["keys"]["mode"]).set_value("many").run()  # mode 2
    assert not at.exception  # no exception
    at.radio(key=g["keys"]["mode"]).set_value("one").run()  # mode 1 again
    after = list(at.session_state[g["keys"]["state"]][aid]["history"])  # history
    assert before == after and len(before) == 1, "the mode 1 history must survive the switch"  # kept
    print(f"7 switch: mode 1 history kept after a round trip ({len(after)} year), mode 2 created no state, no exception")  # result


def cells(row: str) -> list:  # text of the cells of a row
    """Text of every cell of a row HTML, unescaped."""
    return [html.unescape(re.sub(r"<[^>]+>", "", c)) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row)]  # cell texts


def check_stages() -> None:  # test 8
    """Placeholders before Predict, model columns after it with no reality, everything after Show reality."""
    ph = tx["placeholder"]  # placeholder text
    at = new_app("placeholder")  # nothing pressed
    h0 = table_html(at)  # table
    assert at.button(key=k["predict"]) and not any(b.key == k["reveal"] for b in at.button), "only Predict is offered first"  # buttons
    assert ph in h0 and "Yes" not in h0 and "Likely" not in h0 and "poc-mark" not in h0, "no verdict or mark before Predict"  # placeholder only
    assert "class='poc-note'" not in h0 and h0.count("class='poc-ph'") == 5 * len(rt["artists"]) and "colspan='5'" not in h0, "five equal placeholder cells per row and no note before Predict"  # placeholders
    assert not any(e.label == tx["how_label"] for e in at.expander), "no How to read this expander before Predict"  # expander absent
    at = go(at, "predict")  # Predict
    h1 = table_html(at)  # table
    assert at.button(key=k["reveal"]) and not any(b.key == k["predict"] for b in at.button), "only Show reality is offered after Predict"  # buttons
    rows1 = dict(rows_of(at))  # rows
    assert "Yes" in h1 or "No" in h1, "model verdicts must show after Predict"  # model columns
    assert "poc-mark" not in h1 and tx["summary_when"].split("{")[0] not in html.unescape(h1), "no mark and no summary before Show reality"  # no reality yet
    assert html.escape(tx["what_missing"].format(n=CONFIG["what"]["min_previous_works"])) in rows1["boygenius"], "the WHAT missing text shows after Predict"  # missing text
    how = next(e for e in at.expander if e.label == tx["how_label"])  # the expander after Predict
    caps = [c.value for c in how.caption]  # its lines
    assert len(caps) == 3 and cfg["poc"]["texts"]["legend_reference"] in caps and sum("in 10" in c for c in caps) == 2, "the two reliability lines and the legend are in the expander"  # contents
    assert "class='poc-note'" not in h1 and "poc-pill" in h1, "no note under the table after Predict, verdicts are pills"  # notes location
    assert ph in rows1["Kaytranada"], "realities stay placeholders after Predict"  # placeholder reality
    at = go(at, "reveal")  # Show reality
    h2 = table_html(at)  # table
    assert not any(b.key in (k["predict"], k["reveal"]) for b in at.button), "no step button after the reveal"  # buttons gone
    assert "poc-mark" in h2 and tx["summary_when"].split("{")[0] in html.unescape(h2) and tx["summary_what"].split("{")[0] in html.unescape(h2), "marks and both summary lines after Show reality"  # full
    assert h2.count("class='poc-note'") == 2 and cfg["poc"]["texts"]["legend_reference"] not in html.unescape(h2), "only the two summary lines under the table, the legend is in the expander"  # notes location
    assert cfg["poc"]["texts"]["legend_reference"] in [c.value for e in at.expander if e.label == tx["how_label"] for c in e.caption], "the legend stays in the expander after Show reality"  # legend
    print(f"8 stages: {h0.count(chr(39) + 'poc-ph' + chr(39))} placeholder cells -> model pills, expander with {len(caps)} lines, realities still '{ph}' -> marks and 2 summary lines under the table")  # result


def check_structure() -> None:  # test 10
    """Controls inside a collapsed expander, primary buttons, pills without green or red, and a style block with only .poc-* rules."""
    at = new_app("placeholder")  # nothing pressed
    exp = next(e for e in at.expander if e.label == tx["change_label"])  # the controls expander
    assert not exp.proto.expanded, "the controls expander must be collapsed by default"  # collapsed
    assert {s.key for s in exp.selectbox} == {k["year"], k["genre"]} and {m.key for m in exp.multiselect} == {k["artists"]}, "year, genre and artists are inside the expander"  # contents
    assert at.button(key=k["predict"]).proto.type == "primary", "Predict is a primary button"  # button type
    assert rt["selection_note"] in [c.value for c in at.caption if c not in exp.caption], "the selection note stays visible above the table"  # note outside
    at = go(at, "predict")  # Predict
    assert at.button(key=k["reveal"]).proto.type == "primary", "Show reality is a primary button"  # button type
    full = next(h.value for h in at.get("html") if "poc-table" in h.value)  # table element with its style block
    pills = re.findall(r"<span class='poc-pill (\w+)'>", full)  # pill classes
    assert pills and set(pills) <= {"yes", "no"}, f"pills must be yes or no only: {set(pills)}"  # no green or red
    styles = "".join(re.findall(r"<style>(.*?)</style>", full, flags=re.S))  # every style block of the element
    rules = [r.split("{")[0] for r in re.findall(r"[^{}]+\{[^{}]*\}", styles)]  # selectors of the style blocks
    assert rules and all(sel.startswith(".poc-") for sel in rules), f"a rule outside .poc-* exists: {[s for s in rules if not s.startswith('.poc-')]}"  # scoping
    assert all(sel.startswith(".poc-multi") for sel in rules if "poc-pill" in sel or "poc-ph" in sel), "pill and placeholder rules are scoped to the wrapper"  # wrapper scope
    print(f"10 structure: controls expander collapsed with year, genre and artists, selection note above, both buttons primary, pills {sorted(set(pills))}, {len(rules)} style rules all .poc-*")  # result


def check_reset() -> None:  # test 9
    """The state resets on a year change and on an artist change; a change in mode 1 does not touch it."""
    at = new_app("reveal")  # full table
    at.selectbox(key=k["year"]).select(2019).run()  # year change
    assert at.session_state[k["stage"]]["stage"] == "placeholder" and "poc-mark" not in table_html(at), "a year change must reset to placeholders"  # reset
    at = go(at, "reveal")  # full again
    at.multiselect(key=k["artists"]).unselect(by_name["BTS"]).run()  # artist change
    assert at.session_state[k["stage"]]["stage"] == "placeholder", "an artist change must reset to placeholders"  # reset
    at = go(at, "predict")  # predict stage
    at.radio(key=g["keys"]["mode"]).set_value("one").run()  # mode 1
    at.selectbox(key=g["keys"]["genre"]).select("Pop").run()  # a change in mode 1
    assert at.session_state[k["stage"]]["stage"] == "predict", "a change in mode 1 must not touch the mode 2 state"  # kept
    at.radio(key=g["keys"]["mode"]).set_value("many").run()  # back to mode 2
    assert at.session_state[k["stage"]]["stage"] == "predict" and not at.exception  # still predict
    print("9 reset: year change and artist change return to placeholders, mode 1 changes leave the mode 2 stage (predict) untouched")  # result


for fn in (check_default, check_order_and_summary, check_boygenius, check_years, check_manual_and_cap, check_switch, check_stages, check_reset, check_structure):  # all checks
    t = time.time()  # clock
    fn()  # run the check
    print(f"   {fn.__name__} {time.time() - t:.2f} s")  # timing
print(f"\ntotal time {time.time() - T0:.2f} s")  # total
