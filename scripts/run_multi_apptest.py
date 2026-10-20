"""AppTest run of mode 2 of tab 6 (several artists in one year, Work rows and year-end rows). Prints only, writes no files.

Usage:
    venv/bin/python3 scripts/run_multi_apptest.py
"""

import html  # unescape of table text
import logging  # capture of the Streamlit widget warning
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
import demo_multi  # module, to stub the counts in one check
from demo_multi import hidden_counts, hidden_lines, multi_rows, multi_summary  # logic under test

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


def artist_cells(at: AppTest) -> dict:  # artist cells of the table
    """Artist name -> inner HTML of its artist cell (the name and, for a year-end row, the grey tag)."""
    return {html.unescape(re.sub(r"<[^>]+>", "", a.split("<div")[0])): a for a, _ in re.findall(r"<tr><th>(.*?)</th>(.*?)</tr>", table_html(at)) if not a.startswith(tx["col_artist"])}  # name before the tag div


def rows_of(at: AppTest) -> list:  # parsed rows
    """(artist, row HTML) of every table row, in display order; the artist is the name only, without the year-end tag."""
    return [(html.unescape(a.split("<div")[0]), r) for a, r in re.findall(r"<tr><th>(.*?)</th>(.*?)</tr>", table_html(at)) if not a.startswith(tx["col_artist"])]  # artist cells and the rest, header row left out


def tag_text(year: int) -> str:  # escaped year-end tag
    """The year-end tag of a year as it is written in the HTML."""
    return html.escape(tx["year_end_tag"].format(year=year))  # text from the yaml


def stored_when(ids: list, year: int, target: str = "next_1y") -> pd.DataFrame:  # independent WHEN reference from the stored files
    """Stored WHEN score, label and flags of each artist-year: the representative Work when the artist has a Work in the year, else the year-end row. Read from the csv files only (no app function)."""
    works = pd.concat([pd.read_csv(ROOT / "results/demo/works_demo.csv"), pd.read_csv(ROOT / pb["works"])])  # Works of both pools
    rep = works[(works["year"] == year) & works["is_representative"] & works["artist_id"].isin(ids)][["artist_id", "work_id"]]  # representative Work per artist
    sc = pd.concat([pd.read_csv(ROOT / "results/demo/when_demo_scores.csv"), pd.read_csv(ROOT / pb["when_scores"])])  # stored scores
    work_rows = rep.merge(sc[sc["target"] == target], on=["artist_id", "work_id"]).assign(is_year_end=False)  # Work rows
    ye = pd.read_csv(ROOT / "results/demo/when_demo_scores_year_end.csv")  # stored year-end scores
    ye = ye[(ye["snapshot_year"] == year) & ye["artist_id"].isin(ids) & (ye["target"] == target) & ~ye["artist_id"].isin(work_rows["artist_id"])].assign(is_year_end=True)  # year-end rows of artists without a Work row
    return pd.concat([work_rows, ye], ignore_index=True)  # one row per artist with a row


def cascade_reference(ids: list, year: int) -> pd.DataFrame:  # both horizons side by side
    """One row per artist with a row: both scores, labels, counted flags and whether the 2-year pill is shown, from the stored files."""
    a, b = stored_when(ids, year, "next_1y"), stored_when(ids, year, "next_2y")  # both horizons
    m = a.merge(b, on="artist_id", suffixes=("", "_2"))  # side by side
    m["name"] = m["artist_id"].map(names)  # names
    m["v1"], m["v2"] = m["score"] >= g["threshold"], m["score_2"] >= g["threshold"]  # Yes verdicts
    m["show2"] = ~m["v1"] | ~m["v2"]  # second pill when next year is No or the 2-year verdict is No
    m["c1"] = m["label"].notna() & ~m["is_projection"] & ~m["label_after_holdout"]  # next-year reality counts
    m["c2"] = m["label_2"].notna() & ~m["is_projection_2"] & ~m["label_after_holdout_2"]  # 2-year reality counts
    return m  # reference


def check_default() -> None:  # test 1
    """Default view: mode 2, year 2019, the 10 prepared artists, the selection note."""
    at = new_app("placeholder")  # app, nothing pressed
    assert at.radio(key=g["keys"]["mode"]).value == g["default_mode"], "mode 2 must open first"  # mode
    assert at.selectbox(key=k["year"]).value == rt["year"], "the default year"  # year
    assert rt["selection_note"] in [c.value for c in at.caption], "the selection note must be on screen"  # disclosure line
    shown = [a for a, _ in rows_of(at)]  # artists in the table
    assert sorted(shown) == sorted(names[i] for i in rt["artists"]), "the table must hold the prepared artists"  # prepared list
    assert shown == [names[i] for i in rt["artists"]], "placeholder stage keeps the selection order"  # no score-based order yet
    print(f"1 default: mode {at.radio(key=g['keys']['mode']).value}, year {rt['year']}, {len(shown)} artists in selection order, note on screen, no exception")  # result


def check_order_and_summary() -> None:  # tests 2 and 3
    """Rows sorted by the next-year score descending against the stored files (Work rows and year-end rows); the summary equals an independent recount."""
    at = new_app()  # app
    ref = cascade_reference(rt["artists"], rt["year"])  # stored reference
    want = ref.sort_values(["score", "name"], ascending=[False, True])["name"].tolist()  # independent order
    got = [a for a, _ in rows_of(at)]  # table order
    assert got == want, f"the order differs: {got} vs {want}"  # sorted by score descending
    k1, n1 = int(((ref["v1"] == (ref["label"] == 1))[ref["c1"]]).sum()), int(ref["c1"].sum())  # independent next-year count
    sel = ref["show2"] & ref["c2"]  # rows where the 2-year pill is shown and the reality counts
    k2, n2 = int(((ref["v2"] == (ref["label_2"] == 1))[sel]).sum()), int(sel.sum())  # independent 2-year count
    text = html.unescape(table_html(at))  # table HTML
    top = load_tables()["what_top10"]["score_cutoff"]  # WHAT cutoff
    rows = multi_rows(cfg, rt["artists"], rt["year"], g["threshold"], top, rt["max_artists"])  # rows from the logic
    labs = pd.concat([pd.read_csv(ROOT / "results/demo/live/what_labels.csv"), pd.read_csv(ROOT / pb["what_labels"])])  # stored WHAT labels
    work_ids = ref.loc[~ref["is_year_end"], "artist_id"]  # artists with a Work row: only they have a WHAT score in the table
    lab = labs[(labs["year"] == rt["year"]) & labs["artist_id"].isin(work_ids) & labs["label"].notna() & ~labs["label_after_holdout"]]  # counted WHAT labels
    right, n_what = 0, 0  # right WHAT verdicts and rows with a WHAT score
    for _, r in lab.iterrows():  # every counted WHAT label
        row = rows[rows["artist_id"] == r["artist_id"]].iloc[0]  # same artist in the table rows
        if not row["what_available"]:  # no WHAT score for that artist-year
            continue  # not counted
        n_what += 1  # a counted WHAT row
        right += int(row["what_verdict"] == ("yes" if r["label"] == 1 else "no"))  # verdict against the stored label
    s = multi_summary(rows)  # summary from the logic
    assert (s["when_right"], s["when_n"], s["when2_right"], s["when2_n"]) == (k1, n1, k2, n2), "the WHEN summary differs from the recount"  # WHEN counts
    assert tx["summary_when"].format(k=k1, n=n1, k2=k2, n2=n2) in text, "the WHEN summary line differs from the recount"  # summary line
    assert (s["what_right"], s["what_n"]) == (right, n_what), "the WHAT summary differs from the recount"  # WHAT count
    print(f"2 order equals the stored-file order ({len(want)} rows); 3 summary equals the recount: next year {k1} of {n1}, within 2 years {k2} of {n2}, WHAT {right} of {n_what}")  # result


def check_boygenius() -> None:  # test 4
    """boygenius has a year-end row in 2019 and a Work row without a WHAT score in 2018; both show their WHAT text over two columns and stay out of the WHAT count."""
    at = new_app()  # app
    row = dict(rows_of(at))["boygenius"]  # its 2019 row
    want = html.escape(tx["what_year_end"])  # expected text of a year-end row
    assert "colspan='2'" in row and want in row, "the year-end WHAT cells must be one colspan text"  # colspan text
    at.selectbox(key=k["year"]).select(2018).run()  # 2018
    go(at, "reveal")  # the state reset, press again
    row18 = dict(rows_of(at))["boygenius"]  # its 2018 row
    want18 = html.escape(tx["what_missing"].format(n=CONFIG["what"]["min_previous_works"]))  # expected text of a Work row without a WHAT score
    assert "colspan='2'" in row18 and want18 in row18 and "poc-sub" not in dict(artist_cells(at))["boygenius"], "the Work row must show the not-eligible WHAT text and no year-end tag"  # colspan text
    top = load_tables()["what_top10"]["score_cutoff"]  # WHAT cutoff
    for year in (2018, 2019):  # both years
        rows = multi_rows(cfg, rt["artists"], year, g["threshold"], top, rt["max_artists"])  # rows from the logic
        assert not rows[rows["artist"] == "boygenius"]["what_available"].iloc[0], f"boygenius has no WHAT score in {year}"  # not available
    print("4 boygenius: 2019 year-end row and 2018 Work row both show one colspan=2 WHAT text:", html.unescape(want), "|", html.unescape(want18))  # result


def check_years() -> None:  # test 5
    """2019 has a row without Work; 2024 has asterisk marks and is uncounted; 2026 has no reality."""
    at = new_app()  # app
    at.selectbox(key=k["year"]).select(2019).run()  # 2019
    go(at, "reveal")  # the state reset, press again
    assert not at.exception  # no exception
    boy = artist_cells(at)["boygenius"]  # boygenius artist cell in 2019
    assert tag_text(2019) in boy, "boygenius has no Work in 2019, its year-end row is shown"  # year-end row
    at.selectbox(key=k["year"]).select(2024).run()  # 2024
    go(at, "reveal")  # the state reset, press again
    h24 = table_html(at)  # table in 2024
    marks = re.findall(r"<span class='poc-mark (\w+)'>", h24)  # mark colors
    glyphs = re.findall(r"<span class='poc-mark (\w+)'>(.*?)</span>", h24)  # every mark with its glyph
    assert glyphs and "amber" not in h24 and all((c == "grey" and g == "●") or (c in ("green", "red") and g in ("✓*", "✗*") and (c == "green") == (g == "✓*")) for c, g in glyphs), f"2024 marks must be green or red with the asterisk, or the grey dot: {glyphs}"  # no amber, asterisk and color match
    assert tx["summary_when"].format(k=0, n=0, k2=0, n2=0) in html.unescape(h24), "2024 rows are not counted"  # not counted
    at.selectbox(key=k["year"]).select(2026).run()  # 2026
    go(at, "reveal")  # the state reset, press again
    h26 = table_html(at)  # table in 2026
    assert tx["no_reality"] in h26 and set(re.findall(r"<span class='poc-mark (\w+)'>", h26)) <= {"grey"}, "2026 has no reality, grey marks only"  # last data year
    print(f"5 years: 2019 year-end row, 2024 marks {sorted(set(glyphs))} and 0 counted, 2026 no reality with grey marks, no exception")  # result


def check_manual_and_cap() -> None:  # test 6
    """Genre filter and manual picks from both pools; the cap is the widget limit and the function limit."""
    at = new_app()  # app
    at.selectbox(key=k["genre"]).select("Country").run()  # genre filter
    ms = at.multiselect(key=k["artists"])  # picker
    ms.unselect(by_name["BTS"]).run()  # make room: the widget refuses more than the cap
    ms = at.multiselect(key=k["artists"])  # picker after the rerun
    ms.select(by_name["Luke Combs"]).run()  # a pool A artist next to the prepared pool B artist Kacey Musgraves
    assert not at.exception  # no exception
    go(at, "reveal")  # the state reset after the artist change, press again
    shown = [a for a, _ in rows_of(at)]  # table artists
    assert "Luke Combs" in shown and "Kacey Musgraves" in shown, "manual picks from both pools must be in the table"  # both pools
    assert ms.proto.max_selections == rt["max_artists"], "the widget must carry the cap"  # widget cap
    big = multi_rows(cfg, [by_name[n] for n in list(by_name)[:rt["max_artists"] + 3]], 2018, g["threshold"], 0.4943, rt["max_artists"])  # more ids than the cap
    assert len(big) <= rt["max_artists"], "the function must cap the artists"  # function cap
    print(f"6 manual picks: Luke Combs (A) added to Kacey Musgraves (B) with genre Country, widget cap {ms.proto.max_selections}, function cap {len(big)} rows for {rt['max_artists'] + 3} ids")  # result


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
    assert html.escape(tx["what_year_end"]) in rows1["boygenius"], "the year-end WHAT text shows after Predict"  # year-end text
    how = next(e for e in at.expander if e.label == tx["how_label"])  # the expander after Predict
    caps = [c.value for c in how.caption]  # its lines
    assert len(caps) == 4 and cfg["poc"]["texts"]["legend_reference"] in caps and tx["legend_cascade"] in caps and sum("in 10" in c for c in caps) == 2, "the two reliability lines, the legend and the cascade sentence are in the expander"  # contents
    assert "class='poc-note'" not in h1 and "poc-pill" in h1, "no note under the table after Predict, verdicts are pills"  # notes location
    assert ph in rows1["SZA"], "realities stay placeholders after Predict"  # placeholder reality
    at = go(at, "reveal")  # Show reality
    h2 = table_html(at)  # table
    assert not any(b.key in (k["predict"], k["reveal"]) for b in at.button), "no step button after the reveal"  # buttons gone
    assert "poc-mark" in h2 and tx["summary_when"].split("{")[0] in html.unescape(h2) and tx["summary_what"].split("{")[0] in html.unescape(h2), "marks and both summary lines after Show reality"  # full
    assert h2.count("class='poc-note'") == 2 and cfg["poc"]["texts"]["legend_reference"] not in html.unescape(h2), "only the two summary lines under the table, the legend is in the expander"  # notes location
    assert cfg["poc"]["texts"]["legend_reference"] in [c.value for e in at.expander if e.label == tx["how_label"] for c in e.caption], "the legend stays in the expander after Show reality"  # legend
    print(f"8 stages: {h0.count(chr(39) + 'poc-ph' + chr(39))} placeholder cells -> model pills, expander with {len(caps)} lines, realities still '{ph}' -> marks and 2 summary lines under the table")  # result


def check_year_end() -> None:  # test 11
    """Year-end rows: the tag, the WHAT text, the cascade pills only when the next year or the 2-year verdict is No, the marks count, the summary against route_table_year_end.csv, the hidden artists line."""
    at = new_app("predict")  # Predict stage: pills without realities
    ref = cascade_reference(rt["artists"], rt["year"])  # stored reference
    rows1 = dict(rows_of(at))  # rows after Predict
    cells_by_artist = artist_cells(at)  # artist cells after Predict
    for _, r in ref.iterrows():  # every artist with a row
        row = rows1[r["name"]]  # its row
        assert row.count(html.escape(tx["pill_two"])) == int(r["show2"]) and row.count(html.escape(tx["pill_next"])) == 1, f"{r['name']}: the 2-year pill must show only when the next year or the 2-year verdict is No"  # cascade rule
        cell = cells_by_artist[r["name"]]  # artist cell of the row
        assert (tag_text(rt["year"]) in cell) == bool(r["is_year_end"]) and ("poc-sub" in cell) == bool(r["is_year_end"]) and ((html.escape(tx["what_year_end"]) in row) == bool(r["is_year_end"])), f"{r['name']}: the year-end tag and the WHAT text belong to year-end rows only"  # tag under the artist and WHAT text
    at = go(at, "reveal")  # Show reality
    rows2 = dict(rows_of(at))  # rows after Show reality
    for _, r in ref.iterrows():  # every artist with a row
        row = rows2[r["name"]]  # its row
        n_marks = len(re.findall(r"<span class='poc-mark ", row))  # marks of the row
        assert n_marks == 1 + int(r["show2"]) + 1, f"{r['name']}: one mark per pill shown plus the WHAT mark, got {n_marks}"  # marks count
        assert row.count(html.escape(tx["pill_two"])) == 2 * int(r["show2"]), f"{r['name']}: the 2-year reality follows the 2-year pill"  # same structure in the reality cell
    table = pd.read_csv(ROOT / "results/demo/route_table_year_end.csv")  # extended route table
    t = table[(table["year"] == rt["year"]) & table["artist_id"].isin(rt["artists"])]  # the prepared artists in the route year
    show2 = (t["verdict_1y"] == "no") | (t["verdict_2y"] == "no")  # rows with the second pill
    k1, n1 = int((t["right_1y"] == True).sum()), int(t["right_1y"].notna().sum())  # next-year figures of the route table
    k2, n2 = int((t["right_2y"][show2] == True).sum()), int(t["right_2y"][show2].notna().sum())  # 2-year figures where shown
    assert (k1, n1, k2, n2) == (5, 10, 2, 4), f"the route table figures for 2019 changed: {(k1, n1, k2, n2)}"  # expected figures
    kw, nw = int((t["what_right"] == True).sum()), int(t["what_right"].notna().sum())  # WHAT figures of the route table
    assert tx["summary_when"].format(k=k1, n=n1, k2=k2, n2=n2) in html.unescape(table_html(at)), "the WHEN summary line must equal the route table figures"  # WHEN line
    assert tx["summary_what"].format(k=kw, n=nw) in html.unescape(table_html(at)), "the WHAT summary line must equal the route table figures"  # WHAT line
    assert not [c for c in at.caption if "earlier releases" in c.value], "no hidden-artists line when every selected artist has a row"  # no line
    at.selectbox(key=k["year"]).select(2018).run()  # 2018: two prepared-list artists would be hidden, check the line with the independent rule
    ids18 = [i for i in rt["artists"]]  # the prepared artists
    feats = pd.concat([pd.read_csv(ROOT / "results/demo/live/when_features.csv"), pd.read_csv(ROOT / pb["when_features"]), pd.read_csv(ROOT / "results/demo/live/when_features_year_end.csv")])  # every WHEN row of the files
    for year, extra in ((2018, ["Peso Pluma", "Rauw Alejandro"]), (2019, ["Peso Pluma"])):  # plural and singular
        has = set(feats[feats["snapshot_year"] == year]["artist_id"])  # artists with a row in the year
        want_hidden = [i for i in [by_name[n] for n in extra] if i not in has]  # selected artists without a row
        assert len(want_hidden) == len(extra) and hidden_counts([by_name[n] for n in extra], year, 99) == {"none": len(extra), "yet": 0}, "the hidden count must equal the files"  # independent count
        sel = ids18[:rt["max_artists"] - len(extra)] + [by_name[n] for n in extra]  # the prepared artists plus the hidden ones
        at = new_app("placeholder")  # fresh app
        at.selectbox(key=k["year"]).select(year).run()  # the year
        at.multiselect(key=k["artists"]).set_value(sel).run()  # the selection
        key = "no_history_one" if len(extra) == 1 else "no_history_many"  # wording variant
        assert tx[key].format(n=len(extra), year=year) in [c.value for c in at.caption], f"the hidden artists line is missing for {year}"  # line under the table
        assert all(n not in [a for a, _ in rows_of(at)] for n in extra), "hidden artists are not in the table"  # not shown
    print(f"11 year-end: tag and WHAT text on year-end rows only, 2-year pill only when next year or 2-year is No, marks = pills + WHAT, summary equals route_table_year_end.csv ({k1} of {n1}, {k2} of {n2}, WHAT {kw} of {nw}), hidden lines for 2018 (2, plural) and 2019 (1, singular)")  # result


def check_titles() -> None:  # test 13
    """Release titles in the reality cells (from release_pick.csv, read directly), cut at title_max_chars with the full title in the tooltip, no Release column, the year-end tag under the artist, scoreboard-relevant counts unchanged."""
    picks = pd.read_csv(ROOT / "results/demo/release_pick.csv")  # independent source of the titles
    ref = cascade_reference(rt["artists"], rt["year"])  # stored reference of the default view
    at = new_app()  # full table, default view
    head = re.findall(r"<th>(.*?)</th>", table_html(at).split("</tr>")[0])  # header cells
    assert head == [html.escape(tx[c]) for c in ("col_artist", "col_when", "col_when_reality", "col_what", "col_what_reality", "col_result")] and "col_release" not in tx and "Release" not in head, f"the Release column must be gone: {head}"  # header
    limit, shown_title, shown_cut = rt["title_max_chars"], 0, 0  # title limit and counters
    rows = dict(rows_of(at))  # rows by artist
    for _, r in ref.iterrows():  # every artist with a row
        row = rows[r["name"]]  # its row
        reality = re.findall(r"<td>(.*?)</td>", row)[1]  # the WHEN reality cell (second plain cell: after the model cascade)
        t1 = picks[(picks["artist_id"] == r["artist_id"]) & (picks["year"] == rt["year"] + 1)]  # pick of the next year
        if r["label"] == 1 and not t1.empty:  # next-year reality is Yes and a title exists
            full = t1.iloc[0]["title"]  # full title
            want = full if len(full) <= limit else full[:limit - 1].rstrip() + "\u2026"  # expected text
            assert html.escape("Yes: " + want) in reality, f"{r['name']}: the next-year reality cell must read 'Yes: {want}'"  # text with the title
            assert (f"title='{html.escape(full, quote=True)}'" in reality), f"{r['name']}: the full title must be in the tooltip"  # tooltip
            shown_title += 1  # counted
            shown_cut += int(len(full) > limit)  # cut titles
        elif r["label"] == 0:  # next-year reality is No
            assert "title=" not in reality.split("poc-step")[1], f"{r['name']}: a No reality has no title"  # the next-year item is the first step of the cell
    assert shown_title >= 1 and shown_cut >= 1, "the default view must show a title and a cut title (BTS)"  # both cases present in the data
    assert "\u2026" in re.sub(r"<[^>]+>", "|", rows["BTS"]) and "Map Of The Soul 7 ~ The Journey ~" in rows["BTS"], "the BTS title is cut with an ellipsis and keeps the full title in the tooltip"  # cut title
    cells_by_artist = artist_cells(at)  # artist cells
    ye_names = {r["name"] for _, r in ref.iterrows() if r["is_year_end"]}  # year-end rows
    assert ye_names and all(("poc-sub" in cells_by_artist[n]) == (n in ye_names) for n in cells_by_artist), "the tag sits under the artist of year-end rows only"  # tag placement
    assert all(tag_text(rt["year"]) in cells_by_artist[n] for n in ye_names), "the tag text is 'No release in {year} \u00b7 estimated from history'"  # tag text
    print(f"13 titles: {shown_title} reality cells with a title ({shown_cut} cut at {limit} with the full title in the tooltip), Release column gone, tag under the artist of {sorted(ye_names)} only")  # result


def widget_warnings(run) -> list:  # widget warnings logged while a function runs
    """Messages of the Streamlit session-state widget warning logged during `run()`."""
    records = []  # collected messages
    class Catch(logging.Handler):  # collects the records of one logger
        def emit(self, record):  # one record
            records.append(record.getMessage())  # keep the text
    log = logging.getLogger("streamlit.elements.lib.policies")  # logger of the widget policies
    handler = Catch()  # collector
    log.addHandler(handler)  # attach
    try:  # always detach
        run()  # the code under test
    finally:  # detach
        log.removeHandler(handler)  # back to the state before
    return [m for m in records if "Session State" in m]  # the widget warning only


def check_no_widget_warning() -> None:  # test 14
    """The route_year selectbox gets index only at its first draw: no session-state widget warning when the year changes. A positive control shows the capture sees the warning when index and a session value meet."""
    control = widget_warnings(lambda: AppTest.from_string("import streamlit as st\nst.session_state['w'] = 2\nst.selectbox('x', [1, 2, 3], index=1, key='w')").run())  # index plus a session value
    assert control, "the capture must see the warning in the positive control"  # the capture works
    def change_year():  # first draw and two year changes
        at = new_app("placeholder")  # first run
        at.selectbox(key=k["year"]).select(2020).run()  # year change
        at.selectbox(key=k["year"]).select(2021).run()  # second change
        assert at.selectbox(key=k["year"]).value == 2021 and not at.exception  # the value follows
    assert not widget_warnings(change_year), "the route_year widget must not warn"  # no warning
    print(f"14 widget warning: captured in the positive control ({len(control)}), none for route_year over a first draw and two year changes")  # result


def check_hidden_last_year() -> None:  # test 12
    """Last data year: artists with history but no row get the 'no release yet' line (singular and plural), never the 'no earlier releases' line; both lines show when both cases occur."""
    last = int(load_tables()["works"]["year"].max())  # last data year
    feats = pd.concat([pd.read_csv(ROOT / "results/demo/live/when_features.csv"), pd.read_csv(ROOT / pb["when_features"]), pd.read_csv(ROOT / "results/demo/live/when_features_year_end.csv")])  # every WHEN row of the files
    has = set(feats[feats["snapshot_year"] == last]["artist_id"])  # artists with a row in the last year
    works = pd.concat([pd.read_csv(ROOT / "results/demo/works_demo.csv"), pd.read_csv(ROOT / pb["works"])])  # Works of both pools
    early = set(works[works["year"] < last]["artist_id"])  # artists with a Work before the last year
    cand = [n for n in ("Billy Joel", "Coldplay", "Usher") if by_name[n] not in has and by_name[n] in early]  # selected names with history but no row
    assert len(cand) == 3, f"expected three artists with history and no {last} row: {cand}"  # test data present
    for names_sel in (cand[:1], cand):  # singular, then plural
        ids = [by_name[n] for n in names_sel]  # ids of the selection
        assert hidden_counts(ids, last, 99) == {"none": 0, "yet": len(ids)}, "the count must come from the data: history but no row"  # independent expectation
        at = new_app("placeholder")  # fresh app
        at.selectbox(key=k["year"]).select(last).run()  # last data year
        at.multiselect(key=k["artists"]).set_value(ids + [by_name["Kacey Musgraves"]]).run()  # hidden artists plus one with a row
        key = "no_release_one" if len(ids) == 1 else "no_release_many"  # wording variant
        caps = [c.value for c in at.caption]  # captions on the page
        assert tx[key].format(n=len(ids), year=last) in caps, f"the no-release line is missing for {len(ids)} artists"  # line under the table
        assert not [c for c in caps if "earlier releases" in c], "the 'no earlier releases' line must not appear for artists with history"  # false text absent
        assert all(n not in [a for a, _ in rows_of(at)] for n in names_sel), "hidden artists are not in the table"  # not shown
    real = demo_multi.hidden_counts  # the real function
    demo_multi.hidden_counts = lambda ids, year, cap: {"none": 2, "yet": 1}  # no table has both cases in the data, so the counts are stubbed
    try:  # always restore
        lines = hidden_lines(cfg, [1], last)  # lines for the stubbed counts
    finally:  # restore the real function
        demo_multi.hidden_counts = real  # back
    assert lines == [tx["no_history_many"].format(n=2, year=last), tx["no_release_one"].format(n=1, year=last)], f"both lines expected, none-case first: {lines}"  # two lines
    print(f"12 last year {last}: 'no release yet' line for 1 and 3 artists with history, no 'no earlier releases' line; mixed case (counts stubbed) gives the two lines")  # result


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
    at.selectbox(key=k["year"]).select(2020).run()  # year change
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


for fn in (check_default, check_order_and_summary, check_boygenius, check_years, check_manual_and_cap, check_switch, check_stages, check_reset, check_structure, check_year_end, check_titles, check_no_widget_warning, check_hidden_last_year):  # all checks
    t = time.time()  # clock
    fn()  # run the check
    print(f"   {fn.__name__} {time.time() - t:.2f} s")  # timing
print(f"\ntotal time {time.time() - T0:.2f} s")  # total
