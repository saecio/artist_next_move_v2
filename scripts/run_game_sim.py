"""Walk-through simulation of the tab 6 game for three artists in both modes. Prints only, writes no files.

Usage:
    venv/bin/python3 scripts/run_game_sim.py
"""

import sys  # module path
import time  # timings
from pathlib import Path  # project root

ROOT = Path(__file__).resolve().parent.parent  # project root, independent of the working directory
sys.path.append(str(ROOT / "src"))  # src/ importable

import pandas as pd  # tables

from app_data import load_demo_config  # demo settings
from app_live import artist_timeline, load_tables  # live data layer
from config import CONFIG  # project settings
from demo_game import hidden_counts, new_state, next_year, play, scored_years, scoreboard, session_total, year_cards, reliability, verdict, outside_text_key  # game functions

cfg = load_demo_config()["game"]  # game settings
CAP = load_demo_config()["poc"]["titles_per_card"]  # most titles drawn on one card
THRESHOLD = cfg["threshold"]  # model rule
ARTISTS = [7592689, 1124645, 4778]  # 0 train rows, hidden re-recordings, a regular artist
stored = load_tables()["when_stored"]  # stored scores and labels, independent source for the check
stored = stored[stored["target"] == "next_1y"]  # next_1y only


def walk(aid: int, user_on: bool) -> list:  # play every year of one artist
    """Play every year of one artist with a user who always answers yes."""
    tl = artist_timeline(aid)  # timeline
    years = scored_years(tl, CONFIG["years"]["holdout_start"])  # playable years
    state, year = new_state(years, user_on), years[0]  # state and first year
    while year is not None:  # until the last playable year
        row = tl[tl["year"] == year].iloc[0]  # representative Work
        state["history"].append(play(row, "yes" if user_on else None, THRESHOLD))  # question, step A and step B in one call
        year = next_year(years, year)  # next playable year
    return state["history"]  # played years


def expected(aid: int, history: list) -> pd.DataFrame:  # counted rows from the stored scores
    """Stored rows of the representative Works of the played years that must be counted."""
    tl = artist_timeline(aid)  # timeline, representative Work per year
    reps = tl[tl["year"].isin([h["year"] for h in history])][["year", "work_id"]].rename(columns={"year": "snapshot_year"})  # played representative Works
    s = stored[stored["artist_id"] == aid].merge(reps, on=["snapshot_year", "work_id"])  # stored rows of those Works
    return s[s["label"].notna() & (s["snapshot_year"] + 1 <= CONFIG["years"]["last_complete"])]  # counted rows: a label whose reality year is a complete year


t0 = time.time()  # clock
for user_on in (False, True):  # both modes
    print(f"\n===== mode: user prediction {'ON (always yes)' if user_on else 'OFF'} =====")  # print the result
    histories = []  # for the session total
    for aid in ARTISTS:  # three artists
        h = walk(aid, user_on)  # play the artist
        histories.append(h)  # keep for the total
        board = scoreboard(h)  # scoreboard of the game logic
        tl = artist_timeline(aid)  # timeline for the check
        rep = tl.set_index("year")  # by year
        ref = expected(aid, h)  # stored counted rows
        ref_rep = ref.set_index("snapshot_year")  # by year
        model_ok = int(sum(((rep.loc[y, "next_1y"] >= THRESHOLD) == (r["label"] == 1)) for y, r in ref_rep.iterrows()))  # model right, from live score
        stored_ok = int(sum(((r["score"] >= THRESHOLD) == (r["label"] == 1)) for _, r in ref.iterrows()))  # model right, from stored score
        user_ok = int((ref["label"] == 1).sum())  # always yes is right when the label is 1
        print(f"\nartist {aid}: train rows {tl['artist_train_rows'].iloc[0]}, played years {[x['year'] for x in h]}, counted {board['n']}")  # print the result
        print("scoreboard:", board)  # print the result
        print(f"check: counted {board['n']} vs stored {len(ref)}: {board['n'] == len(ref)}; model right {board['model_right']} vs live {model_ok} vs stored-score {stored_ok}: {board['model_right'] == model_ok == stored_ok}", end="")  # print the result
        print(f"; user right {board['user_right']} vs stored {user_ok}: {board['user_right'] == user_ok}" if user_on else "; user fields None: " + str(board["user_n"] is None and board["user_right"] is None and board["agree"] is None))  # print the result
        print("not counted:", [(x["year"], x["label"]) for x in h if not x["counted"]])  # print the result
    print("\nsession total:", session_total(histories))  # print the result

print("\n== 7592689 in 2021 (0 train rows)")  # print the result
row = artist_timeline(7592689).query("year == 2021").iloc[0]  # the requested year
print(play(row, "yes", THRESHOLD), "| title:", row["title"], "| next_1y", round(row["next_1y"], 4), "| WHAT:", row["stage0_score"], row["what_label"])  # print the result
print("\n== 1124645 hidden re-recordings per year:", hidden_counts(load_tables()["works"], 1124645))  # print the result
print("\n== card cap, 7592689 in 2021 and 1124645 in 2021")  # print the result
works = load_tables()["works"]  # Works of the demo artists
for aid in (7592689, 1124645):  # the requested case and a case with hidden re-recordings
    aw = works[works["artist_id"] == aid]  # this artist's Works
    in_year = aw[aw["year"] == 2021]  # file rows of the year
    for hide in (True, False):  # key on and off
        yc = year_cards(aw, 2021, hide, CAP)  # cards of the year
        visible = int((~in_year["is_rerecording"]).sum()) if hide else len(in_year)  # visible Works by direct count
        print(f"{aid} hide={hide}: Works in file {yc['total']} (direct {len(in_year)}), visible {visible}, cards drawn {len(yc['cards'])}, +{yc['more']} more, cap {CAP}")  # print the result
        assert yc["total"] == len(in_year) and len(yc["cards"]) == min(visible, CAP) and yc["more"] == max(visible - CAP, 0)  # cap and more count
print("cap checks passed")  # print the result
print("\n== uncounted years, 7592689 and 1124645 (reality year after the last complete year)")  # section title
for aid in (7592689, 1124645):  # the two requested artists
    for user_on in (False, True):  # both modes
        h = walk(aid, user_on)  # full walk
        for i, x in enumerate(h):  # every played year
            if x["year"] >= CONFIG["years"]["last_complete"]:  # reality in 2026 or later: not counted
                before = scoreboard(h[:i])  # board before the year
                after = scoreboard(h[:i + 1])  # board after the year
                print(f"{aid} user={user_on} {x['year']}: model {x['model']}, user {x['user']}, label {x['label']}, counted {x['counted']}, n {before['n']} -> {after['n']}, user_n {before['user_n']} -> {after['user_n']}")  # one line per year
                assert not x["counted"] and before == after, "an uncounted year changed the scoreboard"  # board unchanged
print("uncounted-year checks passed")  # all asserts held
print("\n== WHAT never moves the scoreboard")  # section title
for aid in (1124645, 7592689, 4778):  # three artists
    tl = artist_timeline(aid).set_index("year")  # timeline with the WHAT columns
    h = walk(aid, False)  # full walk
    for i, x in enumerate(h):  # every played year
        delta = scoreboard(h[: i + 1])["n"] - scoreboard(h[:i])["n"]  # change of the counted years
        row = tl.loc[x["year"]]  # timeline row of the year
        assert delta == int(x["counted"]), "the scoreboard change differs from the WHEN counted flag"  # WHEN only
        assert not any("what" in k for k in x), "a WHAT field entered the played-year record"  # no WHAT field in the record
        if x["year"] in (2020, 2023, 2024, 2026) or pd.isna(row["stage0_score"]):  # years of interest
            print(f"{aid} {x['year']}: WHAT score {'none' if pd.isna(row['stage0_score']) else round(row['stage0_score'], 3)}, WHAT label {row['what_label']}, flagged {row['what_label_after_holdout']}, WHEN counted {x['counted']}, delta n {delta}")  # one line per year
top = load_tables()["what_top10"]  # WHAT cutoff and holdout figures read from the file
print(f"\nWHAT cutoff {top['score_cutoff']:.4f}, reliability x and y {reliability(top, cfg["reliability_scale"])}, from the holdout file")  # numbers behind the reliability line
for aid, yr in ((1124645, 2020), (1124645, 2023), (7592689, 2023), (7592689, 2019), (1103159, 2021), (7592689, 2021), (1967268, 2023)):  # requested cases
    r = artist_timeline(aid).set_index("year").loc[yr]  # timeline row
    h = walk(aid, False)  # full walk
    i = [x["year"] for x in h].index(yr)  # position of the year
    delta = scoreboard(h[: i + 1])["n"] - scoreboard(h[:i])["n"]  # scoreboard change of that year
    verdict_what = "no WHAT score" if pd.isna(r["stage0_score"]) else ("Yes" if verdict(r["stage0_score"], top["score_cutoff"]) == "yes" else "No")  # WHAT verdict
    print(f"{aid} {yr}: WHAT verdict {verdict_what}, score {r['stage0_score']}, label {r['what_label']}, flagged {r['what_label_after_holdout']}, delta n {delta}, WHEN counted {h[i]['counted']}")  # one line per case
    assert delta == int(h[i]["counted"])  # WHAT adds nothing
import csv  # independent recomputation, no pandas
with open(ROOT / "results/holdout_what/top10_lift.csv", encoding="utf-8") as f:  # top 10% file
    cut = float(next(r for r in csv.DictReader(f) if r["model"] == "stage 0")["score_cutoff"])  # cutoff of the stage 0 row
with open(ROOT / "results/holdout_what/row_scores.csv", encoding="utf-8") as f:  # holdout scores
    pairs = [(float(r["stage0_score"]), float(r["label"])) for r in csv.DictReader(f)]  # score and label of every row
low = [lab for s, lab in pairs if s < cut]  # labels of the rows below the cutoff
z_independent = round(sum(low) / len(low) * cfg["reliability_scale"])  # rounded positive rate out of the scale
print(f"\nrows below the cutoff {len(low)} of {len(pairs)}, positives {int(sum(low))}, rate {sum(low) / len(low):.4f}, z {z_independent}")  # counts behind the No-verdict line
below = load_tables()["what_below"]  # the same numbers from the app layer
print(f"app layer: rows {below['rows']}, positives {below['positives']}, rate {below['rate']:.4f}, z {below['z']}")  # app-layer values
assert (below["rows"], below["positives"], below["z"]) == (len(low), int(sum(low)), z_independent)  # z equals the independent recomputation
y = reliability(top, cfg["reliability_scale"])[1]  # rounded positive rate of any artist
key = outside_text_key(below["z"], y, cfg["similar_gap"])  # wording variant
print(f"z {below['z']}, y {y}, variant applied: {key}")  # which variant applies to the real values
assert outside_text_key(3, 3, cfg["similar_gap"]) == "what_reliability_no_similar" and outside_text_key(3, 4, cfg["similar_gap"]) == "what_reliability_no_similar" and outside_text_key(3, 5, cfg["similar_gap"]) == "what_reliability_no_other"  # both variants work
print(cfg[key].format(z=below["z"], y=y, scale=cfg["reliability_scale"]))  # the No-verdict line
print(cfg["what_reliability"].format(x=reliability(top, cfg["reliability_scale"])[0], y=y, scale=cfg["reliability_scale"]))  # the Yes-verdict line
print("WHAT scoreboard checks passed")  # all asserts held
print("\n== pool A regression against the earlier runs")  # section title
pins = {7592689: (6, 3), 1124645: (7, 7), 4778: (7, 6)}  # counted years and model right under the rule by reality year (the earlier runs had 5/3, 6/6 and 6/6: 2024 now counts)
for aid, (n_old, right_old) in pins.items():  # three pool A artists
    b = scoreboard(walk(aid, False))  # scoreboard now
    print(f"{aid}: n {b['n']} (earlier {n_old}), model right {b['model_right']} (earlier {right_old})")  # one line per artist
    assert (b["n"], b["model_right"]) == (n_old, right_old)  # unchanged
w1, w2 = artist_timeline(1124645).set_index("year"), artist_timeline(7592689).set_index("year")  # timelines
assert (round(w1.loc[2020, "stage0_score"], 3), round(w2.loc[2023, "stage0_score"], 3)) == (0.445, 0.453)  # WHAT lines of the earlier runs
print("pool A unchanged: WHAT score 1124645 in 2020 and 7592689 in 2023 equal the earlier runs")  # proof
print("\n== pool B walks")  # section title
names_b = {int(a["discogs_id"]): a["name"] for a in load_demo_config()["artists_b"]}  # pool B names
for aid in (1778977, 52835, 13200183, 6774153):  # the four requested pool B artists
    tl = artist_timeline(aid)  # timeline
    playable = scored_years(tl, CONFIG["years"]["holdout_start"])  # playable years
    what_years = [int(y) for y in tl.loc[tl["stage0_score"].notna(), "year"]]  # years with a WHAT score
    for user_on in (False, True):  # both modes
        h = walk(aid, user_on)  # full walk
        b = scoreboard(h)  # scoreboard
        ref = expected(aid, h)  # stored counted rows
        model_ok = int(sum((((tl.set_index("year").loc[x["year"], "next_1y"] >= THRESHOLD) == (x["label"] == "yes")) for x in h if x["counted"])))  # model right from the live score
        print(f"{names_b[aid]} ({aid}) user={user_on}: playable {playable}, counted {b['n']} vs stored {len(ref)}, model right {b['model_right']} vs live {model_ok}, user_n {b['user_n']}")  # one line
        assert b["n"] == len(ref) and b["model_right"] == model_ok  # same as the stored labels
        assert [x["year"] for x in h] == playable  # every playable year was played
    print(f"   WHAT years {what_years}")  # WHAT coverage
    if aid == 6774153:  # boygenius
        assert not what_years and len(playable) == 3  # 3 playable years, no WHAT score
    if aid == 13200183:  # Peso Pluma
        assert what_years and min(what_years) >= 2023  # WHAT only from 2023
print("pool B walk checks passed")  # all asserts held
tot = session_total([walk(a, False) for a in (7592689, 1124645, 4778, 1778977, 52835, 13200183, 6774153)])  # session total over both pools
print("session total over pool A and pool B walks:", tot)  # both pools counted  # all asserts held
print("\n== mode 2 (several artists in one year)")  # section title
from demo_multi import multi_rows, multi_summary, route_ids, row_marks  # mode 2 logic
rcfg = load_demo_config()  # full settings
rrows = multi_rows(rcfg, route_ids(rcfg), rcfg["route"]["year"], cfg["threshold"], top["score_cutoff"], rcfg["route"]["max_artists"])  # prepared selection, route year
sc = rrows.loc[rrows["has_row"], "when_score"].tolist()  # scores in display order
assert sc == sorted(sc, reverse=True), "rows must be sorted by WHEN score descending"  # order
assert len(rrows) == len(rcfg["route"]["artists"]) <= rcfg["route"]["max_artists"]  # all prepared artists, within the cap
assert len(multi_rows(rcfg, route_ids(rcfg) * 2, 2018, cfg["threshold"], top["score_cutoff"], 5)) == 5  # the cap applies to the number of artists
s = multi_summary(rrows)  # summary
w = rrows[rrows["when_label"].notna() & rrows["when_counted"]]  # counted WHEN rows
assert (s["when_right"], s["when_n"]) == (int((w["when_verdict"] == w["when_label"]).sum()), len(w))  # WHEN count
w2 = rrows[rrows["show2"] & rrows["when2_label"].notna() & rrows["when2_counted"]]  # counted 2-year rows where the second pill is shown
assert (s["when2_right"], s["when2_n"]) == (int((w2["when2_verdict"] == w2["when2_label"]).sum()), len(w2))  # 2-year count
assert (rrows["show2"] == ((rrows["when_verdict"] == "no") | (rrows["when2_verdict"] == "no"))).all()  # cascade rule
assert all(len(row_marks(r)) == 2 + int(r["show2"]) for _, r in rrows.iterrows())  # one mark per pill shown plus WHAT
a = rrows[rrows["what_available"] & rrows["what_label"].notna() & rrows["what_counted"]]  # counted WHAT rows
assert (s["what_right"], s["what_n"]) == (int((a["what_verdict"] == a["what_label"]).sum()), len(a))  # WHAT count
r24 = multi_rows(rcfg, route_ids(rcfg), 2024, cfg["threshold"], top["score_cutoff"], 10)  # 2024: next-year reality in 2025 counts
assert multi_summary(r24)["when_n"] == int((r24["when_label"].notna() & r24["when_counted"]).sum()) > 0  # counted without asterisk
assert (r24["when_counted"] == r24["when_label"].notna()).all() and not r24["when_star"].any()  # reality year 2025: counted exactly when a label exists, no asterisk
assert (r24["when2_star"] == (r24["when2_label"].notna() & (r24["when_label"] != "yes"))).all()  # 2-year reality in 2026 (no release in 2025) carries the asterisk
r25 = multi_rows(rcfg, route_ids(rcfg), 2025, cfg["threshold"], top["score_cutoff"], 10)  # 2025: every reality is in 2026
assert multi_summary(r25)["when_n"] == 0 and multi_summary(r25)["what_n"] == 0 and not r25["when_counted"].any()  # not counted
assert (r25["when_star"] == r25["when_label"].notna()).all()  # asterisk exactly where a reality exists
assert not r24[r24["is_year_end"]]["what_star"].any() and not r25[r25["is_year_end"]]["what_star"].any()  # year-end rows never carry the WHAT asterisk
r26 = multi_rows(rcfg, route_ids(rcfg), 2026, cfg["threshold"], top["score_cutoff"], 10)  # 2026: forecast for 2027
assert r26["is_forecast"].all() and r26["when_label"].isna().all() and not (r26["when_counted"] | r26["when_star"] | r26["what_counted"] | r26["what_star"]).any()  # no reality, nothing counted, no asterisk
print(f"{rcfg['route']['year']} rows {len(rrows)} sorted by WHEN score, summary {s}, 2024 summary {multi_summary(r24)}, caps respected")  # result
from demo_multi import advance_stage, multi_signature  # two-step state machine
sig = multi_signature(2018, [1, 2])  # signature of a selection
s0 = advance_stage(None, sig, None)  # first run
s1 = advance_stage(s0, sig, "reveal")  # Show reality before Predict does nothing
s2 = advance_stage(s0, sig, "predict")  # Predict
s3 = advance_stage(s2, sig, "reveal")  # Show reality after Predict
assert (s0["stage"], s1["stage"], s2["stage"], s3["stage"]) == ("placeholder", "placeholder", "predict", "reveal")  # the three stages in order
assert advance_stage(s3, multi_signature(2019, [1, 2]), None)["stage"] == "placeholder" and advance_stage(s3, multi_signature(2018, [1]), None)["stage"] == "placeholder"  # a change of year or artists resets
assert advance_stage(s3, sig, "predict")["stage"] == "reveal"  # no step back
assert list(multi_rows(rcfg, route_ids(rcfg), rcfg["route"]["year"], cfg["threshold"], top["score_cutoff"], 10, "placeholder")["artist_id"]) == route_ids(rcfg)  # placeholder keeps the selection order
print("stage machine checks passed: placeholder -> predict -> reveal, reset on change, selection order before predict")  # result
print("mode 2 checks passed")  # all asserts held
print("\n== release titles (rule B), mode 1 years and cascade, scoreboard")  # section title
from demo_multi import play_year, playable_years, reality_titles  # mode 1 reuse of the mode 2 logic
picks = pd.read_csv(ROOT / "results/demo/release_pick.csv")  # the pick file
pool_b = load_demo_config()["pool_b"]["paths"]  # pool B file paths
allw = pd.concat([pd.read_csv(ROOT / CONFIG["paths"]["works_demo"]), pd.read_csv(ROOT / pool_b["works"])])  # Works of both pools
allw = allw[allw["year"] >= CONFIG["years"]["holdout_start"]]  # holdout start on
snap = pd.read_parquet(ROOT / CONFIG["paths"]["snapshots"], columns=["work_id", "is_album_like"], filters=[("work_id", "in", allw["work_id"].tolist())])  # album marker of the snapshots
album = dict(zip(snap["work_id"], snap["is_album_like"]))  # work_id -> album-like
want = {}  # independent rule B, plain loops
for (a, y), grp in allw[~allw["is_rerecording"]].groupby(["artist_id", "year"]):  # every artist-year with a non-re-recording Work
    ids = grp.sort_values("works_so_far_this_year")["work_id"].tolist()  # within-year order
    alb = [w for w in ids if album[w]]  # album-like Works in that order
    want[(a, y)] = (alb[0], "album") if alb else (ids[-1], "fallback")  # first album-like, else the representative (last in the order)
got = {(r.artist_id, r.year): (r.work_id, r.picked_by) for r in picks.itertuples()}  # the file
assert got == want, "release_pick.csv differs from an independent recomputation of rule B"  # same picks
assert set(picks["year"]) >= {2024, 2025, 2026}, "the file must cover the years after the holdout and 2024"  # coverage
print(f"rule B recomputed independently: {len(want)} artist-years, {picks['picked_by'].value_counts().to_dict()}")  # counts
assert reality_titles(picks, 4778, 2021, "yes", "yes")[0] == picks[(picks["artist_id"] == 4778) & (picks["year"] == 2022)]["title"].iloc[0]  # next year uses year + 1
route = pd.read_csv(ROOT / "results/demo/route_table_year_end.csv")  # route table with the year-end rows
route["counted_1y"] = route["reality_1y"].notna() & (route["year"] + 1 <= CONFIG["years"]["last_complete"])  # counting rule by reality year, derived here, the stored flag follows the old rule
rows_by = {}  # (artist id, year) -> mode 2 row
all_ids = sorted(route["artist_id"].unique().tolist())  # every artist of both pools
for yr in sorted(route["year"].unique()):  # one call per year, cap above the number of artists
    for _, mr in multi_rows(rcfg, all_ids, int(yr), cfg["threshold"], top["score_cutoff"], 100).iterrows():  # rows of the year
        rows_by[(int(mr["artist_id"]), int(yr))] = mr  # keep
title_of = {(r.artist_id, r.year): r.title for r in picks.itertuples()}  # (artist id, year) -> title, read from the file
n_t = 0  # checked reality titles
for r in route.itertuples():  # every route row, Work rows and year-end rows
    mr = rows_by[(r.artist_id, r.year)]  # same artist-year in the mode 2 rows
    assert bool(mr["is_year_end"]) == bool(r.is_year_end) and mr["when_verdict"] == r.verdict_1y, "mode 2 row differs from the route table"  # same rows and verdicts
    t = lambda y: title_of.get((r.artist_id, y))  # title of a year, None without a pick
    e1 = t(r.year + 1) if r.reality_1y == "yes" else None  # next-year title
    e2 = (t(r.year + 1) if r.reality_1y == "yes" else t(r.year + 2)) if r.reality_2y == "yes" else None  # title of the year of the release inside 2 years
    assert (None if pd.isna(mr["when_title"]) else mr["when_title"]) == e1 and (None if pd.isna(mr["when2_title"]) else mr["when2_title"]) == e2, f"{r.artist} {r.year}: reality titles differ"  # titles (a DataFrame turns None into NaN)
    n_t += int(e1 is not None) + int(e2 is not None)  # count
assert n_t > 0  # something was checked
print(f"reality titles of {len(route)} route rows equal the rule: {n_t} titles checked")  # result
n_w, n_y, n_ns = 0, 0, 0  # rows with a stored WHAT label, of them yes, year-end rows checked
for (aid_, yr_), mr in rows_by.items():  # every mode 2 row of both pools
    if mr["is_year_end"]:  # a year-end WHAT row is reference only, its new-styles row is the one of the last active year
        n_ns += 1  # count
        assert not mr["what_counted"] and not mr["what_star"], "a year-end WHAT row is never counted and has no asterisk"  # outside every count
        continue  # nothing else to compare
    if mr["what_available"] and isinstance(mr["what_label"], str):  # a stored WHAT label
        n_w += 1  # count
        assert (mr["what_label"] == "yes") == (len(mr["what_new"]) > 0), f"{aid_} {yr_}: the new styles must exist exactly when the label is yes"  # label rule at artist-year level
        assert mr["what_year"] is not None and int(mr["what_year"]) > yr_, f"{aid_} {yr_}: the label refers to a later release"  # next active year
        n_y += int(mr["what_label"] == "yes")  # count
assert n_w > 0 and n_y > 0, "no WHAT label was checked"  # something was compared
print(f"WHAT new styles: {n_w} stored labels agree with the new styles ({n_y} yes), {n_ns} year-end rows are reference only")  # result
board_checks = 0  # artists whose boards were checked
for aid in all_ids:  # every artist
    tl = artist_timeline(aid)  # timeline
    years = playable_years(aid, tl, CONFIG["years"]["holdout_start"])  # mode 1 years
    sub = route[route["artist_id"] == aid]  # route rows of the artist
    assert years == sorted(sub["year"].tolist()), f"{aid}: mode 1 years differ from the route table years"  # Work years and year-end years
    ye_years = set(sub[sub["is_year_end"]]["year"])  # year-end years
    h_all = [play_year(rows_by[(aid, y)], None) for y in years]  # history over every playable year
    h_work = [h for h in h_all if h["year"] not in ye_years]  # Work years only
    h_ye_unc = [h for h in h_all if h["year"] in set(sub[sub["is_year_end"] & ~sub["counted_1y"]]["year"])]  # year-end years whose reality is in 2026
    assert scoreboard(h_all)["n"] == int(sub["counted_1y"].sum()), f"{aid}: the scoreboard must follow counted_1y"  # counted flag of the route table
    assert scoreboard([h for h in h_all if h not in h_ye_unc]) == scoreboard(h_all), f"{aid}: year-end rows with a reality in 2026 changed the scoreboard"  # uncounted rows add nothing
    assert scoreboard(h_work)["n"] == int(sub[~sub["is_year_end"]]["counted_1y"].sum()), f"{aid}: the Work-year scoreboard changed"  # Work years as before
    board_checks += 1  # counted
print(f"mode 1 years equal the route table years and the scoreboard follows counted_1y for {board_checks} artists")  # result
for aid in ARTISTS:  # the three regression artists
    assert scoreboard([h for h in [play_year(rows_by[(aid, y)], None) for y in playable_years(aid, artist_timeline(aid), 2018)] if h["year"] in scored_years(artist_timeline(aid), 2018)]) == scoreboard(walk(aid, False)), f"{aid}: Work-year board differs from the earlier walk"  # unchanged by the year-end years
print("scoreboard of the Work years equals the earlier walk for the three regression artists")  # result

print(f"\nsimulation time {time.time() - t0:.2f} s")  # print the result
