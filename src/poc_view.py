"""HTML builders of the one-screen proof of concept: marks, year cards, window, prediction table, chips. Pure string functions, no Streamlit."""

import html  # escaping of titles and texts

import pandas as pd  # missing-value checks

from demo_game import card_status, flag, names, outside_text_key, reliability, verdict, year_cards  # game logic reused as it is


def css(poc: dict) -> str:  # style block from the config
    """Style block built from the colors and sizes of the poc section."""
    c, s = poc["colors"], poc["sizes"]  # colors and sizes
    rules = [  # one CSS rule per line
        f".poc-line{{display:flex;gap:{s['card_gap']}px;align-items:stretch}}",  # timeline of cards
        f".poc-card{{flex:1 1 {s['card_min_width']}px;min-width:{s['card_min_width']}px;border:1px solid {c['card_border']};border-radius:{s['card_radius']}px;padding:{s['card_padding']}px;font-size:{s['card_font']}px;line-height:1.25;position:relative}}",  # one card
        ".poc-year{font-weight:700}",  # year label, never truncated
        f".poc-title{{display:-webkit-box;-webkit-line-clamp:{s['title_lines']};-webkit-box-orient:vertical;overflow:hidden}}",  # titles on at most two lines
        f".poc-muted{{opacity:{s['muted_opacity']}}}",  # secondary text keeps the theme text color
        f".poc-pred{{border:{s['predicted_border']}px solid {c['card_border_predicted']};box-shadow:0 0 {s['predicted_shadow']}px {c['card_border_predicted']}}}",  # card of the predicted year
        f".poc-marks{{display:flex;gap:{s['card_gap']}px;margin-top:{s['card_gap']}px}}",  # row of the two marks
        f".poc-mark{{display:inline-flex;align-items:center;justify-content:center;min-width:{s['mark_size']}px;height:{s['mark_size']}px;box-sizing:border-box;padding:0 {s['mark_pad']}px;border-radius:{s['pill_radius']}px;font-size:{s['mark_font']}px;font-weight:{s['mark_weight']};line-height:1}}",  # one solid mark, round for one glyph, a pill for a glyph and an asterisk
        f".poc-earlier{{font-size:{s['note_font']}px;opacity:{s['muted_opacity']};margin-bottom:{s['card_gap']}px}}",  # marker of hidden earlier years
        f".poc-panel{{font-size:{s['table_font']}px;margin-top:{s['card_padding']}px}}",  # prediction panel
        ".poc-panel-title{font-weight:700;margin-bottom:4px}",  # panel title
        f".poc-table{{border-collapse:collapse;width:100%}}",  # table
        f".poc-table th,.poc-table td{{border-bottom:1px solid {c['card_border']};padding:{s['cell_padding']}px {s['card_padding']}px;text-align:left}}",  # cells
        f".poc-note{{font-size:{s['note_font']}px;opacity:{s['muted_opacity']};margin-top:2px}}",  # small notes under the table
        f".poc-chips{{display:flex;gap:{s['card_gap']}px;flex-wrap:wrap}}",  # scoreboard chips
        f".poc-chip{{background:{c['chip_bg']};color:{c['chip_fg']};border-radius:{s['card_radius']}px;padding:{s['cell_padding']}px {s['card_padding']}px;font-size:{s['chip_font']}px;font-weight:600}}",  # one chip
    ]  # close the rule list
    for kind in ("green", "red", "grey"):  # one rule per color
        rules.append(f".poc-cell.{kind}{{background:{c[kind]['bg']};color:{c[kind]['fg']};font-weight:600}}")  # result cell colors
        rules.append(f".poc-mark.{kind}{{background:{c['mark_' + kind]['bg']};color:{c['mark_' + kind]['fg']}}}")  # solid mark colors
    return "<style>" + "".join(rules) + "</style>"  # the style element


def badge_kind(done: bool, label: str | None, counted: bool, predicted: str) -> tuple:  # color and mark key of a prediction
    """Color and mark key: grey before the reveal or without reality, green or red otherwise, with an asterisk after the glyph when the result is shown for reference only (not counted)."""
    if not done or label is None:  # before the reveal, or no stored label
        return "grey", "mark_dot"  # grey dot
    if not counted:  # result shown for reference only: the real result, green or red, with an asterisk
        return ("green", "mark_ok_flagged") if predicted == label else ("red", "mark_bad_flagged")  # check or cross followed by an asterisk
    return ("green", "mark_ok") if predicted == label else ("red", "mark_bad")  # match or mismatch


def what_parts(row: pd.Series, top: dict) -> tuple:  # WHAT verdict, label and flag
    """WHAT verdict ('yes' or 'no', None without a score), stored label ('yes', 'no' or None) and whether the label is counted-safe (not flagged)."""
    if pd.isna(row["stage0_score"]):  # artist-year not eligible
        return None, None, True  # nothing to show
    label = None if pd.isna(row["what_label"]) else ("yes" if row["what_label"] == 1 else "no")  # stored WHAT label
    return verdict(row["stage0_score"], top["score_cutoff"]), label, not flag(row["what_label_after_holdout"])  # verdict, label, not flagged


def marks(row: pd.Series, played: dict | None, top: dict, threshold: float, last: bool) -> list:  # marks of the predicted year
    """The WHEN and WHAT marks of the predicted year as (title key, color, mark key)."""
    done = played is not None  # the reveal happened
    when = verdict(row["next_1y"], threshold)  # WHEN prediction
    if last and done:  # last data year
        out = [("mark_when_title", "grey", "mark_dot")]  # grey, no reality
    else:  # normal year
        kind, mark = badge_kind(done, played["label"] if done else None, played["counted"] if done else False, when)  # color and mark
        out = [("mark_when_title", kind, mark)]  # WHEN mark
    what, label, safe = what_parts(row, top)  # WHAT parts
    kind, mark = ("grey", "mark_dot") if what is None else badge_kind(done, label, safe, what)  # WHAT color and mark
    return out + [("mark_what_title", kind, mark)]  # both marks


def window_years(current: int, first: int, last_shown: int, before: int, after: int) -> tuple:  # years of the window
    """Years of the window and whether earlier years are cut; 0 for both counts means all years."""
    if before == 0 and after == 0:  # all years
        return list(range(first, last_shown + 1)), False  # no cut
    start = max(first, current - before)  # first year of the window
    return list(range(start, min(last_shown, current + after) + 1)), start > first  # window and the cut flag


def card_html(year: int, aw: pd.DataFrame, hidden: dict, until: int, pred_year: int, mark_list: list, g: dict, poc: dict) -> str:  # one year card
    """One year card: year, Work titles up to titles_per_card with a more line, hidden count, and the two marks on the predicted year."""
    yc = year_cards(aw, year, g["hide_rerecordings"], poc["titles_per_card"])  # Works drawn this year
    status = card_status(yc["total"], yc["cards"], year, until)  # future, gap, hidden or card
    body = [f"<div class='poc-year'>{year}</div>"]  # year first
    if status == "card":  # Works are shown
        for c in yc["cards"]:  # one block per Work
            body.append(f"<div class='poc-title' title='{html.escape(str(c['title']) + ' | ' + names(c['genres']) + ' | ' + names(c['styles']), quote=True)}'>{html.escape(str(c['title']))}</div>")  # title, full text on hover
        if yc["more"]:  # more Works than the cap
            body.append(f"<div class='poc-muted'>{html.escape(g['more_line'].format(n=yc['more']))}</div>")  # "+N more" line
    else:  # future, gap or only hidden re-recordings
        text = {"future": poc["texts"]["future_card"], "gap": poc["texts"]["gap_card"], "hidden": poc["texts"]["only_rerecordings"]}[status]  # state text
        body.append(f"<div class='poc-muted'>{html.escape(text)}</div>")  # state line
    if g["hide_rerecordings"] and year <= until and hidden.get(year):  # hidden re-recordings in this year
        body.append(f"<div class='poc-muted'>{html.escape(g['hidden_line'].format(n=hidden[year]))}</div>")  # count line
    if year == pred_year:  # the predicted year carries the marks
        t = poc["texts"]  # texts
        body.append("<div class='poc-marks'>" + "".join(f"<span class='poc-mark {k}' title='{html.escape(t[title])}'>{html.escape(t[m])}</span>" for title, k, m in mark_list) + "</div>")  # two marks, no text
    cls = "poc-card poc-pred" if year == pred_year else "poc-card"  # predicted card is highlighted
    return f"<div class='{cls}' data-year='{year}'>{''.join(body)}</div>"  # the card


def timeline_html(aw: pd.DataFrame, hidden: dict, window: tuple, until: int, pred_year: int, mark_list: list, g: dict, poc: dict) -> str:  # the timeline element
    """The style block, the earlier-years marker when needed, and the line of cards of the window."""
    years, cut = window  # years and cut flag
    earlier = f"<div class='poc-earlier'>{html.escape(poc['texts']['earlier_hidden'])}</div>" if cut else ""  # marker
    cards = [card_html(y, aw, hidden, until, pred_year, mark_list, g, poc) for y in years]  # one card per year
    return css(poc) + earlier + "<div class='poc-line'>" + "".join(cards) + "</div>"  # style, marker and line


def result_cell(done: bool, label: str | None, counted: bool, predicted: str, poc: dict, no_reality_key: str) -> tuple:  # result cell color and text
    """Result cell as (color, text): '?' before the reveal, a mark after it, a text when there is no reality."""
    t = poc["texts"]  # texts
    if not done:  # before the reveal
        return "grey", t["unknown"]  # unknown
    if label is None:  # no stored label
        return "grey", t[no_reality_key]  # no reality text
    kind, mark = badge_kind(done, label, counted, predicted)  # color and mark key
    return kind, t[mark]  # mark of the result


def table_rows(row: pd.Series, played: dict | None, top: dict, threshold: float, answer: str | None, g: dict, poc: dict) -> list:  # rows of the prediction table
    """Rows of the prediction table as dicts; the WHAT row carries a single text when there is no WHAT score."""
    t, done = poc["texts"], played is not None  # texts and the reveal flag
    yn = {"yes": g["answer_yes"], "no": g["answer_no"]}  # Yes / No words
    when = verdict(row["next_1y"], threshold)  # WHEN prediction
    label = played["label"] if done else None  # stored WHEN label
    counted = played["counted"] if done else False  # counted flag
    kind, text = result_cell(done, label, counted, when, poc, "result_no_reality")  # WHEN result
    rows = [{"label": t["row_when"], "score": f"{row['next_1y']:.3f}", "prediction": yn[when], "reality": yn[label] if label else t["unknown"], "kind": kind, "result": text}]  # WHEN row
    what, wlabel, safe = what_parts(row, top)  # WHAT parts
    if what is None:  # no WHAT score
        rows.append({"label": t["row_what"], "single": t["no_what_score"]})  # one text for the whole row
    else:  # WHAT score exists
        kind, text = result_cell(done, wlabel, safe, what, poc, "result_no_what_reality")  # WHAT result
        rows.append({"label": t["row_what"], "score": f"{row['stage0_score']:.3f}", "prediction": t["likely"] if what == "yes" else t["not_likely"], "reality": yn[wlabel] if (done and wlabel) else t["unknown"], "kind": kind, "result": text})  # WHAT row, reality only after the reveal
    if answer:  # the user predicted this year
        kind, text = result_cell(done, label, counted, answer, poc, "result_no_reality")  # user result, same coloring as WHEN
        rows.append({"label": t["row_you"], "score": "", "prediction": yn[answer], "reality": yn[label] if label else t["unknown"], "kind": kind, "result": text})  # user row
    return rows  # all rows


def notes(row: pd.Series, played: dict | None, top: dict, below: dict, cut: int | None, yn: dict, g: dict, poc: dict, n_min: int) -> list:  # small notes under the table
    """Small notes under the table: next_2y line, WHAT reliability, delay notes, warnings."""
    out = []  # note texts
    if played is not None:  # after the reveal
        lab2 = row["label_next_2y"]  # stored next_2y label
        out.append(g["next2_info"].format(score=row["next_2y"], label_part="" if pd.isna(lab2) else g["label_part"].format(label=yn["yes" if lab2 == 1 else "no"])))  # next_2y information
    what, label, safe = what_parts(row, top)  # WHAT parts
    if what is not None:  # a WHAT score exists
        rel = reliability(top)  # x and y
        out.append(g["what_reliability"].format(x=rel[0], y=rel[1]) if what == "yes" else g[outside_text_key(below["z"], rel[1], g["similar_gap"])].format(z=below["z"], y=rel[1]))  # reliability, Yes or No variant
        if played is not None and label is not None and not safe:  # label after the holdout
            out.append(g["what_delay_note"])  # cataloguing-delay note
    if cut is not None:  # some year is shown but not counted
        out.append(g["uncounted_note"].format(year=cut))  # fixed note
    if row["artist_train_rows"] == 0:  # artist absent from training
        out.append(g["unseen_warning"])  # model has not seen the artist
    out.append(g["what_years_warning"].format(n=n_min))  # WHAT coverage
    return out  # all notes


def panel_html(rows: list, note_list: list, poc: dict) -> str:  # the always visible panel
    """The always visible panel: title, table and small notes."""
    t = poc["texts"]  # texts
    head = "".join(f"<th>{html.escape(t[k])}</th>" for k in ("col_score", "col_prediction", "col_reality", "col_result"))  # header cells
    body = []  # table rows
    for r in rows:  # one row each
        if "single" in r:  # a single text for the whole row
            body.append(f"<tr><th>{html.escape(r['label'])}</th><td colspan='4'>{html.escape(r['single'])}</td></tr>")  # no WHAT score
        else:  # full row
            body.append(f"<tr><th>{html.escape(r['label'])}</th><td>{html.escape(r['score'])}</td><td>{html.escape(r['prediction'])}</td><td>{html.escape(r['reality'])}</td><td class='poc-cell {r['kind']}'>{html.escape(r['result'])}</td></tr>")  # colored result cell
    legend = f"<div class='poc-note'>{html.escape(t['legend_reference'])}</div>"  # fixed legend line under the table
    notes_html = "".join(f"<div class='poc-note'>{html.escape(n)}</div>" for n in note_list)  # small notes
    return css(poc) + f"<div class='poc-panel'><div class='poc-panel-title'>{html.escape(t['panel_title'])}</div><table class='poc-table'><tr><th></th>{head}</tr>{''.join(body)}</table>{legend}{notes_html}</div>"  # panel, legend first


def chips_html(lines: list, poc: dict) -> str:  # scoreboard chips
    """Scoreboard chips from already formatted lines."""
    return css(poc) + "<div class='poc-chips'>" + "".join(f"<span class='poc-chip'>{html.escape(x)}</span>" for x in lines) + "</div>"  # one chip per line


def multi_css(poc: dict) -> str:  # style rules of the mode 2 table, all scoped to the .poc-multi wrapper
    """Style block of the mode 2 table; every rule starts with .poc-multi so nothing can reach another tab."""
    c, s = poc["colors"], poc["sizes"]  # colors and sizes
    rules = [  # one CSS rule per line
        ".poc-multi .poc-table{table-layout:fixed}",  # equal column widths, so all placeholder cells are equally wide
        f".poc-multi .poc-table th:first-child{{width:{s['multi_artist_col']}%}}",  # artist column
        f".poc-multi .poc-ph{{background:{c['placeholder_bg']};border:{s['placeholder_border_width']}px dashed {c['placeholder_border']};text-align:center}}",  # dark placeholder cell, dashed border, no grey fill
        f".poc-multi .poc-ph span{{color:currentColor;opacity:{s['placeholder_opacity']}}}",  # dim text that follows the theme text color
        f".poc-multi .poc-pill{{display:inline-block;padding:{s['pill_pad_v']}px {s['pill_pad_h']}px;border-radius:{s['pill_radius']}px;border:{s['pill_border']}px solid {c['accent']};font-size:{s['pill_font']}px;font-weight:600;white-space:nowrap}}",  # pill shape
        f".poc-multi .poc-pill.yes{{background:{c['accent']};color:{c['accent_fg']}}}",  # Yes and Likely: filled
        f".poc-multi .poc-pill.no{{background:transparent;color:{c['accent']}}}",  # No and Not likely: outlined
        f".poc-multi .poc-sub{{font-size:{s['note_font']}px;opacity:{s['muted_opacity']}}}",  # small muted line or label
        f".poc-multi .poc-casc{{display:flex;align-items:flex-end;gap:{s['card_gap']}px;flex-wrap:wrap}}",  # the cascade of pills in one cell
        ".poc-multi .poc-step{display:flex;flex-direction:column;align-items:flex-start}",  # one label above its pill or text
        ".poc-multi th .poc-sub{font-weight:400}",  # the year-end tag under an artist name is not bold
    ]  # end of the rules
    return "<style>" + "".join(rules) + "</style>"  # the style element


def pill(text: str, yes: bool) -> str:  # a verdict pill
    """A rounded verdict pill: filled for Yes and Likely, outlined for No and Not likely."""
    return f"<span class='poc-pill {'yes' if yes else 'no'}'>{html.escape(text)}</span>"  # pill element


def how_lines(stage: str, rel_lines: list, legend: str, cascade: str) -> list:  # lines of the How to read this expander
    """The two reliability lines, the legend and the cascade sentence, from the predict stage on; nothing in the placeholder stage."""
    return [] if stage == "placeholder" else list(rel_lines) + [legend, cascade]  # empty before Predict

def short_title(title: str, limit: int) -> str:  # title cut at a maximum number of characters
    """The title, cut to `limit` characters with an ellipsis when longer."""
    return title if len(title) <= limit else title[:limit - 1].rstrip() + "\u2026"  # ellipsis counts in the limit

def year_end_tag(rt: dict, year: int) -> str:  # grey line of a year-end row
    """'No release in {year} · estimated from history', the line of a year-end row in both modes."""
    return rt["year_end_tag"].format(year=year)  # text from the yaml

def reality_item(label: str | None, title: str | None, yn: dict, rt: dict) -> str:  # one reality cell content
    """'Yes: <title>' with the title cut at route.title_max_chars and the full title in a tooltip, 'Yes' without a title, 'No', or the no-reality text when there is no stored label."""
    if label is None or pd.isna(label):  # no stored label
        return html.escape(rt["no_reality"])  # no-reality text
    if label == "no" or title is None or pd.isna(title):  # No, or a Yes with no title in the file
        return html.escape(yn[label])  # plain word
    return f"<span title='{html.escape(str(title), quote=True)}'>{html.escape(yn[label] + ': ' + short_title(str(title), rt['title_max_chars']))}</span>"  # Yes, cut title, tooltip with the full title

def cascade_div(r: pd.Series, items: list, rt: dict) -> str:  # the next-year item and, when shown, the 2-year item
    """Cascade with the next-year item first (small label above) and, when show2, an arrow and the 2-year item. `items` holds the two inner HTML strings."""
    parts = [f"<div class='poc-step'><span class='poc-sub'>{html.escape(rt['pill_next'])}</span>{items[0]}</div>"]  # next-year item
    if r["show2"]:  # the second item is shown
        parts.append("<span class='poc-arrow'>\u2192</span>")  # arrow
        parts.append(f"<div class='poc-step'><span class='poc-sub'>{html.escape(rt['pill_two'])}</span>{items[1]}</div>")  # 2-year item
    return f"<div class='poc-casc'>{''.join(parts)}</div>"  # the cascade

def cascade_cell(r: pd.Series, items: list, rt: dict) -> str:  # cascade inside a table cell
    """The cascade as a table cell."""
    return f"<td>{cascade_div(r, items, rt)}</td>"  # the cell

def side_panel_html(r: pd.Series, reveal: bool, g: dict, route: dict, poc: dict) -> str:  # mode 1 side panel
    """Mode 1 WHEN cascade from a multi_rows row, the same one as mode 2: the model pills, and after the reveal the reality items, with the year-end tag for a year without a Work."""
    rt = {**route["texts"], "title_max_chars": route["title_max_chars"]}  # route texts with the title limit
    yn = {"yes": g["answer_yes"], "no": g["answer_no"]}  # Yes / No words
    parts = [cascade_div(r, [pill(yn[r["when_verdict"]], r["when_verdict"] == "yes"), pill(yn[r["when2_verdict"]], r["when2_verdict"] == "yes")], rt)]  # model pills
    if reveal:  # realities
        parts.append(cascade_div(r, [reality_item(r["when_label"], r["when_title"], yn, rt), reality_item(r["when2_label"], r["when2_title"], yn, rt)], rt))  # reality items, same structure
    tag = f"<div class='poc-sub'>{html.escape(year_end_tag(rt, int(r['year'])))}</div>" if r["is_year_end"] else ""  # grey tag of a year-end row
    return css(poc) + multi_css(poc) + f"<div class='poc-multi'>{tag}{''.join(parts)}</div>"  # style blocks and the scoped wrapper

def multi_table_html(rows: pd.DataFrame, summary: dict, g: dict, route: dict, poc: dict, marks_fn, n_min: int, year: int, stage: str = "reveal") -> str:  # mode 2 table
    """The mode 2 table as one HTML string. Stage placeholder: the artist (with the year-end tag when there is no Work) and placeholders. Stage predict: the model verdicts as pills (next year, then within 2 years where shown). Stage reveal: the realities, the marks and the two summary lines."""
    t, rt = poc["texts"], route["texts"]  # poc and route texts
    rt = {**rt, "title_max_chars": route["title_max_chars"]}  # the title limit sits next to the texts
    yn = {"yes": g["answer_yes"], "no": g["answer_no"]}  # Yes / No words
    head = "".join(f"<th>{html.escape(rt[k])}</th>" for k in ("col_artist", "col_when", "col_when_reality", "col_what", "col_what_reality", "col_result"))  # header cells
    ph = f"<td class='poc-ph'><span>{html.escape(rt['placeholder'])}</span></td>"  # one placeholder cell, one per column
    body = []  # table rows
    for _, r in rows.iterrows():  # one row per artist
        tag = f"<div class='poc-sub'>{html.escape(year_end_tag(rt, year))}</div>" if r["is_year_end"] else ""  # grey tag under the artist of a year-end row
        name = f"<th>{html.escape(r['artist'])}{tag}</th>"  # artist cell, always shown
        if stage == "placeholder":  # nothing but the artist
            body.append(f"<tr>{name}{ph * 5}</tr>")  # five equal placeholder cells
            continue  # next artist
        when = cascade_cell(r, [pill(yn[r["when_verdict"]], r["when_verdict"] == "yes"), pill(yn[r["when2_verdict"]], r["when2_verdict"] == "yes")], rt)  # WHEN verdict pills
        if r["what_available"]:  # WHAT score exists
            what_model = f"<td>{pill(t['likely'] if r['what_verdict'] == 'yes' else t['not_likely'], r['what_verdict'] == 'yes')}</td>"  # WHAT verdict pill
            what_rest = ph  # WHAT reality placeholder
        elif r["is_year_end"]:  # year-end row: no WHAT score because no release
            what_model, what_rest = f"<td colspan='2'>{html.escape(rt['what_year_end'])}</td>", ""  # one text over the two WHAT columns
        else:  # Work year without a WHAT score
            what_model, what_rest = f"<td colspan='2'>{html.escape(rt['what_missing'].format(n=n_min))}</td>", ""  # reason from the yaml
        if stage == "predict":  # realities and marks stay placeholders
            body.append(f"<tr>{name}{when}{ph}{what_model}{what_rest}{ph}</tr>")  # model columns shown
            continue  # next artist
        reality = cascade_cell(r, [reality_item(r["when_label"], r["when_title"], yn, rt), reality_item(r["when2_label"], r["when2_title"], yn, rt)], rt)  # WHEN realities in the same structure, with the release title
        what_real = f"<td>{reality_item(r['what_label'], None, yn, rt)}</td>" if r["what_available"] else ""  # WHAT reality, plain
        mk = "".join(f"<span class='poc-mark {k}'>{html.escape(t[key])}</span>" for k, key in marks_fn(r))  # one mark per pill shown plus WHAT
        body.append(f"<tr>{name}{when}{reality}{what_model}{what_real}<td>{mk}</td></tr>")  # full row
    lines = [rt["summary_when"].format(k=summary["when_right"], n=summary["when_n"], k2=summary["when2_right"], n2=summary["when2_n"]), rt["summary_what"].format(k=summary["what_right"], n=summary["what_n"])] if stage == "reveal" else []  # only the two summary lines, after Show reality
    notes_html = "".join(f"<div class='poc-note'>{html.escape(x)}</div>" for x in lines)  # small notes
    return css(poc) + multi_css(poc) + f"<div class='poc-panel poc-multi'><table class='poc-table'><tr>{head}</tr>{''.join(body)}</table>{notes_html}</div>"  # table and notes in the scoped wrapper

def style_table_html(rows: pd.DataFrame, summary: dict | None, flagged: bool, st: dict, poc: dict, marks_fn, stage: str, top_n: int) -> str:  # tab 7 table
    """The tab 7 table as one HTML string. Stage placeholder: ranks and placeholder cells. Stage predict: style, expected Works as a pill and the range. Stage reveal: actual Works, the in-range mark and, for a counted year, the two summary lines under the table."""
    t, tx = poc["texts"], st["texts"]  # poc and tab texts
    head = "".join(f"<th>{html.escape(tx[k])}</th>" for k in ("col_rank", "col_style", "col_expected_growth", "col_expected", "col_range", "col_actual", "col_actual_growth", "col_result"))  # header cells
    ph = f"<td class='poc-ph'><span>{html.escape(tx['placeholder'])}</span></td>"  # one placeholder cell
    pct = lambda v: html.escape(tx["percent_format"].format(v * 100))  # a growth as a percentage
    body = []  # table rows
    for i in range(1, top_n + 1):  # one row per rank
        if stage == "placeholder":  # nothing about the styles yet
            body.append(f"<tr><th>{i}</th>{ph * 7}</tr>")  # rank and seven placeholders
            continue  # next rank
        r = rows[rows["rank"] == i]  # forecast row of the rank
        if r.empty:  # fewer eligible styles than top_n
            continue  # no row
        r = r.iloc[0]  # the row
        model = f"<td>{html.escape(str(r['style']))}</td><td>{pct(r['expected_growth'])}</td><td><span class='poc-pill yes'>{r['expected_works']:,.0f}</span></td><td>{r['range_low']:,.0f} to {r['range_high']:,.0f}</td>"  # style, expected pill and range
        if stage == "predict":  # reality stays placeholders
            body.append(f"<tr><th>{i}</th>{model}{ph}{ph}{ph}</tr>")  # model columns shown, three reality placeholders
            continue  # next rank
        color, mark = marks_fn(r["in_range"], flagged)  # mark of the row
        actual = html.escape(tx["no_actual"]) if pd.isna(r["actual_works"]) else f"{r['actual_works']:,.0f}"  # actual Works, plain
        growth = html.escape(tx["no_actual"]) if pd.isna(r["actual_growth"]) else pct(r["actual_growth"])  # actual growth, same units as the expected growth
        body.append(f"<tr><th>{i}</th>{model}<td>{actual}</td><td>{growth}</td><td><span class='poc-mark {color}'>{html.escape(t[mark])}</span></td></tr>")  # full row
    lines = []  # notes under the table
    if stage == "reveal" and summary is not None:  # counted years only
        lines = [tx["summary_hits"].format(k=summary["hits"], n=summary["n"]), tx["summary_range"].format(k=summary["in_range"], n=summary["m"])]  # both summary lines
    notes_html = "".join(f"<div class='poc-note'>{html.escape(x)}</div>" for x in lines)  # small notes
    return css(poc) + multi_css(poc) + f"<div class='poc-panel poc-multi'><table class='poc-table'><tr>{head}</tr>{''.join(body)}</table>{notes_html}</div>"  # table and notes in the scoped wrapper
