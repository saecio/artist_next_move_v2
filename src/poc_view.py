"""HTML builders of tab 6 and tab 7: year cards, window, mode 1 panel blocks, mode 2 and tab 7 tables, chips. Pure string functions, no Streamlit."""

import html  # escaping of titles and texts

import pandas as pd  # missing-value checks

from demo_game import card_status, flag, names, verdict, year_cards  # game logic reused as it is


def css(poc: dict) -> str:  # style block from the config
    """Style block built from the colors and sizes of the poc section."""
    c, s = poc["colors"], poc["sizes"]  # colors and sizes
    rules = [  # one CSS rule per line
        f".poc-line{{display:flex;gap:{s['card_gap']}px;align-items:stretch}}",  # timeline of cards
        f".poc-card{{flex:1 1 {s['card_min_width']}px;min-width:{s['card_min_width']}px;border:1px solid {c['card_border']};border-radius:{s['card_radius']}px;padding:{s['card_padding']}px;font-size:{s['card_font']}px;line-height:1.25;position:relative;overflow-wrap:anywhere}}",  # one card, long words wrap instead of overflowing
        ".poc-year{font-weight:700}",  # year label, never truncated
        f".poc-title{{display:-webkit-box;-webkit-line-clamp:{s['title_lines']};-webkit-box-orient:vertical;overflow:hidden}}",  # titles on at most two lines
        f".poc-muted{{opacity:{s['muted_opacity']}}}",  # secondary text keeps the theme text color
        f".poc-pred{{border:{s['predicted_border']}px solid {c['card_border_predicted']};box-shadow:0 0 {s['predicted_shadow']}px {c['card_border_predicted']}}}",  # card of the predicted year
        f".poc-marks{{display:flex;gap:{s['card_gap']}px;margin-top:{s['card_gap']}px}}",  # row of the two marks
        f".poc-mark{{display:inline-flex;align-items:center;justify-content:center;min-width:{s['mark_size']}px;height:{s['mark_size']}px;box-sizing:border-box;padding:0 {s['mark_pad']}px;border-radius:{s['pill_radius']}px;font-size:{s['mark_font']}px;font-weight:{s['mark_weight']};line-height:1}}",  # one solid mark, round for one glyph, a pill for a glyph and an asterisk
        f".poc-earlier{{font-size:{s['note_font']}px;opacity:{s['muted_opacity']};margin-bottom:{s['card_gap']}px}}",  # marker of hidden earlier years
        f".poc-who{{font-weight:700;font-size:{s['table_font']}px}}",  # year heading of the left panel
        f".poc-panel{{font-size:{s['table_font']}px;margin-top:{s['card_padding']}px}}",  # prediction panel
        f".poc-table{{border-collapse:collapse;width:100%}}",  # table
        f".poc-table th,.poc-table td{{border-bottom:1px solid {c['card_border']};padding:{s['cell_padding']}px {s['card_padding']}px;text-align:left}}",  # cells
        f".poc-note{{font-size:{s['note_font']}px;opacity:{s['muted_opacity']};margin-top:2px}}",  # small notes under the table
        f".poc-chips{{display:flex;gap:{s['card_gap']}px;flex-wrap:wrap}}",  # scoreboard chips
        f".poc-chip{{background:{c['chip_bg']};color:{c['chip_fg']};border-radius:{s['card_radius']}px;padding:{s['cell_padding']}px {s['card_padding']}px;font-size:{s['chip_font']}px;font-weight:600}}",  # one chip
    ]  # close the rule list
    for kind in ("green", "red", "grey"):  # one rule per color
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


def window_years(current: int, first: int, last_shown: int, before: int, after: int) -> tuple:  # years of the window
    """Years of the window and whether earlier years are cut; 0 for both counts means all years."""
    if before == 0 and after == 0:  # all years
        return list(range(first, last_shown + 1)), False  # no cut
    start = max(first, current - before)  # first year of the window
    return list(range(start, min(last_shown, current + after) + 1)), start > first  # window and the cut flag


def card_html(year: int, aw: pd.DataFrame, hidden: dict, until: int, pred_year: int, mark_list: list, g: dict, poc: dict, ye: dict | None = None) -> str:  # one year card
    """One year card: year, Work titles up to titles_per_card with a more line, hidden count, the year-end tag of a year without a Work (`ye` maps year to its tag), and the two marks on the predicted year."""
    yc = year_cards(aw, year, g["hide_rerecordings"], poc["titles_per_card"])  # Works drawn this year
    status = card_status(yc["total"], yc["cards"], year, until)  # future, gap, hidden or card
    body = [f"<div class='poc-year'>{year}</div>"]  # year first
    if status == "card":  # Works are shown
        for c in yc["cards"]:  # one block per Work
            body.append(f"<div class='poc-title' title='{html.escape(str(c['title']) + ' | ' + names(c['genres']) + ' | ' + names(c['styles']), quote=True)}'>{html.escape(str(c['title']))}</div>")  # title, full text on hover
        if yc["more"]:  # more Works than the cap
            body.append(f"<div class='poc-muted'>{html.escape(g['more_line'].format(n=yc['more']))}</div>")  # "+N more" line
    else:  # future, gap or only hidden re-recordings
        text = {"future": poc["texts"]["future_card"], "gap": (ye or {}).get(year, poc["texts"]["gap_card"]), "hidden": poc["texts"]["only_rerecordings"]}[status]  # state text, the year-end tag replaces the gap text
        body.append(f"<div class='poc-muted'>{html.escape(text)}</div>")  # state line
    if g["hide_rerecordings"] and year <= until and hidden.get(year):  # hidden re-recordings in this year
        body.append(f"<div class='poc-muted'>{html.escape(g['hidden_line'].format(n=hidden[year]))}</div>")  # count line
    if year == pred_year:  # the predicted year carries the marks
        t = poc["texts"]  # texts
        body.append("<div class='poc-marks'>" + "".join(f"<span class='poc-mark {k}' title='{html.escape(t[title])}'>{html.escape(t[m])}</span>" for title, k, m in mark_list) + "</div>")  # two marks, no text
    cls = "poc-card poc-pred" if year == pred_year else "poc-card"  # predicted card is highlighted
    return f"<div class='{cls}' data-year='{year}'>{''.join(body)}</div>"  # the card


def timeline_html(aw: pd.DataFrame, hidden: dict, window: tuple, until: int, pred_year: int, mark_list: list, g: dict, poc: dict, ye: dict | None = None) -> str:  # the timeline element
    """The style block, the earlier-years marker when needed, and the line of cards of the window."""
    years, cut = window  # years and cut flag
    earlier = f"<div class='poc-earlier'>{html.escape(poc['texts']['earlier_hidden'])}</div>" if cut else ""  # marker
    cards = [card_html(y, aw, hidden, until, pred_year, mark_list, g, poc, ye) for y in years]  # one card per year
    return css(poc) + earlier + "<div class='poc-line'>" + "".join(cards) + "</div>"  # style, marker and line


def card_marks(r: pd.Series, done: bool, marks_fn) -> list:  # WHEN and WHAT marks of the predicted card from a mode 2 row
    """The WHEN and WHAT marks as (title key, color, mark key): grey dots before the reveal, after it the first (next year) and the last (WHAT) mark of the mode 2 marks, so a year-end year works too."""
    what_title = "mark_what_title" if r["what_available"] else "mark_not_forecast"  # hover text of the WHAT mark
    if not done:  # before the reveal
        return [("mark_when_title", "grey", "mark_dot"), (what_title, "grey", "mark_dot")]  # two grey dots
    shown = marks_fn(r)  # next year, optional 2-year, WHAT
    return [("mark_when_title", *shown[0]), (what_title, *shown[-1])]  # WHEN and WHAT


def work_panel_html(row: pd.Series, year: int, tag: str | None, poc: dict) -> str:  # left panel: the year and its representative Work
    """Year heading and the representative Work title (two lines at most, genres and styles on hover); the year-end tag replaces the Work in a year without one."""
    if tag is not None:  # year without a Work
        body = f"<div class='poc-sub'>{html.escape(tag)}</div>"  # grey tag
    else:  # a Work year
        hover = html.escape(str(row["title"]) + " | " + names(row["genres"]) + " | " + names(row["styles"]), quote=True)  # full text on hover
        body = f"<div class='poc-title' title='{hover}'>{html.escape(str(row['title']))}</div>"  # title
    return css(poc) + multi_css(poc) + f"<div class='poc-multi'><div class='poc-who'>{year}</div>{body}</div>"  # style blocks and the scoped wrapper


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
        f".poc-multi .poc-q{{font-weight:700;font-size:{s['table_font']}px}}",  # question of a block
        f".poc-multi .poc-ln{{font-size:{s['table_font']}px;margin-top:{s['card_gap']}px;overflow-wrap:anywhere}}",  # one line of a block, long words wrap
    ]  # end of the rules
    return "<style>" + "".join(rules) + "</style>"  # the style element


def na_cell(poc: dict) -> str:  # grey cell of a row without a WHAT score
    """Grey dash with a tooltip, the same for a year-end row and a row that is not eligible."""
    t = poc["texts"]  # texts
    return f"<span class='poc-muted' title='{html.escape(t['what_na_tip'], quote=True)}'>{html.escape(t['what_na'])}</span>"  # dash and tooltip

def pill(text: str, yes: bool, tip: str = "") -> str:  # a verdict pill
    """A rounded verdict pill: filled for Yes and Likely, outlined for No and Not likely; `tip` is the hover text (the score)."""
    return f"<span class='poc-pill {'yes' if yes else 'no'}'{f' title={chr(39)}{html.escape(tip, quote=True)}{chr(39)}' if tip else ''}>{html.escape(text)}</span>"  # pill element, tooltip only when given


def how_lines(stage: str, rel_lines: list, legend: str, cascade: str, independent: str) -> list:  # lines of the How to read this expander
    """The two reliability lines, the legend, the cascade sentence and the independence sentence, from the predict stage on; nothing in the placeholder stage."""
    return [] if stage == "placeholder" else list(rel_lines) + [legend, cascade, independent]  # empty before Predict

def short_title(title: str, limit: int) -> str:  # title cut at a maximum number of characters
    """The title, cut to `limit` characters with an ellipsis when longer."""
    return title if len(title) <= limit else title[:limit - 1].rstrip() + "\u2026"  # ellipsis counts in the limit

def year_end_tag(rt: dict, year: int) -> str:  # grey line of a year-end row
    """'No release in {year} · forecast from earlier releases', the line of a year-end row in both modes."""
    return rt["year_end_tag"].format(year=year)  # text from the yaml

def reality_item(label: str | None, title: str | None, yn: dict, rt: dict, year=None) -> str:  # one reality cell content
    """'Yes: <title> (<year>)' with the title cut at route.title_max_chars and the full title in a tooltip, the plain word (with the year when given) without a title, or the no-reality text when there is no stored label."""
    if label is None or pd.isna(label):  # no stored label
        return html.escape(rt["no_reality"])  # no-reality text
    when = "" if year is None or pd.isna(year) else f" ({int(year)})"  # release year after the word or the title
    if title is None or pd.isna(title):  # no title in the file
        return html.escape(yn[label] + when)  # plain word, with the year when given
    return f"<span title='{html.escape(str(title), quote=True)}'>{html.escape(yn[label] + ': ' + short_title(str(title), rt['title_max_chars']) + when)}</span>"  # word, cut title, year, tooltip with the full title

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

def tip_of(poc: dict, score: float) -> str:  # tooltip text of a prediction word
    """The score as a plain number (never a percentage, it is not calibrated)."""
    return poc["texts"]["score_tip"].format(score=score)  # text from the yaml

def line(label: str, value: str) -> str:  # one 'label value' line of a block
    """A label followed by an already escaped value."""
    return f"<div class='poc-ln'>{html.escape(label)} {value}</div>"  # one line

def styles_text(styles: list, poc: dict) -> str:  # at most three style names, then +N
    """Style names from the data, comma separated, the rest as '+N'."""
    t = poc["texts"]  # texts
    shown = ", ".join(styles[:t["max_styles"]])  # first names
    return shown + (" " + t["more_styles"].format(n=len(styles) - t["max_styles"]) if len(styles) > t["max_styles"] else "")  # '+N' for the rest

def release_line(title, year: int, rt: dict) -> str:  # 'Title (year)' under a reality line
    """Release title cut at route.title_max_chars with the full title in a tooltip, then its year; the year alone without a title."""
    if title is None or pd.isna(title):  # no title in the file
        return html.escape(rt["release_year_only"].format(year=year))  # year only
    return f"<span title='{html.escape(str(title), quote=True)}'>{html.escape(rt['release_line'].format(title=short_title(str(title), rt['title_max_chars']), year=year))}</span>"  # cut title, year, full title on hover

def block_html(label: str, question: str, lines: list, poc: dict) -> str:  # one question block of mode 1
    """Small grey label, the question and the lines, inside the scoped wrapper."""
    return css(poc) + multi_css(poc) + f"<div class='poc-multi poc-blk'><div class='poc-sub'>{html.escape(label)}</div><div class='poc-q'>{html.escape(question)}</div>{''.join(lines)}</div>"  # label, question, lines

def timing_html(r: pd.Series, step: str, answer: str | None, g: dict, route: dict, poc: dict) -> str:  # Timing block of mode 1
    """Timing block: the question, the user prediction (only when given), the model prediction (next year and, when the cascade rule shows it, within 2 years) and the reality lines, '?' before the reveal. Steps: 'ask' (nothing shown yet), 'model', 'reveal'."""
    t, rt = poc["texts"], {**route["texts"], "title_max_chars": route["title_max_chars"]}  # texts, with the title limit
    yn = {"yes": g["answer_yes"], "no": g["answer_no"]}  # Yes / No words
    year, shown, done = int(r["year"]), step != "ask", step == "reveal"  # question year and the two stage flags
    lines = [f"<div class='poc-sub'>{html.escape(year_end_tag(rt, year))}</div>"] if r["is_year_end"] else []  # grey tag of a year-end row
    if answer and shown:  # the user predicted this year
        lines.append(line(t["your_prediction"], html.escape(yn[answer])))  # user answer, only in this block
    two = bool(r["show2"]) and shown  # the 2-year items follow the cascade rule
    lines.append(line(t["prediction"], pill(yn[r["when_verdict"]], r["when_verdict"] == "yes", tip_of(poc, r["when_score"])) if shown else html.escape(t["unknown"])))  # next-year prediction
    if two:  # within 2 years
        lines.append(line(t["within_two"], pill(yn[r["when2_verdict"]], r["when2_verdict"] == "yes", tip_of(poc, r["when2_score"]))))  # 2-year prediction
    if not done:  # before the reveal: only '?'
        lines.append(line(t["reality"], html.escape(t["unknown"])))  # next-year reality
        if two:  # second reality line
            lines.append(line(t["reality_two"], html.escape(t["unknown"])))  # 2-year reality
        return block_html(t["label_timing"], t["q_timing"], lines, poc)  # block
    first = r["when_label"] if isinstance(r["when_label"], str) else None  # next-year label
    items = [(t["reality"], first, r["when_title"], year + 1)]  # next-year reality: label, title, release year
    if two:  # the 2-year reality has its own line
        items.append((t["reality_two"], r["when2_label"] if isinstance(r["when2_label"], str) else None, r["when2_title"], year + 1 if first == "yes" else year + 2))  # release year of the 2-year reality
    for name, label, title, y in items:  # one reality line each
        lines.append(line(name, html.escape(rt["no_reality"] if label is None else yn[label])))  # Yes, No or no reality
        if label == "yes":  # a release happened
            lines.append(f"<div class='poc-ln'>{release_line(title, y, {**t, "title_max_chars": rt["title_max_chars"]})}</div>")  # its title and year
    return block_html(t["label_timing"], t["q_timing"], lines, poc)  # block

def style_html(r: pd.Series, step: str, g: dict, route: dict, poc: dict) -> str:  # Style block of mode 1
    """Style block: the question, the always visible grey scope line with the shown year, the model prediction and the reality ('?' before the reveal, new styles for a Yes, the release styles and 'all used before' for a No); a row without a WHAT score shows the grey dash."""
    t, rt = poc["texts"], {**route["texts"], "title_max_chars": route["title_max_chars"]}  # texts, with the title limit
    shown, done = step != "ask", step == "reveal"  # stage flags
    lines = [f"<div class='poc-sub'>{html.escape(t['style_scope'].format(year=int(r['year'])))}</div>"]  # grey line, always visible
    if not r["what_available"]:  # no WHAT score: year-end row or not eligible
        lines.append(f"<div class='poc-ln'>{na_cell(poc)}</div>")  # grey dash with tooltip
        return block_html(t["label_style"], t["q_style"], lines, poc)  # block
    likely = r["what_verdict"] == "yes"  # model answer
    lines.append(line(t["prediction"], pill(t["likely"] if likely else t["not_likely"], likely, tip_of(poc, r["what_score"])) if shown else html.escape(t["unknown"])))  # prediction word, the score in the tooltip
    if not done:  # before the reveal: only '?'
        lines.append(line(t["reality"], html.escape(t["unknown"])))  # reality
        return block_html(t["label_style"], t["q_style"], lines, poc)  # block
    label = r["what_label"] if isinstance(r["what_label"], str) else None  # stored WHAT label
    if label is None:  # score but no stored label
        lines.append(f"<div class='poc-ln'>{html.escape(g['what_no_reality'])}</div>")  # explicit empty state
    elif label == "yes":  # new style: the styles come from the artist-year file
        lines.append(f"<div class='poc-ln'>{html.escape(g['what_reality_yes'].format(styles=styles_text(r['what_new'], poc)))}</div>")  # Yes with the new styles
        lines.append(f"<div class='poc-ln'>{release_line(r['what_release_title'], int(r['what_year']), {**t, 'title_max_chars': rt['title_max_chars']})}</div>")  # release title and its year, the next active year
    else:  # no new style: the release styles are all used before
        lines.append(f"<div class='poc-ln'>{html.escape(g['what_reality_no'])}</div>")  # No
        used = f" \u00b7 {html.escape(styles_text(r['what_styles'], poc))} {html.escape(t['used_before'])}" if r["what_styles"] else ""  # release styles, all used before
        lines.append(f"<div class='poc-ln'>{release_line(r['what_release_title'], int(r['what_year']), {**t, 'title_max_chars': rt['title_max_chars']})}{used}</div>")  # release title, its year and styles
    return block_html(t["label_style"], t["q_style"], lines, poc)  # block

def multi_table_html(rows: pd.DataFrame, summary: dict, g: dict, route: dict, poc: dict, marks_fn, year: int, stage: str = "reveal") -> str:  # mode 2 table
    """The mode 2 table as one HTML string. Stage placeholder: the artist (with the year-end tag when there is no Work) and placeholders. Stage predict: the model verdicts as pills (next year, then within 2 years where shown). Stage reveal: the realities, the marks and the two summary lines."""
    t, rt = poc["texts"], route["texts"]  # poc and route texts
    rt = {**rt, "title_max_chars": route["title_max_chars"]}  # the title limit sits next to the texts
    yn = {"yes": g["answer_yes"], "no": g["answer_no"]}  # Yes / No words
    two = lambda text, tip="": f"<th colspan='2'{f' title={chr(39)}{html.escape(tip, quote=True)}{chr(39)}' if tip else ''}>{html.escape(text)}</th>"  # header cell over a prediction and a reality column
    sub_head = "".join(f"<th>{html.escape(rt[k])}</th>" for k in ("col_prediction", "col_reality") * 2)  # second header row
    head = f"<th rowspan='2'>{html.escape(rt['col_artist'])}</th>{two(rt['col_timing'])}{two(rt['col_style'], rt['col_style_tip'])}<th rowspan='2'>{html.escape(rt['col_result'])}</th></tr><tr>{sub_head}"  # first header row, closed to open the second
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
        else:  # no WHAT score (year-end row or not eligible): one grey dash over the two WHAT columns
            what_model, what_rest = f"<td colspan='2'>{na_cell(poc)}</td>", ""  # data-driven, a score in the data shows as in any row
        if stage == "predict":  # realities and marks stay placeholders
            body.append(f"<tr>{name}{when}{ph}{what_model}{what_rest}{ph}</tr>")  # model columns shown
            continue  # next artist
        reality = cascade_cell(r, [reality_item(r["when_label"], r["when_title"], yn, rt, year + 1 if r["when_label"] == "yes" else None), reality_item(r["when2_label"], r["when2_title"], yn, rt, (year + 1 if r["when_label"] == "yes" else year + 2) if r["when2_label"] == "yes" else None)], rt)  # WHEN realities in the same structure, with the release title and its year
        what_real = f"<td>{reality_item(r['what_label'], r['what_release_title'], yn, rt, r['what_year'])}</td>" if r["what_available"] else ""  # WHAT reality, with the title and year of the next active year
        shown = marks_fn(r)  # next year, optional 2-year, WHAT
        tips = [""] * (len(shown) - 1) + ["" if r["what_available"] else f" title='{html.escape(t['mark_not_forecast'], quote=True)}'"]  # only the WHAT mark of a row without a score gets a tooltip
        mk = "".join(f"<span class='poc-mark {k}'{tip}>{html.escape(t[key])}</span>" for (k, key), tip in zip(shown, tips))  # one mark per pill shown plus WHAT
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
