"""HTML and CSS flow card of tab 2 (inline SVG does not render through st.html). Texts come from the caller, colors from the palette."""

import html  # escape of text

from ui_components import card, chip, info  # card, pill and small i with a tooltip


def _box(label: str, count: str, tip: str, color: str, left: bool = False) -> str:  # one flow box
    """A box with a label (and its tooltip) and a big count in `color`."""
    return f"<div class='ui-flow-box' style='border-color:{color}'><div class='ui-label'>{html.escape(label)}{info(tip, left=left) if tip else ''}</div><div class='ui-flow-count' style='color:{color}'>{html.escape(count)}</div></div>"  # box


def clean_flow(cfg: dict, t: dict, n: int, counts: dict) -> str:  # tab 2 cleaning flow
    """Two input boxes, an arrow to the Releases box, a muted connector word to the Artists box; counts holds master, standalone, works and artists as text."""
    color = cfg["design"]["palette"]["highlight"]  # box color
    arrow = "<div class='ui-flow-arrow'></div>"  # CSS arrow pointing down
    by = f"<div class='ui-flow-by'>{html.escape(t['t2_by'])}</div>"  # connector word between releases and artists
    inputs = _box(t["t2_in_master"], counts["master"], t["t2_tip_master"], color) + _box(t["t2_in_standalone"].format(n=n), counts["standalone"], t["t2_tip_standalone"], color, left=True)  # the two sources
    return f"<div class='ui-flow'><div class='ui-flow-row'>{inputs}</div>{arrow}{_box(t['t2_out'], counts['works'], t['t2_tip_out'], color)}{by}{_box(t['t2_artists'], counts['artists'], '', color)}</div>"  # whole flow

def method_css(cfg: dict) -> str:  # style rules of the tab 4 infographic and model chips
    """Rules prefixed ui-m-; colors from the palette, sizes from design.method."""
    d = cfg["design"]  # design settings
    c, ty, m = d["palette"], d["type"], d["method"]  # palette, type scale and method sizes
    rules = [  # one rule per element
        f".ui-m-row{{display:grid;grid-template-columns:{m['label_w']}px 1fr;align-items:center;margin-bottom:{m['row_gap']}px}}",  # label column and track
        f".ui-m-label{{font-size:{ty['label']}px;color:{c['muted']}}}",  # muted run label
        f".ui-m-row.final .ui-m-label{{color:{c['highlight']};font-weight:700}}",  # accented final label
        f".ui-m-track{{display:grid;height:{m['row_h']}px}}",  # one row of the shared year axis, columns inline in percent
        f".ui-m-train{{background:{c['chance']};display:flex;align-items:center;padding-left:{m['text_pad']}px;font-size:{ty['caption']}px;color:{c['text']};white-space:nowrap;overflow:hidden}}",  # train bar, flat
        f".ui-m-row.final .ui-m-train{{background:{c['highlight']};color:{c['text']}}}",  # final train bar
        f".ui-m-gap{{border:{d['box']['border']}px dashed {c['muted']};box-sizing:border-box}}",  # gap block, outline only
        f".ui-m-test{{line-height:1.1;background:{c['rule']};color:{c['text']};display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:{ty['caption']}px;font-weight:600;border-radius:{m['test_radius']}px}}",  # test block
        f".ui-m-row.final .ui-m-test{{background:{c['highlight']}}}",  # sealed test block
        f".ui-m-sealed{{display:flex;align-items:center;gap:{m['text_pad']}px}}",  # lock and the word Sealed
        f".ui-m-lock{{position:relative;display:inline-block;width:{m['lock_w']}px;height:{m['lock_h']}px;background:{c['text']};border-radius:2px;margin-top:{m['shackle_h']}px}}",  # lock body
        f".ui-m-lock::before{{content:'';position:absolute;left:50%;transform:translateX(-50%);top:-{m['shackle_h']}px;width:{m['lock_w'] - 4}px;height:{m['shackle_h']}px;border:2px solid {c['text']};border-bottom:none;border-radius:{m['lock_w']}px {m['lock_w']}px 0 0;box-sizing:border-box}}",  # lock shackle
        f".ui-m-axis{{position:relative;height:{ty['caption'] * 2}px}}",  # tick row
        f".ui-m-tick{{position:absolute;transform:translateX(-50%);font-size:{ty['caption']}px;color:{c['muted']}}}",  # one year label
        f".ui-m-chips{{display:flex;flex-wrap:wrap;align-items:flex-end;gap:{d['box']['gap']}px}}",  # model chips with arrows
        f".ui-m-col{{display:flex;flex-direction:column;align-items:flex-start}}",  # chip and its flag
        f".ui-m-flag{{font-size:{ty['caption']}px;font-weight:700;color:{c['highlight']};min-height:{ty['caption'] * 2}px}}",  # Final model flag
        f".ui-m-arrow{{position:relative;width:{m['arrow_len']}px;height:{d['box']['border'] * 2}px;background:{c['muted']};margin-bottom:{m['arrow_lift']}px}}",  # arrow shaft
        f".ui-m-arrow::after{{content:'';position:absolute;right:-1px;top:50%;transform:translateY(-50%);border-top:{m['arrow_head']}px solid transparent;border-bottom:{m['arrow_head']}px solid transparent;border-left:{m['arrow_head']}px solid {c['muted']}}}",  # arrow head
        f".ui-m-inline{{display:flex;align-items:center}}",  # chip and its tooltip icon on one line, centered
        f".ui-m-legend{{display:flex;flex-wrap:wrap;align-items:center;gap:{d['box']['gap'] * 2}px;font-size:{ty['caption']}px;color:{c['muted']}}}",  # legend row above the card
        f".ui-m-key{{display:inline-flex;align-items:center;gap:{m['text_pad']}px}}",  # swatch and its word
        f".ui-m-sw{{display:inline-block;width:{m['swatch_w']}px;height:{m['swatch_h']}px;box-sizing:border-box}}",  # legend swatch
        f".ui-m-finals{{display:flex;flex-wrap:wrap;align-items:center;gap:{d['box']['gap']}px;font-size:{ty['label']}px;color:{c['muted']}}}",  # final models line
    ]
    return "<style>" + "".join(rules) + "</style>"  # the style element

def method_card(cfg: dict, t: dict, runs: list, last_year: int) -> str:  # tab 4 time infographic
    """Six rows on one year axis: five practice runs and the final exam; every column width is a share of the axis in percent."""
    m = cfg["design"]["method"]  # sizes
    start, step = cfg["design"]["method_axis_start"], cfg["design"]["method_tick_every"]  # axis start and tick spacing
    span = last_year - start + 1  # years on the axis, the last year included
    pct = lambda years: f"{years / span * 100:.4f}%"  # width of a block in percent
    rows = []  # one row of html per run
    for i, r in enumerate(runs):  # practice runs first, final exam last
        final = i == len(runs) - 1  # the last run is the final exam
        cols = [r["train_end"] + 1 - start, r["test_start"] - r["gap_start"], r["test_end"] - r["test_start"] + 1, last_year - r["test_end"]]  # train, gap, test and tail in years
        test_text = f"<span class='ui-m-sealed'><i class='ui-m-lock'></i>{html.escape(t['t4_sealed'])}</span>" if final else ""  # lock and Sealed only on the final exam
        test = f"<div class='ui-m-test'>{test_text}<span>{html.escape(t['t4_test'].format(first=r['test_start'], last=r['test_end']))}</span></div>"  # test block with its years
        train_text = t["t4_train_final"].format(year=r["train_end"]) if final else t["t4_train"]  # the final exam names its last training year
        blocks = f"<div class='ui-m-train'>{html.escape(train_text)}</div><div class='ui-m-gap' title='{html.escape(t['t4_tip_gap'], quote=True)}'></div>{test}" + ("<div></div>" if cols[3] else "")  # train, gap, test and the empty tail
        grid = " ".join(pct(y) for y in cols if y)  # percent columns, an empty tail is left out
        label = t["t4_final"] if final else t["t4_run"].format(n=i + 1)  # row label
        attrs = f"data-train-end='{r['train_end']}' data-gap='{r['gap_start']}-{r['test_start'] - 1}' data-test='{r['test_start']}-{r['test_end']}'"  # years of the row, for the control
        rows.append(f"<div class='ui-m-row{' final' if final else ''}' {attrs}><div class='ui-m-label'>{html.escape(label)}</div><div class='ui-m-track' style='grid-template-columns:{grid}'>{blocks}</div></div>")  # row
    ticks = "".join(f"<span class='ui-m-tick' style='left:{pct(y - start)}'>{y}</span>" for y in range(-(-start // step) * step, last_year + 1, step))  # a label every `step` years
    axis = f"<div class='ui-m-row'><div></div><div class='ui-m-axis'>{ticks}</div></div>"  # year axis under the rows
    return card("".join(rows) + axis)  # one card

def method_legend(cfg: dict, t: dict) -> str:  # tab 4 legend above the card
    """Three swatches: train bar, dashed gap block (with its tooltip) and test block."""
    c, b = cfg["design"]["palette"], cfg["design"]["box"]["border"]  # palette and border width
    key = lambda swatch, text, tip="": f"<span class='ui-m-key'>{swatch}{html.escape(text)}{info(tip) if tip else ''}</span>"  # one swatch with its word
    train = f"<i class='ui-m-sw' style='background:{c['chance']}'></i>"  # train swatch, same flat color as the bars
    gap = f"<i class='ui-m-sw' style='border:{b}px dashed {c['muted']}'></i>"  # gap swatch, dashed outline
    test = f"<i class='ui-m-sw' style='background:{c['rule']};border-radius:{cfg['design']['method']['test_radius'] // 2}px'></i>"  # test swatch
    return "<div class='ui-m-legend'>" + key(train, t["t4_legend_train"]) + key(gap, t["t4_legend_gap"], t["t4_tip_gap"]) + key(test, t["t4_legend_test"]) + "</div>"  # legend row

def final_models_html(cfg: dict, tags: list) -> str:  # tab 4 final models line
    """The label "Final models:" and one small tag per final model."""
    t, c = cfg["static_tabs"]["texts"], cfg["design"]["palette"]  # texts and palette
    return "<div class='ui-m-finals'><span>" + html.escape(t["t4_final_models"]) + "</span>" + "".join(chip(tag, c["rule"]) for tag in tags) + "</div>"  # label and tags

def method_chips(cfg: dict, t: dict, names: list) -> str:  # tab 4 model chips
    """Release counts, then the compared models with arrows; the last model is the final one and carries the flag and the row tooltip."""
    pal = cfg["design"]["palette"]  # palette
    items = [(t["t5_col_rule"], pal["muted"], "")] + [(n, pal["rule"], "") for n in names[:-1]] + [(names[-1], pal["highlight"], t["t4_final_flag"])]  # label, color and flag of each chip
    parts = [f"<div class='ui-m-col'><span class='ui-m-flag'>{html.escape(flag)}</span><div class='ui-m-inline'>{chip(text, color)}{info(t['t4_chips_tip'], left=True) if flag else ''}</div></div>" for text, color, flag in items]  # one column per chip, the tooltip icon sits right after the final one
    return "<div class='ui-m-chips'>" + "<div class='ui-m-arrow'></div>".join(parts) + "</div>"  # arrows between the chips

def close_css(cfg: dict) -> str:  # style rules of the tab 8 closing cards and band
    """Rules prefixed ui-close-; colors from the palette, the marker size from design.method."""
    d = cfg["design"]  # design settings
    c, ty, b = d["palette"], d["type"], d["box"]  # palette, type scale and box sizes
    rules = [  # one rule per element
        f".ui-close-title{{display:flex;align-items:center;gap:{b['gap']}px;font-size:{ty['subtitle']}px;font-weight:700;color:{c['text']};margin-bottom:{b['gap']}px}}",  # marker and card title
        f".ui-close-dot{{width:{d['method']['dot']}px;height:{d['method']['dot']}px;border-radius:50%;background:{c['highlight']};flex:none}}",  # round marker in the highlight color
        f".ui-close-item{{font-size:{ty['label']}px;color:{c['text']};line-height:1.4;margin-bottom:{b['gap'] // 2}px;text-align:left}}",  # one item
        f".ui-band{{background:{c['tint']};border-top:{b['border']}px solid {c['line']};border-bottom:{b['border']}px solid {c['line']};padding:{b['gap']}px;text-align:left}}",  # closing band, paper tint between two thin rules
        f".ui-close-blocks{{display:flex;gap:{b['gap'] * 2}px;margin-bottom:{b['gap']}px}}",  # the three blocks side by side, one text line above the closing sentence
        f".ui-close-block{{flex:1 1 0;min-width:0;display:flex;flex-direction:column;align-items:center;text-align:center;gap:{b['gap'] // 3}px;background:{c['card']};border-top:{b['border'] * 3}px solid {c['line']};border-radius:{b['radius']}px;padding:{b['gap']}px}}",  # tile: name, big number, caption and random line, centered, colored top rule set inline
        f".ui-close-name{{font-family:'{d['font']['heading']}',Georgia,serif;font-size:{ty['subtitle']}px;font-weight:600;color:{c['text']}}}",  # block name in the serif
        f".ui-close-big{{font-family:'{d['font']['heading']}',Georgia,serif;font-size:{round(ty['big'] * 1.3)}px;line-height:1.1;font-weight:600;white-space:nowrap;letter-spacing:{ty['head_tracking']}}}",  # big number, never wraps, color set inline to the question color
        f".ui-close-of{{font-family:'{d['font']['body']}',sans-serif;font-size:{round(ty['big'] * 1.3 / 3)}px;font-weight:400;letter-spacing:normal;color:{c['muted']};margin-left:{b['gap'] // 2}px}}",  # small muted "of n" after the number, same baseline
        f".ui-close-q{{font-size:{ty['label']}px;color:{c['text']};line-height:1.3}}",  # tile caption
        f".ui-close-rand{{font-size:{ty['caption']}px;color:{c['muted']};line-height:1.3}}",  # muted random line
        f".ui-close-status{{font-size:{ty['label']}px;color:{c['muted']};text-align:center;margin:-{b['gap'] // 2}px 0 {b['gap']}px}}",  # one sentence under the band title
        f".ui-row:has(.ui-close-title)>.ui-card{{padding:{b['gap']}px {b['pad']}px}}",  # tab 8 cards, shorter
        f".ui-band>.ui-close-title{{justify-content:center}}",  # band title centered above the tiles
        f"@media (max-width:1023px){{.ui-close-blocks{{flex-direction:column}}}}",  # tablet and phone: blocks stacked
        f".ui-band>.ui-close-item{{font-family:'{d['font']['heading']}',Georgia,serif;font-size:clamp({round(ty['subtitle'] * 1.1)}px,1.6vw + 10px,{round(ty['subtitle'] * 1.5)}px);line-height:1.35;margin-bottom:0;text-align:center}}",  # the sentence in the serif, centered
    ]
    return "<style>" + "".join(rules) + "</style>"  # the style element

def dots_css(cfg: dict) -> str:  # style rules of the two dot grids of tab 5
    """Rules prefixed ui-dots-; the grids shrink with their panel (columns share the width, dots keep a square cell), sizes from design.chart."""
    d = cfg["design"]  # design settings
    c, ch, b = d["palette"], d["chart"], d["box"]  # palette, chart sizes and box sizes
    rules = [  # one rule per element
        f".ui-dots{{display:flex;gap:{b['gap']}px;width:100%}}",  # the two grids side by side
        f".ui-dots-col{{flex:1 1 0;min-width:0;display:flex;flex-direction:column;align-items:center;text-align:center}}",  # one grid with its title, number and caption
        f".ui-dots-title{{font-size:{d['type']['label']}px;color:{c['text']};margin-bottom:{b['gap'] // 2}px;overflow-wrap:anywhere}}",  # grid title
        f".ui-dots-grid{{display:grid;grid-template-columns:repeat(var(--side),minmax(0,1fr));width:100%;max-width:calc(var(--side) * {ch['dot_cell']}px)}}",  # circles, never wider than the desktop size
        f".ui-dots-cell{{aspect-ratio:1/1;display:flex}}",  # square cell
        f".ui-dots-dot{{margin:auto;width:{round(ch['dot_fill'] * 100)}%;height:{round(ch['dot_fill'] * 100)}%;border-radius:50%;background:{c['chance']}}}",  # circle, share of its cell
        f".ui-dots-dot.on{{background:{c['style']}}}",  # filled circle in the Style color
        f".ui-dots-big{{font-size:clamp(18px,3vw,{ch['dot_big_font']}px);font-weight:700;color:{c['style']};margin-top:{b['gap'] // 2}px}}",  # big number
        f".ui-dots-cap{{font-size:{ch['dot_caption_font']}px;color:{c['muted']};overflow-wrap:anywhere}}",  # caption
        f"@media (max-width:639px){{.ui-dots{{flex-direction:column}}}}",  # phone: grids stacked
    ]
    return "<style>" + "".join(rules) + "</style>"  # the style element

def dots_html(cfg: dict, panels: list, side: int, texts: dict) -> str:  # two grids of circles
    """One side x side grid per panel; the first k circles use the Style color; a big number and a caption sit under each grid."""
    n = side * side  # circles per grid
    cols = []  # one column per panel
    for p in panels:  # in the given order
        cells = "".join(f"<div class='ui-dots-cell'><i class='ui-dots-dot{' on' if i < p['k'] else ''}'></i></div>" for i in range(n))  # reading order, first k filled
        tip = html.escape(f"{texts['tip']}: {p['rate_text']}", quote=True)  # tooltip of the grid with the exact share
        cols.append(f"<div class='ui-dots-col'><div class='ui-dots-title'>{html.escape(p['title'])}</div><div class='ui-dots-grid' style='--side:{side}' title='{tip}'>{cells}</div><div class='ui-dots-big'>{html.escape(p['big'])}</div><div class='ui-dots-cap'>{html.escape(p['caption'])}</div></div>")  # title, grid, number, caption
    return dots_css(cfg) + f"<div class='ui-dots'>{''.join(cols)}</div>"  # both grids

def inputs_css(cfg: dict) -> str:  # style rules of the tab 4 inputs view
    """Rules prefixed ui-in-; colors from the palette, gaps from design.box."""
    d = cfg["design"]  # design settings
    c, ty, b = d["palette"], d["type"], d["box"]  # palette, type scale and box sizes
    rules = [  # one rule per element
        f".ui-in-top{{display:flex;flex-wrap:wrap;align-items:center;gap:{b['gap']}px}}",  # chip next to the question
        f".ui-in-q{{font-size:{ty['label']}px;color:{c['muted']}}}",  # question line under the card title
        f".ui-in-row{{display:grid;grid-template-columns:minmax(0,2fr) minmax(0,3fr);gap:{b['gap']}px;align-items:center;padding:{b['gap'] // 2}px 0;min-height:{cfg['static_tabs']['inputs']['row_min']}px;box-sizing:border-box;border-top:1px solid {c['line']}}}",  # family and its example values
        ".ui-row:has(.ui-in-row)>.ui-card{display:flex;flex-direction:column}",  # the two cards have the same height, the row of a card fills it
        ".ui-in-row{flex:1 1 0}",  # family rows share the card height equally, so each row has the same height in both cards
        f".ui-in-fam{{display:flex;align-items:center;gap:{b['gap'] // 2}px;font-weight:600}}",  # family name and its count chip
        f".ui-in-vals{{font-size:{ty['caption']}px;color:{c['muted']};min-width:0}}",  # example values
        f"@media (max-width:639px){{.ui-in-row{{grid-template-columns:minmax(0,1fr)}}}}",  # phone: values under the family
    ]
    return "<style>" + "".join(rules) + "</style>"  # the style element

def inputs_card(color: str, title: str, facts: str, question: str, rows: list) -> str:  # one model card of the inputs view
    """Title, a chip with the number of facts and the question; rows holds (family, count, values) as text."""
    top = f"<div class='ui-in-top'>{chip(facts, color)}<span class='ui-in-q'>{html.escape(question)}</span></div>"  # chip and question
    body = "".join(f"<div class='ui-in-row'><div class='ui-in-fam'>{html.escape(f)}{chip(n, color)}</div><div class='ui-in-vals'>{html.escape(v)}</div></div>" for f, n, v in rows)  # one row per family
    return card(top + body, title)  # card

def funnel_css(cfg: dict) -> str:  # style rules of the tab 2 Cleaning view
    """Rules prefixed ui-fx-; the boxes reuse the ui-flow-box rules."""
    d = cfg["design"]  # design settings
    c, ty, b, f = d["palette"], d["type"], d["box"], d["flow"]  # palette, type scale, box and flow sizes
    rules = [  # one rule per element
        f".ui-fx{{display:flex;align-items:center;gap:{b['gap']}px}}",  # stages left to right
        f".ui-fx>.ui-flow-box{{flex:1 1 0;min-width:0}}",  # equal boxes that never push past the card
        f".ui-fx-lanes{{display:flex;flex-direction:column;gap:{b['gap']}px;flex:2 1 0;min-width:0}}",  # the two lanes, one above the other
        f".ui-fx-lane-title{{font-size:{ty['label']}px;font-weight:600;margin-bottom:{b['gap'] // 2}px}}",  # lane name
        f".ui-fx-pair{{display:flex;align-items:center;gap:{b['gap']}px}}",  # Read and Cleaned side by side
        f".ui-fx-pair>.ui-flow-box{{flex:1 1 0;min-width:0}}",  # equal boxes inside a lane
        f".ui-fx-count{{font-size:clamp(14px,2vw,{f['count_font'] * 4 // 5}px);font-weight:600;line-height:1.3;white-space:nowrap;font-variant-numeric:tabular-nums}}",  # count in a box, smaller than the tab 2 flow, never broken across lines
        f".ui-fx-reason{{font-size:{ty['caption']}px;color:{c['muted']};margin-top:{b['gap'] // 2}px}}",  # muted reason line
        f".ui-fx-arrow{{position:relative;flex:none;width:{f['arrow_len']}px;height:{f['arrow_width']}px;background:{c['muted']}}}",  # arrow shaft
        f".ui-fx-arrow::after{{content:'';position:absolute;right:-1px;top:50%;transform:translateY(-50%);border-top:{f['arrow_head']}px solid transparent;border-bottom:{f['arrow_head']}px solid transparent;border-left:{f['arrow_head']}px solid {c['muted']}}}",  # arrow head
        f".ui-fx-sub{{font-size:{ty['caption']}px;color:{c['muted']}}}",  # small muted label over a number
        f".ui-fx-down{{position:relative;width:{f['arrow_width']}px;height:{f['arrow_len'] // 2}px;background:{c['muted']};margin:{b['gap'] // 2}px auto {f['arrow_head']}px}}",  # short arrow between two numbers of a box, shaft
        f".ui-fx-down::after{{content:'';position:absolute;left:50%;bottom:-{f['arrow_head']}px;transform:translateX(-50%);border-left:{f['arrow_head']}px solid transparent;border-right:{f['arrow_head']}px solid transparent;border-top:{f['arrow_head']}px solid {c['muted']}}}",  # arrow head pointing down
        f".ui-fx-chips{{display:flex;flex-wrap:wrap;justify-content:center;gap:{b['gap'] // 2}px;margin-top:{b['gap'] // 2}px}}",  # releases and artists chips
        f"@media (max-width:1100px){{.ui-fx{{flex-direction:column;align-items:stretch}}.ui-fx>.ui-flow-box{{flex:none;width:auto}}.ui-fx>.ui-fx-arrow{{align-self:center;transform:rotate(90deg);margin:{f['arrow_len'] // 2}px 0}}.ui-fx-lanes{{flex:none;width:100%}}.ui-fx-count{{font-size:clamp(14px,3.2vw,{f['count_font'] * 4 // 5}px)}}}}",  # below 1100 px: lanes, Combined, duplicates, table stacked at full row width, arrows point down, numbers sized by the viewport
    ]
    return "<style>" + "".join(rules) + "</style>"  # the style element

def _fx_box(label: str, counts: list, color: str, reason: str = "", chips: list = (), under: bool = False) -> str:  # one box of the Cleaning view
    """A flow box with a label, one count per line, an optional muted reason and optional chips; the label of a (label, count) pair goes over the count, or under it when `under`."""
    one = lambda n: f"<div class='ui-fx-count' style='color:{color}'>{html.escape(n)}</div>"  # one count line
    nums = "".join("<div class='ui-fx-down'></div>" if n is None else (one(n[1]) + f"<div class='ui-fx-sub'>{html.escape(n[0])}</div>" if under else f"<div class='ui-fx-sub'>{html.escape(n[0])}</div>" + one(n[1])) if isinstance(n, tuple) else one(n) for n in counts)  # a count, a (label, count) pair or None for the arrow
    why = f"<div class='ui-fx-reason'>{html.escape(reason)}</div>" if reason else ""  # reason line
    tags = f"<div class='ui-fx-chips'>{''.join(chips)}</div>" if chips else ""  # chips under the counts
    return f"<div class='ui-flow-box' style='border-color:{color}'><div class='ui-label'>{html.escape(label)}</div>{nums}{why}{tags}</div>"  # box

def funnel_flow(cfg: dict, t: dict, v: dict) -> str:  # tab 2 Cleaning flow
    """Two lanes (Read, Cleaned) that merge into Combined, then the duplicates box (releases before and after) and the modelling table; v holds the counts as text."""
    color = cfg["design"]["palette"]["highlight"]  # box color
    arrow = "<div class='ui-fx-arrow'></div>"  # arrow between two boxes
    def lane(title: str, first: str, read: str, kept: str, reason: str) -> str:  # one lane, `first` is the label of its first box
        pair = _fx_box(first, [read], color) + arrow + _fx_box(t["t2c_cleaned"], [kept], color)  # Read, arrow, Cleaned
        return f"<div><div class='ui-fx-lane-title'>{html.escape(title)}</div><div class='ui-fx-pair'>{pair}</div><div class='ui-fx-reason'>{html.escape(reason)}</div></div>"  # lane
    lanes = lane(t["t2c_masters"], t["t2c_read"], v["masters_read"], v["masters_kept"], t["t2c_masters_reason"]) + lane(t["t2c_standalone"], t["t2c_nomaster"], v["standalone_nomaster"], v["standalone_kept"], t["t2c_standalone_reason"])  # both lanes
    final = _fx_box(t["t2c_final"], [(t["t2c_releases"], v["final_table"]), (t["t2c_artists"], v["final_artists"])], color, under=True)  # each number with its own label under it
    stages = f"<div class='ui-fx-lanes'>{lanes}</div>" + arrow + _fx_box(t["t2c_combined"], [v["combined"]], color) + arrow + _fx_box(t["t2c_noise"], [(t["t2c_before"], v["combined"]), None, (t["t2c_after"], v["final_table"])], color, t["t2c_noise_reason"]) + arrow + final  # whole flow
    return f"<div class='ui-fx'>{stages}</div>"  # flow
