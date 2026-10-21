"""HTML and CSS flow card of tab 2 (inline SVG does not render through st.html). Texts come from the caller, colors from the palette."""

import html  # escape of text

from ui_components import card, chip, info  # card, pill and small i with a tooltip


def _box(label: str, count: str, tip: str, color: str) -> str:  # one flow box
    """A box with a label (and its tooltip) and a big count in `color`."""
    return f"<div class='ui-flow-box' style='border-color:{color}'><div class='ui-label'>{html.escape(label)}{info(tip)}</div><div class='ui-flow-count' style='color:{color}'>{html.escape(count)}</div></div>"  # box


def clean_flow(cfg: dict, t: dict, n: int, counts: dict) -> str:  # tab 2 cleaning flow
    """Two input boxes, an arrow to the Works box, an arrow to the Artists box; counts holds master, standalone, works and artists as text."""
    color = cfg["design"]["palette"]["highlight"]  # box color
    arrow = "<div class='ui-flow-arrow'></div>"  # CSS arrow pointing down
    inputs = _box(t["t2_in_master"], counts["master"], t["t2_tip_master"], color) + _box(t["t2_in_standalone"].format(n=n), counts["standalone"], t["t2_tip_standalone"], color)  # the two sources
    return f"<div class='ui-flow'><div class='ui-flow-row'>{inputs}</div>{arrow}{_box(t['t2_out'], counts['works'], t['t2_tip_out'], color)}{arrow}{_box(t['t2_artists'], counts['artists'], t['t2_tip_artists'], color)}</div>"  # whole flow

def method_css(cfg: dict) -> str:  # style rules of the tab 4 infographic and model chips
    """Rules prefixed ui-m-; colors from the palette, sizes from design.method."""
    d = cfg["design"]  # design settings
    c, ty, m = d["palette"], d["type"], d["method"]  # palette, type scale and method sizes
    rules = [  # one rule per element
        f".ui-m-row{{display:grid;grid-template-columns:{m['label_w']}px 1fr;align-items:center;margin-bottom:{m['row_gap']}px}}",  # label column and track
        f".ui-m-label{{font-size:{ty['label']}px;color:{c['muted']}}}",  # muted run label
        f".ui-m-row.final .ui-m-label{{color:{c['highlight']};font-weight:700}}",  # accented final label
        f".ui-m-track{{display:grid;height:{m['row_h']}px}}",  # one row of the shared year axis, columns inline in percent
        f".ui-m-train{{background:linear-gradient(to right,transparent,{c['chance']});display:flex;align-items:center;padding-left:{m['text_pad']}px;font-size:{ty['caption']}px;color:{c['muted']};white-space:nowrap;overflow:hidden}}",  # train bar fading to the left edge
        f".ui-m-row.final .ui-m-train{{background:linear-gradient(to right,transparent,{c['highlight']});color:{c['text']}}}",  # final train bar
        f".ui-m-gap{{border:{d['box']['border']}px dashed {c['muted']};box-sizing:border-box}}",  # gap block, outline only
        f".ui-m-test{{line-height:1.1;background:{c['rule']};color:{c['background']};display:flex;flex-direction:column;align-items:center;justify-content:center;font-size:{ty['caption']}px;font-weight:600;border-radius:{m['test_radius']}px}}",  # test block
        f".ui-m-row.final .ui-m-test{{background:{c['highlight']}}}",  # sealed test block
        f".ui-m-sealed{{display:flex;align-items:center;gap:{m['text_pad']}px}}",  # lock and the word Sealed
        f".ui-m-lock{{position:relative;display:inline-block;width:{m['lock_w']}px;height:{m['lock_h']}px;background:{c['background']};border-radius:2px;margin-top:{m['shackle_h']}px}}",  # lock body
        f".ui-m-lock::before{{content:'';position:absolute;left:50%;transform:translateX(-50%);top:-{m['shackle_h']}px;width:{m['lock_w'] - 4}px;height:{m['shackle_h']}px;border:2px solid {c['background']};border-bottom:none;border-radius:{m['lock_w']}px {m['lock_w']}px 0 0;box-sizing:border-box}}",  # lock shackle
        f".ui-m-axis{{position:relative;height:{ty['caption'] * 2}px}}",  # tick row
        f".ui-m-tick{{position:absolute;transform:translateX(-50%);font-size:{ty['caption']}px;color:{c['muted']}}}",  # one year label
        f".ui-m-chips{{display:flex;align-items:flex-end;gap:{d['box']['gap']}px}}",  # model chips with arrows
        f".ui-m-col{{display:flex;flex-direction:column;align-items:flex-start}}",  # chip and its flag
        f".ui-m-flag{{font-size:{ty['caption']}px;font-weight:700;color:{c['highlight']};min-height:{ty['caption'] * 2}px}}",  # Final model flag
        f".ui-m-arrow{{position:relative;width:{m['arrow_len']}px;height:{d['box']['border'] * 2}px;background:{c['muted']};margin-bottom:{m['arrow_lift']}px}}",  # arrow shaft
        f".ui-m-arrow::after{{content:'';position:absolute;right:-1px;top:50%;transform:translateY(-50%);border-top:{m['arrow_head']}px solid transparent;border-bottom:{m['arrow_head']}px solid transparent;border-left:{m['arrow_head']}px solid {c['muted']}}}",  # arrow head
        f".ui-m-inline{{display:flex;align-items:center}}",  # chip and its tooltip icon on one line, centered
        f".ui-m-legend{{display:flex;align-items:center;gap:{d['box']['gap'] * 2}px;font-size:{ty['caption']}px;color:{c['muted']}}}",  # legend row above the card
        f".ui-m-key{{display:inline-flex;align-items:center;gap:{m['text_pad']}px}}",  # swatch and its word
        f".ui-m-sw{{display:inline-block;width:{m['swatch_w']}px;height:{m['swatch_h']}px;box-sizing:border-box}}",  # legend swatch
        f".ui-m-finals{{display:flex;align-items:center;gap:{d['box']['gap']}px;font-size:{ty['label']}px;color:{c['muted']}}}",  # final models line
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
    train = f"<i class='ui-m-sw' style='background:linear-gradient(to right,transparent,{c['chance']})'></i>"  # train swatch, same fade as the bars
    gap = f"<i class='ui-m-sw' style='border:{b}px dashed {c['muted']}'></i>"  # gap swatch, dashed outline
    test = f"<i class='ui-m-sw' style='background:{c['rule']};border-radius:2px'></i>"  # test swatch
    return "<div class='ui-m-legend'>" + key(train, t["t4_legend_train"]) + key(gap, t["t4_legend_gap"], t["t4_tip_gap"]) + key(test, t["t4_legend_test"]) + "</div>"  # legend row

def final_models_html(cfg: dict, tags: list) -> str:  # tab 4 final models line
    """The label "Final models:" and one small tag per final model."""
    t, c = cfg["static_tabs"]["texts"], cfg["design"]["palette"]  # texts and palette
    return "<div class='ui-m-finals'><span>" + html.escape(t["t4_final_models"]) + "</span>" + "".join(chip(tag, c["rule"]) for tag in tags) + "</div>"  # label and tags

def method_chips(cfg: dict, t: dict, names: list) -> str:  # tab 4 model chips
    """Past habits, then the compared models with arrows; the last model is the final one and carries the flag and the row tooltip."""
    pal = cfg["design"]["palette"]  # palette
    items = [(t["t5_col_rule"], pal["muted"], "")] + [(n, pal["rule"], "") for n in names[:-1]] + [(names[-1], pal["highlight"], t["t4_final_flag"])]  # label, color and flag of each chip
    parts = [f"<div class='ui-m-col'><span class='ui-m-flag'>{html.escape(flag)}</span><div class='ui-m-inline'>{chip(text, color)}{info(t['t4_chips_tip']) if flag else ''}</div></div>" for text, color, flag in items]  # one column per chip, the tooltip icon sits right after the final one
    return "<div class='ui-m-chips'>" + "<div class='ui-m-arrow'></div>".join(parts) + "</div>"  # arrows between the chips

def close_css(cfg: dict) -> str:  # style rules of the tab 8 closing cards
    """Rules prefixed ui-close-; colors from the palette, the marker size from design.method."""
    d = cfg["design"]  # design settings
    c, ty, b = d["palette"], d["type"], d["box"]  # palette, type scale and box sizes
    rules = [  # one rule per element
        f".ui-close-title{{display:flex;align-items:center;gap:{b['gap']}px;font-size:{ty['subtitle']}px;font-weight:700;color:{c['text']};margin-bottom:{b['gap'] * 2}px}}",  # marker and card title
        f".ui-close-dot{{width:{d['method']['dot']}px;height:{d['method']['dot']}px;border-radius:50%;background:{c['highlight']};flex:none}}",  # round marker in the highlight color
        f".ui-close-item{{font-size:{ty['label']}px;color:{c['text']};line-height:1.45;margin-bottom:{b['gap']}px}}",  # one item
    ]
    return "<style>" + "".join(rules) + "</style>"  # the style element
