"""Static tabs 1 to 5 and 8: cached file loaders and the HTML of each tab. Texts and settings come from the static_tabs config."""

import json  # result files of the Timing model

import pandas as pd  # tables
import streamlit as st  # cache decorator

from ui_components import card, card_title, chip, headline, kpi_card  # design system helpers
from static_diagrams import clean_svg, flow_svg  # schemas of tabs 1 and 2

from config import CONFIG, ROOT  # project settings and root
from static_charts import steps_chart  # Altair chart of tab 3
from style_tab import validation_figures  # tab 7 test years, reused for tab 8 (read only)


@st.cache_data  # cache the table
def load_csv(path: str) -> pd.DataFrame:  # one CSV file
    """Read a CSV given by a path relative to the project root."""
    return pd.read_csv(ROOT / path)  # cached by Streamlit


@st.cache_data  # cache the dict
def load_json(path: str) -> dict:  # one JSON file
    """Read a JSON file given by a path relative to the project root."""
    with open(ROOT / path, encoding="utf-8") as f:  # result file
        return json.load(f)  # plain dict


def static_path(sc: dict, key: str) -> str:  # path of a file of the static folder
    """Path of one of the four small data files, relative to the project root."""
    return f"{sc['paths']['dir']}/{sc['paths'][key]}"  # folder and file name


def css(cfg: dict) -> str:  # style rules of the static tabs, scoped by the poc-st prefix
    """Style block of the static tabs; colors come from the poc section."""
    c, s = cfg["poc"]["colors"], cfg["static_tabs"]["sizes"]  # colors and sizes
    rules = [  # one rule per element
        f".poc-st-big{{font-size:{s['big_font']}px;font-weight:700}}",  # big number
        f".poc-st-row{{display:flex;gap:{s['block_pad'] * 3}px}}",  # blocks side by side
        f".poc-st-block{{flex:1;border:1px solid {c['card_border']};border-radius:{cfg['poc']['sizes']['card_radius']}px;padding:{s['block_pad']}px}}",  # one block
        f".poc-st-title{{font-weight:700}}",  # block title
        f".poc-st-bar-row{{display:flex;align-items:center;gap:{s['block_pad']}px;margin:4px 0}}",  # one bucket bar line
        f".poc-st-bar-label{{width:{s['bar_label_width']}px}}",  # bucket label
        f".poc-st-bar{{height:{s['bar_height']}px;background:{c['accent']};color:{c['accent_fg']};padding:0 6px;display:flex;align-items:center;box-sizing:border-box}}",  # the bar with its rate
        f".poc-st-seg{{display:flex;align-items:center;justify-content:center;height:{s['segment_height']}px;border:1px solid {c['card_border']};box-sizing:border-box;text-align:center}}",  # timeline segment
        f".poc-st-seg.test{{background:{c['accent']};color:{c['accent_fg']}}}",  # test segment
        ".poc-st-table{border-collapse:collapse;width:100%}",  # table
        f".poc-st-table th,.poc-st-table td{{border-bottom:1px solid {c['card_border']};padding:{cfg['poc']['sizes']['cell_padding']}px {cfg['poc']['sizes']['card_padding']}px;text-align:left}}",  # cells
        f".poc-st-note{{font-size:{cfg['poc']['sizes']['note_font']}px;opacity:{cfg['poc']['sizes']['muted_opacity']}}}",  # small grey text
        f".poc-st-line{{margin-top:{s['block_pad']}px}}",  # one text line
    ]
    return "<style>" + "".join(rules) + "</style>"  # the style element


def figure_values(cfg: dict) -> dict:  # data_figures.csv as a dict
    """metric -> value of data_figures.csv."""
    sc = cfg["static_tabs"]  # settings
    return load_csv(static_path(sc, "data_figures")).set_index("metric")["value"].to_dict()  # metric -> value

def tab1_html(cfg: dict) -> str:  # problem and question
    """Headline, use sentence, the flow schema and the strip of four numbers."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    t, y = sc["texts"], CONFIG["years"]  # texts and split years
    fig = figure_values(cfg)  # artists and works
    tests = load_json(sc["paths"]["when_tests"])["test_rows"]  # test snapshots
    names = sc["models_compared"]  # compared models
    cards = [  # label, value, tooltip
        (t["t1_kpi_artists"], f"{int(fig['artists']):,}", t["t1_tip_artists"]),  # artists
        (t["t1_kpi_works"], f"{int(fig['works']):,}", t["t1_tip_works"]),  # works
        (t["t1_kpi_models"], str(len(names)), t["t1_tip_models"].format(names=", ".join(names))),  # models compared
        (t["t1_kpi_tests"], f"{int(tests):,}", t["t1_tip_tests"].format(first=y["holdout_start"], last=y["holdout_end"])),  # predictions tested
    ]
    strip = "<div class='ui-row'>" + "".join(card(kpi_card(lab, val, "", pal["highlight"], tip)) for lab, val, tip in cards) + "</div>"  # four cards
    return "<div class='ui-stack'>" + headline(t["t1_headline"], t["t1_use"]) + card(flow_svg(cfg, t, cfg["poc"]["texts"])) + strip + "</div>"  # page

def tab2_head_html(cfg: dict) -> str:  # headline of tab 2
    """Headline and the dump date subtitle with its tooltip."""
    sc = cfg["static_tabs"]  # settings
    t = sc["texts"]  # texts
    return headline(t["t2_headline"], t["t2_source"].format(date=sc["dump_date"]), t["t2_tip_sub"])  # block

def works_per_year(cfg: dict) -> pd.DataFrame:  # chart table of tab 2
    """Dated cleaned Works per year, from works_per_year.csv."""
    return load_csv(static_path(cfg["static_tabs"], "works_per_year"))  # year, works

def tab2_chart_title_html(cfg: dict) -> str:  # title line of the chart card
    """Chart title with its tooltip."""
    t = cfg["static_tabs"]["texts"]  # texts
    return card_title(t["t2_chart_title"], t["t2_tip_chart"])  # title

def tab2_right_html(cfg: dict) -> str:  # right blocks of tab 2
    """Two number cards, then the cleaning schema with the counts of both sources."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    t, rw = sc["texts"], sc["rows"]  # texts and row names
    fig = figure_values(cfg)  # counts
    n = CONFIG["works"]["min_standalone_releases"]  # threshold of standalone releases
    kpis = "<div class='ui-row'>" + card(kpi_card(t["t2_works"], f"{int(fig['works']):,}", "", pal["highlight"], t["t2_tip_works"].format(n=n))) + card(kpi_card(t["t2_artists"], f"{int(fig['artists']):,}", "", pal["highlight"], t["t2_tip_artists"])) + "</div>"  # two cards
    counts = {"master": f"{int(fig[rw['metric_master']]):,}", "standalone": f"{int(fig[rw['metric_standalone']]):,}", "works": f"{int(fig['works']):,}"}  # schema numbers
    return "<div class='ui-stack'>" + kpis + card(clean_svg(cfg, t, n, counts)) + "</div>"  # blocks

def tab4_html(cfg: dict) -> str:  # method
    """Timeline of the split, the two question lines and the simple rule line."""
    sc = cfg["static_tabs"]  # settings
    t, p, y = sc["texts"], cfg["poc"]["texts"], CONFIG["years"]  # texts and years
    gap, first, last = y["label_gap"], y["holdout_start"], y["holdout_end"]  # split years
    seg = lambda text, flex, cls="": f"<div class='poc-st-seg {cls}' style='flex:{flex}'>{text}</div>"  # one segment
    line = lambda label, q, m: f"<p class='poc-st-line'>{t['t4_line'].format(label=label, question=q, model=m)}</p>"  # one question line
    timeline = "<div style='display:flex'>" + seg(t["t4_train"].format(last=first - gap - 1), sc["sizes"]["train_flex"]) + seg(t["t4_gap"].format(n=gap), gap) + seg(t["t4_test"].format(first=first, last=last), last - first + 1, "test") + "</div>"  # three segments
    return css(cfg) + timeline + f"<div class='poc-st-note'>{t['t4_test_note']}</div>" + line(p["label_timing"], p["q_timing"], t["t4_model_timing"]) + line(p["label_style"], p["q_style"], t["t4_model_style"]) + f"<p class='poc-st-line'>{t['t4_rule']}</p>"  # page




def tab8_html(cfg: dict) -> str:  # limits and next steps
    """Three short blocks: Data, Style forecast, Next step."""
    t, y = cfg["static_tabs"]["texts"], CONFIG["years"]  # texts and years
    v = validation_figures(cfg)  # tab 7 tested years: first, last, years
    block = lambda title, text: f"<div class='poc-st-block'><div class='poc-st-title'>{title}</div><div>{text}</div></div>"  # one block
    blocks = block(t["t8_data_title"], t["t8_data"]) + block(t["t8_style_title"], t["t8_style"].format(n=v["years"], first=v["first"], last=v["last"])) + block(t["t8_next_title"], t["t8_next"].format(first=y["holdout_start"], last=y["holdout_end"]))  # three blocks
    return css(cfg) + f"<div class='poc-st-row'>{blocks}</div>"  # page


def results_data(cfg: dict) -> pd.DataFrame:  # chart table of tab 5
    """One row per bar: panel, label, value, lo, hi, color, title_color. Chance has no interval."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    t, p = sc["texts"], cfg["poc"]["texts"]  # texts
    pa, rw = sc["paths"], sc["rows"]  # paths and row names
    wm, wb = load_json(pa["when_model"]), load_json(pa["when_baseline"])  # Timing model and rule
    hl = load_csv(pa["what_headline"]).set_index("model")  # Style rows
    m, b = hl.loc[rw["what_model"]], hl.loc[rw["what_rule"]]  # Style model and rule rows
    cols = {  # panel -> (title color, chance, rule (value, lo, hi), model (value, lo, hi))
        p["label_timing"]: (pal["timing"], wm["positive_rate"], (wb["pr_auc"], wb["pr_auc_lo"], wb["pr_auc_hi"]), (wm["pr_auc"], wm["pr_auc_lo"], wm["pr_auc_hi"])),  # Timing
        p["label_style"]: (pal["style"], m["positive_rate"], (b["pr_auc"], b["pr_auc_lo"], b["pr_auc_hi"]), (m["pr_auc"], m["pr_auc_lo"], m["pr_auc_hi"])),  # Style
    }
    rows = []  # one dict per bar
    for panel, (color, chance, rule, model) in cols.items():  # in panel order
        rows.append({"panel": panel, "label": t["t5_col_chance"], "value": chance, "lo": None, "hi": None, "color": pal["chance"], "title_color": color})  # By chance
        rows.append({"panel": panel, "label": t["t5_col_rule"], "value": rule[0], "lo": rule[1], "hi": rule[2], "color": pal["rule"], "title_color": color})  # History only
        rows.append({"panel": panel, "label": t["t5_col_model"], "value": model[0], "lo": model[1], "hi": model[2], "color": color, "title_color": color})  # Our model
    return pd.DataFrame(rows)  # six rows


def models_beat_rule(data: pd.DataFrame, cfg: dict) -> bool:  # headline condition of tab 5
    """True when, for every panel, the lower bound of the model interval is above the upper bound of the rule interval."""
    t = cfg["static_tabs"]["texts"]  # bar labels
    rule = data[data["label"] == t["t5_col_rule"]].set_index("panel")["hi"]  # rule upper bounds
    model = data[data["label"] == t["t5_col_model"]].set_index("panel")["lo"]  # model lower bounds
    return bool((model > rule.reindex(model.index)).all())  # every panel


def tab5_head_html(cfg: dict, data: pd.DataFrame) -> str:  # headline and subtitle of tab 5
    """The conclusion when the data supports it, the neutral headline otherwise, and the subtitle with its tooltip."""
    t, y = cfg["static_tabs"]["texts"], CONFIG["years"]  # texts and years
    text = t["t5_headline"] if models_beat_rule(data, cfg) else t["t5_headline_neutral"]  # headline rule
    return headline(text, t["t5_sub"].format(first=y["holdout_start"], last=y["holdout_end"]), t["t5_info"])  # block


def tab5_chips_html(cfg: dict) -> str:  # the three column labels with tooltips
    """By chance, History only, Our model: one chip each, the tooltip explains the bar."""
    t, pal, p = cfg["static_tabs"]["texts"], cfg["design"]["palette"], cfg["poc"]["texts"]  # texts, palette, labels
    rule = f"{p['label_timing']}: {t['t5_tip_rule_timing']} {p['label_style']}: {t['t5_tip_rule_style']}"  # both baseline definitions
    chips = chip(t["t5_col_chance"], pal["chance"], t["t5_tip_chance"]) + chip(t["t5_col_rule"], pal["rule"], rule) + chip(t["t5_col_model"], pal["text"], t["t5_tip_model"])  # three chips
    return f"<div class='ui-chips' style='margin:0 0 {cfg['design']['box']['gap']}px 0'>{chips}</div>"  # row

def dots_panels(cfg: dict) -> tuple:  # data of the 100-dot card
    """Headline line and the two panels (title, k, exact rate, big number) from top10_lift.csv; k = round(n x rate)."""
    sc = cfg["static_tabs"]  # settings
    t, n = sc["texts"], sc["sizes"]["dots_side"] ** 2  # texts and circles per grid
    top = load_csv(sc["paths"]["what_top10"]).set_index("model").loc[sc["rows"]["what_model"]]  # Style top group row
    rates = [top["holdout_positive_rate"], top["precision_top"]]  # overall rate, top group rate
    ks = [round(n * r) for r in rates]  # circles filled per grid
    titles = [t["t5_dot_left"].format(n=n), t["t5_dot_right"].format(n=n)]  # grid titles
    panels = [{"title": ti, "k": k, "rate_text": f"{r * 100:.1f}%", "big": t["t5_dot_big"].format(k=k, n=n)} for ti, k, r in zip(titles, ks, rates)]  # two panels
    return t["t5_dot_head"].format(n=n, b=ks[0], k=ks[1]), panels  # line and panels

def series_data(cfg: dict) -> list:  # the two series of tab 3
    """For each configured series: its config entry, its table and its color."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    return [(s, load_csv(static_path(sc, s["file"])), pal[s["key"]]) for s in sc["tab3_series"]]  # in config order


def tab3_head_html(cfg: dict) -> str:  # headline of tab 3
    """The conclusion only when both series strictly decrease in the data."""
    t = cfg["static_tabs"]["texts"]  # texts
    falling = all((df[s["rate"]].diff().dropna() < 0).all() for s, df, _ in series_data(cfg))  # strictly decreasing, both
    return headline(t["t3_headline"] if falling else t["t3_headline_neutral"])  # block


def h0_chip_html(cfg: dict, hid: str, color: str) -> str:  # H0 chip of one hypothesis
    """Chip from hypotheses_summary.csv with the tooltip."""
    t = cfg["static_tabs"]["texts"]  # texts
    h = load_csv(static_path(cfg["static_tabs"], "hypotheses")).set_index("id").loc[hid]  # the hypothesis row
    return f"<div class='ui-chips'>{chip(t['t3_col_h0'] if h['h0_rejected'] == 'yes' else t['t3_h0_kept'], color, t['t3_h0_tip'])}</div>"  # chip row


def tab3_panels(cfg: dict) -> list:  # the two cards of tab 3
    """Per card: title, chart and H0 chip html."""
    t = cfg["static_tabs"]["texts"]  # texts
    out = []  # one dict per card
    for s, df, color in series_data(cfg):  # in config order
        out.append({"title": t[s["title"]], "chart": steps_chart(df, s["bucket"], s["rate"], s["count"], color, cfg), "chip": h0_chip_html(cfg, s["hypothesis"], color)})  # card parts
    return out  # two cards


def tab3_chips_html(cfg: dict) -> str:  # slim row for H2 to H5
    """One chip per remaining hypothesis: question and result."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    h = load_csv(static_path(sc, "hypotheses")).set_index("id")  # hypotheses table
    return "<div class='ui-chips'>" + "".join(chip(f"{h.loc[i, 'question']} {h.loc[i, 'result']}", pal["line"]) for i in sc["chip_hypotheses"]) + "</div>"  # chip row
