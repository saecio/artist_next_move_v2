"""Static tabs 1 to 5 and 8: cached file loaders and the HTML of each tab. Texts and settings come from the static_tabs config."""

import html  # escape of artist names
import json  # result files of the Timing model and the stored card values

import pandas as pd  # tables
import streamlit as st  # cache decorator

from ui_components import card, card_title, chip, headline, info, kpi_card  # design system helpers
from static_diagrams import clean_flow  # HTML and CSS flow of tab 2

from config import CONFIG, ROOT  # project settings and root
from static_charts import genre_chart, mini_chart, steps_chart, timeline_chart  # Altair charts of tabs 3, 2 and 1
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

def tab1_head_html(cfg: dict) -> str:  # headline of tab 1
    """Story headline with the cutoff year, and the use sentence."""
    sc = cfg["static_tabs"]  # settings
    return headline(sc["texts"]["t1_headline"].format(year=sc["story"]["cutoff_year"]), sc["texts"]["t1_use"])  # block


def timeline_data(cfg: dict, artist: str) -> pd.DataFrame:  # one row per release year of one artist
    """Dated Works up to the cutoff year without re-recordings: year, releases, title (picked title, else the representative Work)."""
    sc = cfg["static_tabs"]  # settings
    pa, cutoff = sc["paths"], sc["story"]["cutoff_year"]  # paths and cutoff year
    rt = load_csv(pa["route_table"])  # artist names and ids
    aid = int(rt.loc[rt["artist"] == artist, "artist_id"].iloc[0])  # Discogs id
    works = pd.concat([load_csv(f) for f in pa["works"]])  # both pools
    w = works[(works["artist_id"] == aid) & (works["year"] <= cutoff) & ~works["is_rerecording"]]  # this artist, dated up to the cutoff, no re-recordings
    out = w.groupby("year").size().rename("releases").reset_index()  # releases per year
    rep = w[w["is_representative"]].drop_duplicates("year").set_index("year")["title"]  # representative title per year
    pick = load_csv(pa["release_pick"])  # picked titles
    pick = pick[pick["artist_id"] == aid].set_index("year")["title"]  # this artist's picks
    out["title"] = out["year"].map(pick).fillna(out["year"].map(rep)).fillna("")  # picked title when available
    return out  # year, releases, title


def tab1_panels(cfg: dict) -> list:  # the two story cards
    """Per card: title html (artist name in the question color), timeline chart and question html."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    t, st_ = sc["texts"], sc["story"]  # texts and story settings
    cards = ((st_["timing_artist"], pal["timing"], t["t1_q_timing"]), (st_["style_artist"], pal["style"], t["t1_q_style"]))  # artist, color, question
    return [{"title": f"<div class='ui-card-title' style='color:{color}'>{html.escape(name)}</div>", "chart": timeline_chart(timeline_data(cfg, name), st_["cutoff_year"], st_["future_years"], color, t, cfg), "question": f"<div class='ui-label'>{html.escape(q.format(artist=name))}</div>"} for name, color, q in cards]  # two cards


def tab1_kpis_html(cfg: dict) -> str:  # strip of four numbers
    """Hypotheses tested, models compared, predictions tested on unseen years, test years untouched."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    t, y = sc["texts"], CONFIG["years"]  # texts and split years
    names = sc["models_compared"]  # compared models
    tests = load_json(sc["paths"]["when_tests"])["test_rows"]  # test snapshots
    cards = [  # label, value, tooltip
        (t["t1_kpi_hyp"], str(len(load_csv(static_path(sc, "hypotheses")))), t["t1_tip_hyp"]),  # hypotheses
        (t["t1_kpi_models"], str(len(names)), t["t1_tip_models"].format(names=", ".join(names))),  # models
        (t["t1_kpi_tests"], f"{int(tests):,}", t["t1_tip_tests"]),  # predictions
        (t["t1_kpi_years"], t["t1_years_value"].format(first=y["holdout_start"], last=y["holdout_end"]), t["t1_tip_years"]),  # test years
    ]
    return "<div class='ui-row'>" + "".join(card(kpi_card(lab, val, "", pal["highlight"], tip)) for lab, val, tip in cards) + "</div>"  # four cards

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
    """The cleaning flow card, then the chip with the share of Works without a year."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    t, rw = sc["texts"], sc["rows"]  # texts and row names
    fig = figure_values(cfg)  # counts
    n = CONFIG["works"]["min_standalone_releases"]  # threshold of standalone releases
    counts = {"master": f"{int(fig[rw['metric_master']]):,}", "standalone": f"{int(fig[rw['metric_standalone']]):,}", "works": f"{int(fig['works']):,}", "artists": f"{int(fig['artists']):,}"}  # flow numbers
    undated = int(load_csv(static_path(sc, "data_quality")).set_index("metric").loc["works_undated", "value"])  # Works without a year
    pct = f"{undated / int(fig['works']) * 100:.1f}"  # share in percent
    return "<div class='ui-stack'>" + card(clean_flow(cfg, t, n, counts)) + "<div class='ui-chips' style='margin:0'>" + chip(t["t2_undated"].format(pct=pct), pal["muted"], t["t2_tip_undated"]) + "</div></div>"  # flow and chip

def tab4_html(cfg: dict) -> str:  # method
    """Timeline of the split, the two question lines and the rule of thumb line."""
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
    """One row per bar: panel, label, value, lo, hi, color, title_color, subtitle, tag. Chance has no interval, only the model bar has a tag."""
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
    lines = {p["label_timing"]: t["t5_line_timing"], p["label_style"]: t["t5_line_style"]}  # visible line under each panel title
    rows = []  # one dict per bar
    for panel, (color, chance, rule, model) in cols.items():  # in panel order
        base = {"panel": panel, "title_color": color, "subtitle": lines[panel], "tag": None}  # shared fields
        rows.append({**base, "label": t["t5_col_chance"], "value": chance, "lo": None, "hi": None, "color": pal["chance"]})  # By chance
        rows.append({**base, "label": t["t5_col_rule"], "value": rule[0], "lo": rule[1], "hi": rule[2], "color": pal["rule"]})  # Rule of thumb
        rows.append({**base, "label": t["t5_col_model"], "value": model[0], "lo": model[1], "hi": model[2], "color": color, "tag": t["t5_advantage"].format(d=f"{model[0] - rule[0]:.3f}")})  # Our model with its advantage tag
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
    """By chance, Rule of thumb, Our model: one chip each, the tooltip explains the bar."""
    t, pal, p = cfg["static_tabs"]["texts"], cfg["design"]["palette"], cfg["poc"]["texts"]  # texts, palette, labels
    rule = f"{p['label_timing']}: {t['t5_tip_rule_timing']} {p['label_style']}: {t['t5_tip_rule_style']}"  # both baseline definitions
    chips = chip(t["t5_col_chance"], pal["chance"], t["t5_tip_chance"]) + chip(t["t5_col_rule"], pal["rule"], rule) + chip(t["t5_col_model"], pal["text"], t["t5_tip_model"])  # three chips
    return f"<div class='ui-chips' style='margin:0 0 {cfg['design']['box']['gap']}px 0'>{chips}</div>"  # row

def top_share(cfg: dict) -> tuple:  # share of the top group
    """Share of artists in the top group (selected_share of the stage 0 row) and s = round(100 x share)."""
    sc = cfg["static_tabs"]  # settings
    share = float(load_csv(sc["paths"]["what_top10"]).set_index("model").loc[sc["rows"]["what_model"], "selected_share"])  # fraction of rows selected
    return share, round(100 * share)  # fraction and whole percent


def strip_html(cfg: dict) -> str:  # sorted-artists strip of the dot card
    """Muted line, a strip whose first segment (selected_share wide) is in the Style color, end labels and the segment label."""
    t, pal = cfg["static_tabs"]["texts"], cfg["design"]["palette"]  # texts and palette
    share, s = top_share(cfg)  # width and label number
    seg = f"<div class='ui-strip-top' style='width:{share * 100:.2f}%;background:{pal['style']}'></div>"  # highlighted segment
    ends = f"<div class='ui-strip-labels'><span>{html.escape(t['t5_strip_most'])}</span><span>{html.escape(t['t5_strip_least'])}</span></div>"  # end labels
    top = f"<div class='ui-strip-top-label' style='color:{pal['style']}'>{html.escape(t['t5_strip_top'].format(s=s))}</div>"  # segment label
    return f"<div class='ui-caption'>{html.escape(t['t5_dot_sort'])}</div><div class='ui-strip' style='margin-top:{cfg['design']['box']['gap']}px'>{seg}</div>{ends}{top}"  # whole block


def dots_panels(cfg: dict) -> tuple:  # data of the 100-dot card
    """Headline line and the two panels (title, k, exact rate, big number, caption) from top10_lift.csv; k = round(n x rate)."""
    sc = cfg["static_tabs"]  # settings
    t, n = sc["texts"], sc["sizes"]["dots_side"] ** 2  # texts and circles per grid
    top = load_csv(sc["paths"]["what_top10"]).set_index("model").loc[sc["rows"]["what_model"]]  # Style top group row
    rates = [top["holdout_positive_rate"], top["precision_top"]]  # overall rate, top group rate
    ks = [round(n * r) for r in rates]  # circles filled per grid
    s = top_share(cfg)[1]  # whole percent of the top group
    titles = [t["t5_dot_left"], t["t5_dot_right"].format(s=s)]  # grid titles
    panels = [{"title": ti, "k": k, "rate_text": f"{r * 100:.1f}%", "big": t["t5_dot_big"].format(k=k), "caption": t["t5_dot_caption"].format(n=n)} for ti, k, r in zip(titles, ks, rates)]  # two panels
    return t["t5_dot_head"].format(s=s, n=n, b=ks[0], k=ks[1]), panels  # line and panels


def series_data(cfg: dict) -> list:  # the two series of tab 3
    """For each configured series: its config entry, its table and its color."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    return [(s, load_csv(static_path(sc, s["file"])), pal[s["key"]]) for s in sc["tab3_series"]]  # in config order


def tab3_head_html(cfg: dict) -> str:  # headline of tab 3
    """The conclusion only when both series strictly decrease in the data."""
    t = cfg["static_tabs"]["texts"]  # texts
    falling = all((df[s["rate"]].diff().dropna() < 0).all() for s, df, _ in series_data(cfg))  # strictly decreasing, both
    return headline(t["t3_headline"] if falling else t["t3_headline_neutral"])  # block


def tab3_panels(cfg: dict) -> list:  # the two cards of the Two patterns view
    """Per card: title, chart and the muted one-liner under the chart."""
    t = cfg["static_tabs"]["texts"]  # texts
    out = []  # one dict per card
    for s, df, color in series_data(cfg):  # in config order
        out.append({"title": t[s["title"]], "chart": steps_chart(df, s["bucket"], s["rate"], s["count"], color, cfg), "line": f"<div class='ui-caption'>{html.escape(t[s['line']])}</div>"})  # card parts
    return out  # two cards


def view_selector(cfg: dict, tab: str) -> str:  # pill selector of a tab
    """Segmented control with the view labels of the tab (first one is the default); returns the selected label, the default when the pill is cleared."""
    sc = cfg["static_tabs"]  # settings
    labels = [sc["texts"][k] for k in sc["views"][tab]]  # view labels in config order
    return st.segmented_control(tab, labels, default=labels[0], key=sc["view_keys"][tab], label_visibility="collapsed") or labels[0]  # selected label


def genre_share_data(cfg: dict) -> pd.DataFrame:  # chart table of the Genres view
    """Genre share by year, from genre_share_by_year.csv."""
    return load_csv(static_path(cfg["static_tabs"], "genre_share"))  # year, genre, share


def genre_story(cfg: dict) -> dict:  # numbers of the Genres headline
    """The genre with the largest absolute share increase from the story start year to the last year of the file, with its two shares in whole percent."""
    d = genre_share_data(cfg)  # year, genre, share
    y0, y1 = cfg["design"]["genre_story_start_year"], int(d["year"].max())  # first and last year
    wide = d.pivot(index="genre", columns="year", values="share")  # genre x year
    genre = (wide[y1] - wide[y0]).idxmax()  # largest increase
    return {"genre": genre, "a": f"{wide.loc[genre, y0] * 100:.0f}", "b": f"{wide.loc[genre, y1] * 100:.0f}", "y0": y0, "y1": y1}  # headline values


def tab2g_head_html(cfg: dict) -> str:  # headline of the Genres view
    """Headline from the data and the fixed subtitle."""
    t = cfg["static_tabs"]["texts"]  # texts
    return headline(t["t2g_headline"].format(**genre_story(cfg)), t["t2g_sub"])  # block


def tab2g_parts(cfg: dict) -> dict:  # parts of the Genres card
    """Card title html, the chart and the muted note about the years not shown."""
    t = cfg["static_tabs"]["texts"]  # texts
    chart = genre_chart(genre_share_data(cfg), {"year": t["t2g_tip_year"], "genre": t["t2g_tip_genre"], "share": t["t2g_tip_share"]}, cfg)  # stacked areas
    note = f"<div class='ui-caption'>{html.escape(t['t2g_note'].format(year=cfg['design']['incomplete_from_year'] - 1))}</div>"  # last shown year
    return {"title": card_title(t["t2g_title"], t["t2g_tip_title"]), "chart": chart, "note": note}  # card parts


def cards_table(cfg: dict) -> pd.DataFrame:  # cards of the Six tests view
    """hypotheses_cards.csv sorted by effect size, strongest first, with the answer filled from its stored values."""
    d = load_csv(static_path(cfg["static_tabs"], "hyp_cards")).sort_values("effect_size", ascending=False).reset_index(drop=True)  # strongest first
    d["answer_text"] = [a.format(**json.loads(v)) for a, v in zip(d["answer"], d["values"])]  # placeholders filled
    return d  # one row per card


def tab3s_head_html(cfg: dict) -> str:  # headline of the Six tests view
    """Headline with the plain question of the strongest pattern."""
    return headline(cfg["static_tabs"]["texts"]["t3s_headline"].format(question=cards_table(cfg).loc[0, "question"]))  # block


def card_chart_rows(cfg: dict, hid: str) -> pd.DataFrame:  # mini chart rows of one card
    """label, value: H1 and H6 from their bucket files, the others from hypotheses_charts.csv."""
    sc = cfg["static_tabs"]  # settings
    for s in sc["tab3_series"]:  # the two bucket files
        if s["hypothesis"] == hid:  # H1 or H6
            d = load_csv(static_path(sc, s["file"]))  # bucket file
            return pd.DataFrame({"label": d[s["bucket"]].astype(str), "value": d[s["rate"]]})  # label, value
    d = load_csv(static_path(sc, "hyp_charts"))  # H2 to H5
    return d[d["id"] == hid][["label", "value"]].reset_index(drop=True)  # this card's rows


def card_color(cfg: dict, hid: str) -> str:  # color of a card
    """Timing color for H6, Style color for the others."""
    return cfg["design"]["palette"]["timing" if hid == "H6" else "style"]  # question color


def card_parts(cfg: dict, row: pd.Series) -> dict:  # parts of one Six tests card
    """Question html, mini chart, and the html with the answer, the strength bar and the chip."""
    t, color = cfg["static_tabs"]["texts"], card_color(cfg, row["id"])  # texts and card color
    axis = cfg["static_tabs"]["cards"]["items"][row["id"]]["axis"]  # axis title of the mini chart
    tip = t["t3s_tip_strength"] + (" " + t["t3s_tip_h3"] if row["id"] == "H3" else "")  # H3 adds the rank test sentence
    v = float(row["effect_size"])  # strength
    meter = f"<div class='ui-split'><span class='ui-label'>{html.escape(t['t3s_strength'])}{info(tip)}</span><span class='ui-label' style='color:{cfg['design']['palette']['text']};font-weight:700'>{v:.2f}</span></div><div class='ui-meter'><div class='ui-meter-fill' style='width:{v * 100:.1f}%;background:{color}'></div></div>"  # label, number and bar
    chip_row = f"<div class='ui-chips'>{chip(t['t3s_chip'], color)}</div>" if row["h0_rejected"] == "yes" else ""  # only where the null hypothesis was rejected
    return {"question": f"<div class='ui-qtext'>{html.escape(row['question'])}</div>", "chart": mini_chart(card_chart_rows(cfg, row["id"]), color, axis, cfg), "bottom": f"<div class='ui-answer'>{html.escape(row['answer_text'])}</div>{meter}{chip_row}"}  # card parts
