"""Depth views of tab 5: Year by year, Models compared, Can you trust the score? Data, headline conditions and Altair charts."""

import html  # escape of the muted lines

import altair as alt  # charts, ships with Streamlit
import pandas as pd  # tables

from config import CONFIG  # project settings
from static_charts import register_theme  # shared light paper theme
from static_tabs import load_csv, load_json, static_path  # cached readers
from ui_components import card, card_title, chip, headline  # design system helpers


def pad_domain(values: pd.Series, pad: float) -> list:  # axis range around the data
    """[min, max] of the values widened by `pad` of their range on each side."""
    lo, hi = float(values.min()), float(values.max())  # data range
    span = (hi - lo) or 1.0  # never a zero span
    return [lo - pad * span, hi + pad * span]  # padded domain


def question_colors(cfg: dict) -> dict:  # question label -> color
    """The Timing and Style labels with their palette colors."""
    p, pal = cfg["poc"]["texts"], cfg["design"]["palette"]  # labels and palette
    return {p["label_timing"]: pal["timing"], p["label_style"]: pal["style"]}  # label -> color


# Year by year

def year_data(cfg: dict) -> pd.DataFrame:  # score of each test year
    """question, line, year, score: our model and past habits, Timing from the holdout JSON files, Style from by_year.csv (no 2023 rows)."""
    sc = cfg["static_tabs"]  # settings
    t, p, pa, rw = sc["texts"], cfg["poc"]["texts"], sc["paths"], sc["rows"]  # texts, labels, paths, row names
    rows = []  # one dict per point
    for line, key in ((t["t5_col_model"], "when_model"), (t["t5_col_rule"], "when_baseline")):  # Timing files
        for year, v in load_json(pa[key])["by_year"].items():  # one entry per test year
            rows.append({"question": p["label_timing"], "line": line, "year": int(year), "score": v["pr_auc"]})  # Timing point
    by_year = load_csv(pa["what_by_year"])  # Style rows
    for line, model in ((t["t5_col_model"], rw["what_model"]), (t["t5_col_rule"], rw["what_rule"])):  # Style model and past habits
        for r in by_year[by_year["model"] == model].itertuples():  # one row per test year
            rows.append({"question": p["label_style"], "line": line, "year": int(r.year), "score": r.pr_auc})  # Style point
    d = pd.DataFrame(rows)  # all points
    d["score_txt"] = d["score"].map("{:.3f}".format)  # tooltip text
    return d  # table of the charts


def ahead_every_year(d: pd.DataFrame, cfg: dict) -> bool:  # headline condition of Year by year
    """True when, for both questions, the model score is above past habits in every year the question has."""
    t = cfg["static_tabs"]["texts"]  # line names
    wide = d.pivot_table(index=["question", "year"], columns="line", values="score")  # model and past habits side by side
    return bool((wide[t["t5_col_model"]] > wide[t["t5_col_rule"]]).all())  # every question and year


def years_head_html(cfg: dict, d: pd.DataFrame) -> str:  # headline of Year by year
    """The conclusion when the data supports it, the neutral headline otherwise."""
    t = cfg["static_tabs"]["texts"]  # texts
    return headline(t["t5y_headline"] if ahead_every_year(d, cfg) else t["t5y_headline_neutral"])  # block


def legend_html(cfg: dict, keys: tuple) -> str:  # chips of the chart lines
    """The chips of the Results view named by `keys` (t5_col_* text keys), same tooltips."""
    t, pal = cfg["static_tabs"]["texts"], cfg["design"]["palette"]  # texts, palette
    tips = {"t5_col_rule": f"{cfg['poc']['texts']['label_timing']}: {t['t5_tip_rule_timing']} {cfg['poc']['texts']['label_style']}: {t['t5_tip_rule_style']}", "t5_col_model": t["t5_tip_model"]}  # tooltips as in the Results view
    colors = {"t5_col_rule": pal["rule"], "t5_col_model": pal["text"]}  # chip borders as in the Results view
    chips = "".join(chip(t[k], colors[k], tips[k]) for k in keys)  # one chip each
    return f"<div class='ui-chips' style='margin:0 0 {cfg['design']['box']['gap']}px 0'>{chips}</div>"  # row


def years_charts(d: pd.DataFrame, cfg: dict) -> list:  # one line chart per question
    """Two charts: a solid line in the question color for our model, a muted line for past habits, points with tooltips."""
    register_theme(cfg)  # shared theme
    t, pal, ch = cfg["static_tabs"]["texts"], cfg["design"]["palette"], cfg["design"]["chart"]  # texts, palette, chart settings
    charts = []  # in question order
    for question, color in question_colors(cfg).items():  # Timing then Style
        part = d[d["question"] == question]  # this question
        order = [t["t5_col_rule"], t["t5_col_model"]]  # past habits under, our model on top
        col = alt.Color("line:N", scale=alt.Scale(domain=order, range=[pal["rule"], color]), legend=None)  # line colors
        base = alt.Chart(part).encode(x=alt.X("year:O", title=t["t5y_x"], axis=alt.Axis(labelAngle=0)), y=alt.Y("score:Q", title=t["t5y_y"], scale=alt.Scale(domain=pad_domain(part["score"], ch["axis_pad"]))), color=col, tooltip=[alt.Tooltip("year:O", title=t["t5y_tip_year"]), alt.Tooltip("score_txt:N", title=t["t5y_tip_score"]), alt.Tooltip("line:N", title=t["t5y_tip_line"])])  # shared encoding
        line = base.mark_line(strokeWidth=ch["line_width"])  # the two lines
        points = base.mark_point(filled=True, size=ch["point_size"], opacity=1)  # one point per year
        charts.append(alt.layer(line, points).properties(title=alt.TitleParams(question, color=color, fontSize=ch["title_font"], anchor="start"), height=ch["height_years"], width="container"))  # chart
    return charts  # Timing, Style


# Models compared

def models_data(cfg: dict) -> pd.DataFrame:  # rows of models_compared.csv
    """The stored table with the range text and the tooltip fields."""
    sc = cfg["static_tabs"]  # settings
    d = load_csv(static_path(sc, "models_compared")).copy()  # question, model, mean_score, ci_low, ci_high, source
    d["score_txt"] = d["mean_score"].map("{:.3f}".format)  # tooltip text
    d["range_txt"] = [sc["texts"]["t5_range_fmt"].format(lo=f"{lo:.3f}", hi=f"{hi:.3f}") for lo, hi in zip(d["ci_low"], d["ci_high"])]  # interval text
    return d  # table of the charts


def trees_tie(d: pd.DataFrame, cfg: dict) -> bool:  # headline condition of Models compared
    """True when, for each question, the spread among the three tree models is below their smallest 95% half-width and Logistic Regression is below all three."""
    sc = cfg["static_tabs"]  # settings
    for question in d["question"].unique():  # both questions
        part = d[d["question"] == question].set_index("model")  # this question
        if not set(sc["tree_models"]) | {sc["simple_model"]} <= set(part.index):  # a model is missing
            return False  # the claim cannot be checked
        trees = part.loc[sc["tree_models"]]  # the tree models
        half = ((trees["ci_high"] - trees["ci_low"]) / 2).min()  # smallest half-width
        spread = trees["mean_score"].max() - trees["mean_score"].min()  # spread of the means
        if not (spread < half and part.loc[sc["simple_model"], "mean_score"] < trees["mean_score"].min()):  # tie and the simple model below
            return False  # claim not supported
    return True  # both questions


def models_head_html(cfg: dict, d: pd.DataFrame) -> str:  # headline of Models compared
    """The conclusion when the data supports it, the neutral headline otherwise."""
    t = cfg["static_tabs"]["texts"]  # texts
    return headline(t["t5m_headline"] if trees_tie(d, cfg) else t["t5m_headline_neutral"])  # block


def models_title_html(cfg: dict, question: str) -> str:  # card title with the tooltip
    """Card title for one question with the info tooltip."""
    t = cfg["static_tabs"]["texts"]  # texts
    return card_title(f"{t['t5m_title']}: {question}", t["t5m_tip"])  # title line


def models_domain(part: pd.DataFrame, cfg: dict) -> list:  # x range of a dot plot
    """Padded range of the models and the past habits line of one question."""
    return pad_domain(part["mean_score"], cfg["design"]["chart"]["axis_pad"])  # padded range


def models_note_html(cfg: dict, d: pd.DataFrame, question: str) -> str:  # zoom caption of a dot plot
    """The muted zoom caption when the axis minimum is above zero, else nothing."""
    if models_domain(d[d["question"] == question], cfg)[0] <= 0:  # the axis reaches zero
        return ""  # no caption
    return f"<div class='ui-caption'>{html.escape(cfg['static_tabs']['texts']['t5m_zoom'])}</div>"  # muted caption


def models_charts(d: pd.DataFrame, cfg: dict) -> list:  # one dot plot per question
    """Horizontal dot plot: one row per model sorted best first, a dashed line for past habits, XGBoost in the question color and the rest muted."""
    register_theme(cfg)  # shared theme
    sc, pal, ch = cfg["static_tabs"], cfg["design"]["palette"], cfg["design"]["chart"]  # settings, palette, chart settings
    t = sc["texts"]  # texts
    charts = []  # in question order
    for question, color in question_colors(cfg).items():  # Timing then Style
        part = d[d["question"] == question]  # this question
        dots = part[part["model"] != t["t5_col_rule"]].sort_values("mean_score", ascending=False).copy()  # models, best first
        dots["color"] = [color if m == sc["final_model"] else pal["rule"] for m in dots["model"]]  # final model highlighted
        rule = part[part["model"] == t["t5_col_rule"]]  # past habits row
        dom = models_domain(part, cfg)  # x range
        x = alt.X("mean_score:Q", title=t["t5m_x"], scale=alt.Scale(domain=dom), axis=alt.Axis(tickCount=5))  # score axis
        tip = [alt.Tooltip("model:N", title=t["t5m_tip_model"]), alt.Tooltip("score_txt:N", title=t["t5m_tip_score"]), alt.Tooltip("range_txt:N", title=t["t5_tt_range"].format(ci=round(100 * CONFIG["bootstrap"]["confidence"])))]  # model, mean score, range
        points = alt.Chart(dots).mark_point(filled=True, size=ch["point_size"], opacity=1).encode(x=x, y=alt.Y("model:N", sort=list(dots["model"]), title=None), color=alt.Color("color:N", scale=None), tooltip=tip)  # one dot per model
        line = alt.Chart(rule).mark_rule(strokeDash=[4, 4], color=pal["rule"], strokeWidth=ch["line_width"]).encode(x=x, tooltip=[alt.Tooltip("model:N", title=t["t5m_tip_model"]), alt.Tooltip("score_txt:N", title=t["t5m_tip_score"])])  # dashed past habits line
        label = alt.Chart(rule).mark_text(align="left", dx=4, dy=-4, baseline="bottom", color=pal["muted"], fontSize=ch["axis_font"]).encode(x=x, y=alt.value(0), text="model:N", tooltip=alt.value(None))  # name of the dashed line at the top
        charts.append(alt.layer(line, label, points).properties(height=ch["height_models"], width="container"))  # chart
    return charts  # Timing, Style


# Can you trust the score?

def reliability_data(cfg: dict) -> pd.DataFrame:  # ten bins per question
    """question, bin, mean_score, observed_rate, ci_low, ci_high, tip, color from the reliability files."""
    sc, t = cfg["static_tabs"], cfg["static_tabs"]["texts"]  # settings, texts
    colors = question_colors(cfg)  # label -> color
    parts = []  # one table per question
    for (question, color), key in zip(colors.items(), ("calibration_timing", "calibration_style")):  # Timing then Style
        d = load_csv(sc["paths"][key]).copy()  # reliability table
        d["question"], d["color"] = question, color  # question and its color
        d["tip"] = [t["t5t_tip_point"].format(score=round(100 * s), observed=round(100 * o)) for s, o in zip(d["mean_score"], d["observed_rate"])]  # tooltip text
        parts.append(d)  # keep
    return pd.concat(parts, ignore_index=True)  # both questions


def top_bin(d: pd.DataFrame, question: str) -> pd.Series:  # highest-score bin of a question
    """Row of the highest bin of `question`."""
    part = d[d["question"] == question]  # this question
    return part.loc[part["bin"].idxmax()]  # last bin


def bin_gap(row: pd.Series) -> float:  # gap of a bin
    """Absolute gap between the mean score and the observed rate."""
    return abs(float(row["mean_score"]) - float(row["observed_rate"]))  # gap


def trust_head_html(cfg: dict, d: pd.DataFrame) -> str:  # headline of the calibration view
    """The forecast headline with the top-bin subtitle when the highest Style bin gap is under design.calibration.max_headline_gap, the neutral headline otherwise."""
    t, p, cal = cfg["static_tabs"]["texts"], cfg["poc"]["texts"], cfg["design"]["calibration"]  # texts, labels, thresholds
    row = top_bin(d, p["label_style"])  # highest Style bin
    if bin_gap(row) < cal["max_headline_gap"]:  # small gap
        return headline(t["t5t_headline"], t["t5t_sub"].format(n=cal["group_size"], s=round(100 * row["mean_score"]), o=round(cal["group_size"] * row["observed_rate"])))  # headline and subtitle with the bin numbers
    return headline(t["t5t_headline_neutral"])  # fallback


def trust_read_html(cfg: dict) -> str:  # title and text of the reading card
    """Reading card content: the weather example, numbers from design.calibration."""
    t, cal = cfg["static_tabs"]["texts"], cfg["design"]["calibration"]  # texts, example numbers
    text = t["t5t_read"].format(pct=cal["example_pct"], days=cal["example_days"], total=cal["example_total"])  # example sentence
    return f"<div class='ui-card-title'>{html.escape(t['t5t_read_title'])}</div><div class='ui-read'>{html.escape(text)}</div>"  # title and text in body style


def trust_right_html(cfg: dict, d: pd.DataFrame) -> str:  # the three blocks of the right column
    """How to read it, the two average gaps as big numbers (Style, Timing) and, when the highest Timing bin gap is above design.calibration.note_gap, the Timing note; stacked, filling the height of the chart card."""
    t, p, gaps = cfg["static_tabs"]["texts"], cfg["poc"]["texts"], cfg["poc"]["gap_text"]  # texts, labels, gap texts
    targets, pal = cfg["static_tabs"]["calibration_targets"], cfg["design"]["palette"]  # question -> summary target, palette
    big = lambda target, label, color: f"<div class='ui-trust-gap'><div class='ui-big' style='color:{color}'>{html.escape(gaps[targets[target]])}</div><div class='ui-label'>{html.escape(label)}</div></div>"  # one gap with its word under it
    blocks = [f"<div class='ui-card'>{trust_read_html(cfg)}</div>", card(f"<div class='ui-trust-gaps'>{big('style', p['label_style'], pal['style'])}{big('timing', p['label_timing'], pal['timing'])}</div>", t["t5t_gap_title"])]  # reading card, gaps card
    if bin_gap(top_bin(d, p["label_timing"])) > cfg["design"]["calibration"]["note_gap"]:  # Timing runs off at the top
        blocks.append(f"<div class='ui-card ui-trust-note'><div class='ui-read'>{html.escape(t['t5t_note'])}</div></div>")  # note card
    return f"<div class='ui-trust-right'>{''.join(blocks)}</div>"  # three stacked blocks

def trust_css(cfg: dict) -> str:  # style rules of the Can you trust view
    """Rules prefixed ui-trust-: equal height columns, the chart as wide as its card (about two to one), the right blocks stacked; below 1024 px the right blocks go under the chart."""
    d = cfg["design"]  # design settings
    c, b, ch = d["palette"], d["box"], d["chart"]  # palette, box and chart settings
    rules = [  # one rule per element
        f".st-key-ui-trust [data-testid='stHorizontalBlock']{{align-items:stretch}}",  # both columns as tall as the taller one
        f".st-key-ui-trust [data-testid='stColumn']:last-child [data-testid='stElementContainer'],.st-key-ui-trust [data-testid='stColumn']:last-child [data-testid='stHtml']{{height:100%}}",  # the right blocks can fill the column
        f".ui-trust-right{{display:flex;flex-direction:column;gap:{b['gap']}px;height:100%}}",  # three blocks with a gap
        f".st-key-ui-trust [data-testid='stColumn']{{display:flex;flex-direction:column}}",  # columns as flex columns
        f".st-key-ui-trust [data-testid='stColumn']>[data-testid='stVerticalBlock']{{flex:1 1 auto}}",  # the column content fills the column
        f".st-key-ui-trust [data-testid='stColumn']>[data-testid='stVerticalBlock']>*{{flex:1 1 auto}}",  # the card (left) and the blocks (right) take the full height, so both end on the same edge
        f".ui-trust-right>.ui-card{{flex:1 1 auto}}",  # the blocks share the height
        f".st-key-ui-trust [data-testid='stColumn']:last-child>[data-testid='stVerticalBlock']{{display:flex;flex-direction:column}}",  # the right column is a flex column
        f".st-key-ui-trust [data-testid='stColumn']:last-child [data-testid='stElementContainer']{{flex:1 1 auto}}",  # its only element grows to the chart card height
        f".ui-trust-gaps{{display:flex;justify-content:space-around;gap:{b['gap'] * 2}px;flex-wrap:wrap}}",  # the two big numbers side by side
        f".ui-trust-gap{{text-align:center}}",  # a number with its word under it
        f".ui-trust-note{{border-left:3px solid {c['highlight']}}}",  # note card with a left rule in the highlight color
        f".st-key-ui-trust-chart [data-testid='stVegaLiteChart']{{width:100%;height:min({ch['calibration_height']}px,calc(100vh - 430px)) !important;min-height:220px}}",  # wide plot that fills the card width
        f".st-key-ui-trust-chart .vega-embed{{width:100%;height:100%}}",  # the chart fills its box
        f"@media (max-width:1023px){{.st-key-ui-trust [data-testid='stColumn']{{flex:1 1 100% !important;min-width:100% !important}}.ui-trust-right{{height:auto}}}}",  # tablet and phone: the right blocks under the chart
    ]
    return "<style>" + "".join(rules) + "</style>"  # the style element

def trust_legend_html(cfg: dict) -> str:  # chips of the reliability chart
    """Timing, Style and Perfect chips."""
    p, t, pal = cfg["poc"]["texts"], cfg["static_tabs"]["texts"], cfg["design"]["palette"]  # labels, texts, palette
    chips = chip(p["label_timing"], pal["timing"]) + chip(p["label_style"], pal["style"]) + chip(t["t5t_perfect"], pal["rule"])  # three chips
    return f"<div class='ui-chips' style='margin:0 0 {cfg['design']['box']['gap']}px 0'>{chips}</div>"  # row


def trust_chart(d: pd.DataFrame, cfg: dict) -> alt.LayerChart:  # reliability chart
    """Dashed diagonal, per question a thin line through ten points with Wilson whiskers; equal axes from 0 to 1 in a wide plot."""
    register_theme(cfg)  # shared theme
    t, pal, ch = cfg["static_tabs"]["texts"], cfg["design"]["palette"], cfg["design"]["chart"]  # texts, palette, chart settings
    scale = alt.Scale(domain=[0, 1])  # same range on both axes
    axis = alt.Axis(format="%", tickCount=5)  # percent ticks
    x = alt.X("mean_score:Q", title=t["t5t_x"], scale=scale, axis=axis)  # the model's score
    y = alt.Y("observed_rate:Q", title=t["t5t_y"], scale=scale, axis=axis)  # how often it happened
    col = alt.Color("color:N", scale=None)  # color per row
    tip = alt.Tooltip("tip:N", title=" ")  # friendly sentence
    base = alt.Chart(d).encode(x=x, y=y, color=col)  # shared encoding
    diag = alt.Chart(pd.DataFrame({"a": [0.0, 1.0]})).mark_line(strokeDash=[5, 5], color=pal["rule"], strokeWidth=ch["line_width"]).encode(x=alt.X("a:Q", scale=scale), y=alt.Y("a:Q", scale=scale))  # perfect line
    line = base.mark_line(strokeWidth=ch["thin_line"]).encode(detail="question:N", order="bin:Q")  # thin connecting line
    whisk = base.mark_rule(strokeWidth=ch["thin_line"]).encode(y=alt.Y("ci_low:Q", scale=scale), y2="ci_high:Q", tooltip=tip)  # Wilson interval
    points = base.mark_point(filled=True, size=ch["point_size"], opacity=1).encode(tooltip=tip)  # the ten points
    return alt.layer(diag, line, whisk, points).properties(width="container", height="container")  # chart that fills its square box


def trust_title_html(cfg: dict) -> str:  # card title with the tooltip
    """Card title and info tooltip."""
    t = cfg["static_tabs"]["texts"]  # texts
    return card_title(t["t5t_title"], t["t5t_tip"])  # title line
