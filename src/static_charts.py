"""Altair charts of the static tabs. Data comes from files, colors and sizes from the design section of the demo config."""

import altair as alt  # charts, ships with Streamlit
import textwrap  # grid title wrapping

from config import CONFIG  # project settings (bootstrap level)

import pandas as pd  # tables

_REGISTERED = []  # names of themes already registered


def register_theme(cfg: dict) -> None:  # one shared dark theme
    """Register and enable the dark Altair theme once: transparent background, no border, faint grid, muted text."""
    d = cfg["design"]  # design settings
    name = d["chart"]["theme_name"]  # theme name
    if name in _REGISTERED:  # already done
        return  # nothing to do
    c, ch = d["palette"], d["chart"]  # palette and chart settings

    @alt.theme.register(name, enable=True)  # registered once
    def demo_dark() -> alt.theme.ThemeConfig:  # theme function
        return {"config": {  # vega-lite config
            "background": "transparent",  # the card shows through
            "font": f"{d['font']['body']}, {d['font']['fallback']}",  # one font in every chart
            "view": {"stroke": None},  # no chart border
            "axis": {"gridColor": c["line"], "gridOpacity": ch["grid_opacity"], "domainColor": c["line"], "tickColor": c["line"], "labelColor": c["muted"], "titleColor": c["muted"], "labelFontSize": ch["axis_font"], "titleFontSize": ch["axis_font"]},  # axes, grid at half the line color
            "axisTop": {"domain": False},  # no domain line on top
            "axisRight": {"domain": False},  # no domain line on the right
            "title": {"color": c["text"], "fontSize": ch["title_font"]},  # titles
            "legend": {"labelColor": c["text"], "titleColor": c["muted"], "labelFontSize": ch["axis_font"], "titleFontSize": ch["axis_font"]},  # legends
            "range": {"category": [c["timing"], c["style"], c["rule"], c["chance"], c["highlight"]]},  # no default blue anywhere
        }}

    _REGISTERED.append(name)  # remember it


def panel_chart(df: pd.DataFrame, title: str, subtitle: str, title_color: str, colors: dict, cfg: dict, top: float) -> alt.LayerChart:  # one results panel
    """Three bars with value labels, interval whiskers, a muted line under the title and a tag on the model bar; `df` has label, value, lo, hi, tag, score_txt, range_txt columns. The width follows the container."""
    c, ch, t = cfg["design"]["palette"], cfg["design"]["chart"], cfg["static_tabs"]["texts"]  # palette, chart settings and texts
    order = list(df["label"])  # bar order
    y = alt.Y("value:Q", axis=None, scale=alt.Scale(domain=[0, top]))  # hidden shared y axis, the labels carry the numbers
    x = alt.X("label:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=ch["label_font"], grid=False))  # bar names
    range_title = t["t5_tt_range"].format(ci=round(CONFIG["bootstrap"]["confidence"] * 100))  # "95% range", level from the bootstrap setting
    name_score = [alt.Tooltip("label:N", title=t["t5_tt_name"]), alt.Tooltip("score_txt:N", title=t["t5_tt_score"])]  # friendly rows: bar name and score
    base = alt.Chart(df).encode(x=x)  # shared encoding
    has_range = alt.expr.isValid(alt.datum.lo)  # bars with an interval (past habits and model)
    color = alt.Color("label:N", scale=alt.Scale(domain=order, range=[colors[k] for k in order]), legend=None)  # bar colors
    bar_mark = dict(cornerRadiusTopLeft=ch["bar_corner"], cornerRadiusTopRight=ch["bar_corner"])  # rounded tops
    bars_plain = base.mark_bar(**bar_mark).encode(y=y, color=color, tooltip=name_score).transform_filter(~has_range)  # by chance: name and score only
    bars_range = base.mark_bar(**bar_mark).encode(y=y, color=color, tooltip=name_score + [alt.Tooltip("range_txt:N", title=range_title)]).transform_filter(has_range)  # name, score and 95% range
    whisk = base.mark_rule(color=c["text"], strokeWidth=ch["whisker_width"]).encode(y=alt.Y("lo:Q", axis=None, scale=alt.Scale(domain=[0, top])), y2="hi:Q", tooltip=name_score + [alt.Tooltip("range_txt:N", title=range_title)]).transform_filter(has_range)  # interval, rows with one only
    text = base.mark_text(dy=-8, baseline="bottom", color=c["text"], fontSize=ch["label_font"]).encode(y=alt.Y("label_y:Q", axis=None, scale=alt.Scale(domain=[0, top])), text=alt.Text("value:Q", format=".3f"), tooltip=alt.value(None))  # value on top
    tag = base.mark_text(align="right", dx=ch["tag_dx"], dy=-ch["tag_dy"], baseline="bottom", color=title_color, fontSize=ch["axis_font"]).encode(y=alt.Y("label_y:Q", axis=None, scale=alt.Scale(domain=[0, top])), text="tag:N", tooltip=alt.Tooltip("tag_tip:N", title=" ")).transform_filter(alt.expr.isValid(alt.datum.tag))  # advantage tag above the model value, with its tooltip
    return alt.layer(bars_plain, bars_range, whisk, text, tag).properties(title=alt.TitleParams(title, subtitle=subtitle, subtitleColor=c["muted"], subtitleFontSize=ch["axis_font"], color=title_color, fontSize=ch["title_font"], anchor="start"), height=ch["height_results"], width="container")  # panel


def results_chart(data: pd.DataFrame, cfg: dict) -> list:  # one chart per question, the caller puts them in equal columns
    """Two panels from the results table, one shared y scale; each panel fills the width of its column."""
    register_theme(cfg)  # shared theme
    ch = cfg["design"]["chart"]  # chart settings
    d = data.copy()  # keep the caller's table
    d["label_y"] = d["hi"].fillna(d["value"])  # labels sit above the whisker
    top = float(d["label_y"].max()) * ch["y_headroom_results"]  # one scale for both panels
    panels = []  # one chart per question
    for name, part in d.groupby("panel", sort=False):  # in table order
        colors = dict(zip(part["label"], part["color"]))  # label -> bar color
        panels.append(panel_chart(part.reset_index(drop=True), name, part["subtitle"].iloc[0], part["title_color"].iloc[0], colors, cfg, top))  # panel
    return panels  # Timing then Style


def steps_chart(df: pd.DataFrame, bucket: str, rate: str, count: str, color: str, cfg: dict) -> alt.LayerChart:  # one bar per bucket
    """Bars in file order (descending), the rate printed above each bar, one color."""
    register_theme(cfg)  # shared theme
    ch = cfg["design"]["chart"]  # chart settings
    d = df[[bucket, rate, count]].copy()  # only the columns used
    d.columns = ["bucket", "rate", "count"]  # fixed names
    d["rate_pct"] = d["rate"] * 100  # percent for display
    d["rate_label"] = d["rate_pct"].map(lambda v: f"{v:.1f}%")  # text printed on the bar
    order = list(d["bucket"])  # file order
    x = alt.X("bucket:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=ch["label_font"], grid=False))  # bucket axis
    y = alt.Y("rate_pct:Q", axis=None, scale=alt.Scale(domain=[0, float(d["rate_pct"].max()) * ch["y_headroom"]]))  # hidden y axis
    tip = [alt.Tooltip("bucket:N", title="bucket"), alt.Tooltip("rate_pct:Q", title="rate %", format=".2f"), alt.Tooltip("count:Q", title="rows", format=",")]  # plain values
    base = alt.Chart(d).encode(x=x, y=y, tooltip=tip)  # shared encoding
    bars = base.mark_bar(color=color, cornerRadiusTopLeft=ch["bar_corner"], cornerRadiusTopRight=ch["bar_corner"])  # bars
    text = base.mark_text(dy=-8, baseline="bottom", color=cfg["design"]["palette"]["text"], fontSize=ch["label_font"]).encode(text="rate_label:N")  # rate on each bar
    return alt.layer(bars, text).properties(height=ch["height_steps"])  # chart


def dots_chart(panels: list, side: int, texts: dict, cfg: dict) -> alt.HConcatChart:  # two grids of circles
    """One side x side grid per panel; the first k circles use the Style color, the rest the chance color; a big number and a caption sit under each grid."""
    register_theme(cfg)  # shared theme
    c, ch = cfg["design"]["palette"], cfg["design"]["chart"]  # palette and chart settings
    cell, n = ch["dot_cell"], side * side  # cell size in px and circles per grid
    yes, no = texts["yes"], texts["no"]  # legend-free state names
    depth = side + 3.5  # grid rows plus the two text lines
    charts = []  # one chart per panel
    for p in panels:  # in the given order
        grid = pd.DataFrame({"col": [i % side + 1 for i in range(n)], "row": [i // side + 1 for i in range(n)], "state": [yes if i < p["k"] else no for i in range(n)], "rate": p["rate_text"]})  # reading order, first k filled
        x = alt.X("col:Q", axis=None, scale=alt.Scale(domain=[0.5, side + 0.5]))  # hidden columns axis
        y = alt.Y("row:Q", axis=None, scale=alt.Scale(domain=[0.5, depth], reverse=True))  # hidden rows axis, row 1 on top
        dots = alt.Chart(grid).mark_circle(size=(cell * ch["dot_fill"]) ** 2, opacity=1).encode(x=x, y=y, color=alt.Color("state:N", scale=alt.Scale(domain=[yes, no], range=[c["style"], c["chance"]]), legend=None), tooltip=[alt.Tooltip("rate:N", title=texts["tip"])])  # circles
        mid = (side + 1) / 2  # horizontal center
        big = alt.Chart(pd.DataFrame({"x": [mid], "y": [side + 1.6], "text": [p["big"]]})).mark_text(fontSize=ch["dot_big_font"], fontWeight="bold", color=c["style"]).encode(x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0.5, side + 0.5])), y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0.5, depth], reverse=True)), text="text:N")  # big number
        cap = alt.Chart(pd.DataFrame({"x": [mid], "y": [side + 2.9], "text": [p["caption"]]})).mark_text(fontSize=ch["dot_caption_font"], color=c["muted"]).encode(x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0.5, side + 0.5])), y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[0.5, depth], reverse=True)), text="text:N")  # caption
        title = alt.TitleParams(textwrap.wrap(p["title"], ch["dot_wrap"]), color=c["text"], fontSize=ch["axis_font"] + 1, anchor="middle")  # wrapped grid title
        layers = [dots, big, cap]  # grid, big number, caption
        charts.append(alt.layer(*layers).properties(title=title, width=cell * side, height=cell * depth))  # one grid
    return alt.hconcat(*charts)  # side by side


def works_chart(df: pd.DataFrame, incomplete_from: int, texts: dict, cfg: dict) -> alt.LayerChart:  # Works per year
    """Smooth area with a thin line, a labelled peak, a shaded band for the years still being filled in, and a tooltip."""
    register_theme(cfg)  # shared theme
    c, ch = cfg["design"]["palette"], cfg["design"]["chart"]  # palette and chart settings
    d = df[["year", "works"]].copy()  # only the columns used
    peak = d.loc[d["works"].idxmax()]  # peak year and value, from the data
    first, last = int(d["year"].min()), int(d["year"].max())  # year range
    x = alt.X("year:Q", axis=alt.Axis(format="d", labelAngle=0, labelFontSize=ch["axis_font"], title=None, grid=False, tickMinStep=1), scale=alt.Scale(domain=[first, last], nice=False))  # years
    y = alt.Y("works:Q", axis=alt.Axis(format="~s", title=None, labelFontSize=ch["axis_font"]), scale=alt.Scale(domain=[0, float(d["works"].max()) * ch["y_headroom"]]))  # works
    area = alt.Chart(d).mark_area(color=c["highlight"], opacity=ch["area_opacity"], interpolate="monotone", line=alt.OverlayMarkDef(color=c["highlight"], strokeWidth=ch["line_width"], opacity=1)).encode(x=x, y=y)  # area and thin line
    hover = alt.Chart(d).mark_point(opacity=0, size=ch["peak_size"]).encode(x=x, y=y, tooltip=[alt.Tooltip("year:Q", title=cfg["static_tabs"]["texts"]["t1_tip_year"], format="d"), alt.Tooltip("works:Q", title=cfg["static_tabs"]["texts"]["t1_tip_count"], format=",")])  # invisible points carry the tooltip
    band_data = pd.DataFrame({"x0": [incomplete_from], "x1": [last], "label": [texts["incomplete"]], "tip": [texts["band_tip"].format(year=int(peak["year"]))]})  # shaded years
    band = alt.Chart(band_data).mark_rect(color=c["text"], opacity=ch["band_opacity"]).encode(x=alt.X("x0:Q", scale=alt.Scale(domain=[first, last], nice=False)), x2="x1:Q", tooltip=alt.Tooltip("tip:N", title=" "))  # band with the late submissions sentence
    note = alt.Chart(band_data).mark_text(align="right", baseline="top", dx=-6, color=c["muted"], fontSize=ch["axis_font"]).encode(x=alt.X("x0:Q", scale=alt.Scale(domain=[first, last], nice=False)), y=alt.value(4), text="label:N")  # band label
    peak_data = pd.DataFrame({"year": [int(peak["year"])], "works": [int(peak["works"])], "label": [texts["peak"].format(year=int(peak["year"]), works=f"{int(peak['works']):,}")]})  # peak point
    dot = alt.Chart(peak_data).mark_point(filled=True, color=c["highlight"], stroke=c["text"], strokeWidth=1.5, size=ch["peak_size"], opacity=1).encode(x=x, y=y)  # peak marker
    tag = alt.Chart(peak_data).mark_text(align="right", dx=-10, dy=-4, color=c["text"], fontSize=ch["label_font"]).encode(x=x, y=y, text="label:N")  # peak label, left of the point so it stays inside the plot
    return alt.layer(band, area, hover, dot, tag, note).properties(height=ch["height_works"])  # chart


def timeline_chart(df: pd.DataFrame, cutoff: int, future: int, color: str, texts: dict, cfg: dict) -> alt.LayerChart:  # one artist's releases up to the cutoff
    """One dot per release year (size = releases), a rule at the cutoff labelled with texts['t1_here'], then a dashed line ending in a large question mark. Nothing after the cutoff is drawn from data."""
    register_theme(cfg)  # shared theme
    c, ch = cfg["design"]["palette"], cfg["design"]["chart"]  # palette and chart settings
    first, end = int(df["year"].min()) - 1, cutoff + future  # x range: one empty year before the first release, empty years after the cutoff
    dom = alt.Scale(domain=[first, end], nice=False)  # shared x scale
    y = alt.Y("lane:Q", axis=None, scale=alt.Scale(domain=[0, 1]))  # hidden lane axis
    x = alt.X("year:Q", axis=alt.Axis(format="d", labelAngle=0, labelFontSize=ch["axis_font"], title=None, grid=False, tickMinStep=1), scale=dom)  # years
    tip = [alt.Tooltip("year:Q", title=texts["t1_tip_year"], format="d"), alt.Tooltip("releases:Q", title=texts["t1_tip_count"]), alt.Tooltip("title:N", title=texts["t1_tip_title"])]  # year, releases, picked title
    line = alt.Chart(pd.DataFrame({"x0": [first], "x1": [cutoff], "lane": [0.5]})).mark_rule(color=c["line"]).encode(x=alt.X("x0:Q", scale=dom, axis=None), x2="x1:Q", y=y)  # past line
    dots = alt.Chart(df.assign(lane=0.5)).mark_circle(color=color, opacity=1).encode(x=x, y=y, size=alt.Size("releases:Q", scale=alt.Scale(range=[ch["story_dot_min"], ch["story_dot_max"]]), legend=None), tooltip=tip)  # one dot per year
    here = alt.Chart(pd.DataFrame({"x": [cutoff], "top": [1.0], "lane": [0.5]})).mark_rule(color=c["text"], strokeWidth=ch["whisker_width"], y2Offset=cfg["design"]["story"]["marker_below"]).encode(x=alt.X("x:Q", scale=dom, axis=None), y=alt.Y("top:Q", scale=alt.Scale(domain=[0, 1]), axis=None), y2="lane:Q")  # vertical rule at the cutoff, from the top to a little below the timeline line so it stops above the year labels
    here_label = alt.Chart(pd.DataFrame({"x": [cutoff], "text": [texts["t1_here"]], "lane": [0.95]})).mark_text(align="right", dx=-6, color=c["text"], fontSize=ch["axis_font"]).encode(x=alt.X("x:Q", scale=dom, axis=None), y=y, text="text:N")  # label of the rule
    future_line = alt.Chart(pd.DataFrame({"x0": [cutoff], "x1": [end - 0.5], "lane": [0.5]})).mark_rule(color=color, strokeDash=[5, 5], strokeWidth=ch["whisker_width"]).encode(x=alt.X("x0:Q", scale=dom, axis=None), x2="x1:Q", y=y)  # dashed line into the unknown
    mark = alt.Chart(pd.DataFrame({"x": [end], "text": ["?"], "lane": [0.5]})).mark_text(color=color, fontSize=ch["story_glyph_font"], fontWeight="bold").encode(x=alt.X("x:Q", scale=dom, axis=None), y=y, text="text:N")  # large question mark
    years = sorted({int(df["year"].min()), cutoff})  # first data year and the marker year only, the last data year would overlap the marker label
    ticks = alt.Chart(pd.DataFrame({"x": years, "text": [str(v) for v in years], "lane": 0.25})).mark_text(color=c["muted"], fontSize=ch["axis_font"], dy=cfg["design"]["story"]["label_dy"]).encode(x=alt.X("x:Q", scale=dom, axis=None), y=y, text="text:N")  # muted years below the line, lowered by label_dy
    return alt.layer(line, here, here_label, future_line, dots, mark, ticks).properties(height=ch["height_story"])  # chart


def genre_chart(df: pd.DataFrame, texts: dict, cfg: dict) -> alt.Chart:  # genre share by year
    """Normalized stacked area of the genre shares, one palette color per genre, a legend on the right that highlights a genre on click, and a tooltip."""
    register_theme(cfg)  # shared theme
    d, ch = cfg["design"], cfg["design"]["chart"]  # design and chart settings
    names = list(d["genre_palette"])  # genre names in palette order, which is the stack order
    data = df[["year", "genre", "share"]]  # only the columns used
    pick = alt.selection_point(fields=["genre"], bind="legend")  # click a legend entry to highlight a genre
    x = alt.X("year:Q", axis=alt.Axis(format="d", labelAngle=0, labelFontSize=ch["axis_font"], title=None, grid=False, tickMinStep=1), scale=alt.Scale(domain=[int(data["year"].min()), int(data["year"].max())], nice=False))  # years
    y = alt.Y("share:Q", stack="normalize", axis=alt.Axis(format="%", title=None, labelFontSize=ch["axis_font"]), scale=alt.Scale(domain=[0, 1]))  # shares, each year adds up to 100%
    color = alt.Color("genre:N", scale=alt.Scale(domain=names, range=[d["genre_palette"][n] for n in names]), legend=alt.Legend(orient="right", title=None, symbolType="square"))  # palette colors, legend on the right
    tip = [alt.Tooltip("year:Q", title=texts["year"], format="d"), alt.Tooltip("genre:N", title=texts["genre"]), alt.Tooltip("share:Q", title=texts["share"], format=".1%")]  # year, genre and share
    area = alt.Chart(data).mark_area(interpolate="monotone").encode(x=x, y=y, color=color, order=alt.Order("genre:N", sort="ascending"), opacity=alt.condition(pick, alt.value(1), alt.value(ch["genre_opacity_off"])), tooltip=tip).add_params(pick)  # stacked areas
    return area.properties(height=ch["height_genres"])  # chart


def strength_ticks(ch: dict) -> list:  # tick values of the Six tests axis
    """Axis values from 0 to strength_axis_max in steps of strength_axis_step, so no label repeats and the grid sits on the ticks."""
    n = round(ch["strength_axis_max"] / ch["strength_axis_step"])  # number of steps
    return [round(k * ch["strength_axis_step"], 10) for k in range(n + 1)]  # 0, step, 2 x step, up to the maximum

def strength_chart(df: pd.DataFrame, cfg: dict) -> alt.LayerChart:  # one bar per hypothesis
    """Horizontal bars sorted by effect size, the label on the left, the number at the end of each bar, a tooltip with the question and the answer."""
    register_theme(cfg)  # shared theme
    c, ch = cfg["design"]["palette"], cfg["design"]["chart"]  # palette and chart settings
    t = cfg["static_tabs"]["texts"]  # texts
    d = df[["short_label", "effect_size", "color", "question", "answer_text", "strength_text"]]  # only the columns used
    y = alt.Y("short_label:N", sort=list(d["short_label"]), title=None, axis=alt.Axis(labelFontSize=ch["label_font"], labelLimit=ch["strength_label_limit"], labelColor=c["text"], grid=False, ticks=False, domain=False))  # labels, strongest first
    x = alt.X("effect_size:Q", title=None, scale=alt.Scale(domain=[0, ch["strength_axis_max"]], nice=False), axis=alt.Axis(labelFontSize=ch["axis_font"], grid=True, format=".1f", values=strength_ticks(ch)))  # strength axis with a light grid
    tip = [alt.Tooltip("question:N", title=t["t3s_tt_question"]), alt.Tooltip("answer_text:N", title=t["t3s_tt_answer"]), alt.Tooltip("strength_text:N", title=t["t3s_tt_strength"])]  # question, answer, strength
    base = alt.Chart(d).encode(y=y, x=x, tooltip=tip)  # shared encoding
    bars = base.mark_bar(size=ch["strength_bar"], cornerRadiusTopRight=ch["bar_corner"], cornerRadiusBottomRight=ch["bar_corner"]).encode(color=alt.Color("color:N", scale=None))  # bars, color from the data
    text = base.mark_text(align="left", dx=6, color=c["text"], fontSize=ch["label_font"]).encode(text=alt.Text("effect_size:Q", format=".2f"))  # number at the end of the bar
    return alt.layer(bars, text).properties(height=ch["height_strength"])  # chart
