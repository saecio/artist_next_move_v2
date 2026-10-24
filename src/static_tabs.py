"""Static tabs 1 to 5 and 8: cached file loaders and the HTML of each tab. Texts and settings come from the static_tabs config."""

import base64  # photo as a data URI
from datetime import datetime  # dump date parsing
import html  # escape of artist names
import io  # in-memory image file
import json  # result files of the Timing model and the stored card values
import re  # model names compared without spaces or case

import pandas as pd  # tables
from PIL import Image, ImageOps  # artist photo crop and resize
import streamlit as st  # cache decorator

from ui_components import card, card_title, chip, headline, info  # design system helpers
from static_diagrams import clean_flow, close_css, final_models_html, dots_html, funnel_css, funnel_flow, inputs_card, inputs_css, method_card, method_chips, method_css, method_legend  # HTML and CSS schemas of tabs 2, 4 and 8

from config import CONFIG, ROOT  # project settings and root
from static_charts import genre_chart, short_number, steps_chart, strength_chart, timeline_chart  # Altair charts of tabs 3, 2 and 1
from backtesting import method_spans  # fold and final split years
from models import GENRES, NUMERIC_FEATURES, WHAT_FEATURES, make_what_xgboost, make_xgboost  # builders of the Timing and Style final models (their names give the model names), and their input lists
from what_target import ABLATION_GROUPS  # Style input families
from app_data import all_artists  # demo artists, for the name of the example artist
from style_forecast import FINAL_MODEL  # final model of the style forecast
from style_tab import forecast_beats_repeat, validation_figures  # tab 7 test years, reused for tab 8 (read only)


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


def figure_values(cfg: dict) -> dict:  # data_figures.csv as a dict
    """metric -> value of data_figures.csv."""
    sc = cfg["static_tabs"]  # settings
    return load_csv(static_path(sc, "data_figures")).set_index("metric")["value"].to_dict()  # metric -> value

def tab1_head_html(cfg: dict) -> str:  # headline of tab 1
    """Story headline with the cutoff year, and the use sentence."""
    sc = cfg["static_tabs"]  # settings
    return headline(sc["texts"]["t1_headline"].format(year=sc["story"]["cutoff_year"]), sc["texts"]["t1_use"])  # block


def timeline_data(cfg: dict, artist: str) -> pd.DataFrame:  # one row per release year of one artist
    """Dated Works up to story.last_release_year without re-recordings: year, releases, title (picked title, else the representative Work)."""
    sc = cfg["static_tabs"]  # settings
    pa, last = sc["paths"], sc["story"]["last_release_year"]  # paths and last year drawn
    rt = load_csv(pa["route_table"])  # artist names and ids
    aid = int(rt.loc[rt["artist"] == artist, "artist_id"].iloc[0])  # Discogs id
    works = pd.concat([load_csv(f) for f in pa["works"]])  # both pools
    w = works[(works["artist_id"] == aid) & (works["year"] <= last) & ~works["is_rerecording"]]  # this artist, dated up to the last year drawn, no re-recordings
    out = w.groupby("year").size().rename("releases").reset_index()  # releases per year
    rep = w[w["is_representative"]].drop_duplicates("year").set_index("year")["title"]  # representative title per year
    pick = load_csv(pa["release_pick"])  # picked titles
    pick = pick[pick["artist_id"] == aid].set_index("year")["title"]  # this artist's picks
    out["title"] = out["year"].map(pick).fillna(out["year"].map(rep)).fillna("")  # picked title when available
    return out  # year, releases, title


@st.cache_data  # cache the encoded photo
def photo_uri(path: str, width: int, height: int, focus: str) -> str:  # one photo as a data URI
    """Photo file (path relative to the project root) cropped around `focus` to the column shape, at twice the shown size, as a JPEG data URI."""
    img = ImageOps.exif_transpose(Image.open(ROOT / path)).convert("RGB")  # upright, three channels
    centering = tuple(float(v.strip("%")) / 100 for v in focus.split())  # "50% 20%" as fractions
    img = ImageOps.fit(img, (2 * width, 2 * height), centering=centering)  # crop to the column shape, faces kept
    buf = io.BytesIO()  # in-memory file
    img.save(buf, "JPEG", quality=85)  # compressed
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()  # inline image


def credit_row(cfg: dict, name: str) -> dict:  # credits.csv row of an artist
    """The row of `name` in the credits file as a dict, empty when the artist has none."""
    d = load_csv(cfg["static_tabs"]["paths"]["artist_credits"])  # artist, file, author, license, license_url, source_url
    hit = d[d["artist"] == name]  # this artist
    return hit.iloc[0].to_dict() if len(hit) else {}  # row or nothing


def photo_path(cfg: dict, row: dict) -> str:  # path of an existing photo
    """Photo path relative to the project root when the file of the credits row exists, else an empty string."""
    if not row or pd.isna(row.get("file")):  # no row or no file name
        return ""  # no photo
    path = f"{cfg['static_tabs']['paths']['artist_photos']}/{row['file']}"  # folder and file name
    return path if (ROOT / path).is_file() else ""  # only when the file exists


def monogram(name: str) -> str:  # initials of a name
    """First letters of the first two words, upper case."""
    return "".join(w[0] for w in name.split()[:2]).upper()  # initials


def artist_photo_html(cfg: dict, name: str, color: str) -> str:  # photo column of a card
    """The photo filling its column with the credit on its bottom left, or the initials large on the artist color at low opacity (no credit) when there is no photo."""
    ph = cfg["design"]["photo"]  # photo settings
    box = f"width:100%;max-width:{ph['width']}px;height:{ph['height']}px;border-radius:{ph['radius']}px"  # shared column shape
    path = photo_path(cfg, credit_row(cfg, name))  # existing photo, if any
    if path:  # the file exists
        img = f"<img class='ui-photo' src='{photo_uri(path, ph['width'], ph['height'], ph['focus'])}' alt='{html.escape(name, quote=True)}' style='{box};object-fit:cover;object-position:{ph['focus']};display:block;border:1px solid {cfg['design']['palette']['line']};box-sizing:border-box'>"  # embedded photo
        return f"<div class='ui-photo-wrap' style='max-width:{ph['width']}px;padding-bottom:{ph['bottom_pad']}px'>{img}{credit_html(cfg, name)}</div>"  # photo with its credit line under it
    return f"<div class='ui-mono' style='{box};position:relative;overflow:hidden;display:flex;align-items:center;justify-content:center;font-size:{ph['monogram_font']}px;font-weight:700;color:{color}'><div style='position:absolute;inset:0;background:{color};opacity:{ph['monogram_opacity']}'></div><span style='position:relative'>{html.escape(monogram(name))}</span></div>"  # initials column


def artist_name_html(cfg: dict, name: str, color: str) -> str:  # artist name line
    """The artist name in the question color at the configured size."""
    return f"<div class='ui-card-title' style='color:{color};font-size:{cfg['design']['photo']['name_font']}px'>{html.escape(name)}</div>"  # name


def credit_html(cfg: dict, name: str) -> str:  # credit line under the photo
    """One muted line: "Photo: {author}, {license}", or "Photo: {license}" without an author; nothing without a photo or without a license."""
    row = credit_row(cfg, name)  # credits row
    filled = lambda k: not pd.isna(row.get(k)) and bool(str(row.get(k)).strip())  # a field with text
    if not photo_path(cfg, row) or not filled("license"):  # no photo or no license
        return ""  # no credit
    t, ph, pal = cfg["static_tabs"]["texts"], cfg["design"]["photo"], cfg["design"]["palette"]  # texts, photo settings and palette
    text = t["t1_credit"].format(author=row["author"], license=row["license"]) if filled("author") else t["t1_credit_noauthor"].format(license=row["license"])  # with or without the author
    return f"<div class='ui-credit' title='{html.escape(text, quote=True)}' style='margin-top:{ph['credit_margin']}px;min-height:{ph['credit_line']}px;line-height:{ph['credit_line']}px;font-size:{ph['credit_font']}px;font-weight:400;color:{pal['muted']};opacity:{ph['credit_opacity']};overflow-wrap:anywhere'>{html.escape(text)}</div>"  # wraps instead of truncating, left aligned


def tab1_panels(cfg: dict) -> list:  # the two story cards
    """Per card: photo column html, name html, timeline chart, question html (main text color, bottom padding)."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    t, st_ = sc["texts"], sc["story"]  # texts and story settings
    cards = ((st_["timing_artist"], pal["timing"], t["t1_q_timing"]), (st_["style_artist"], pal["style"], t["t1_q_style"]))  # artist, color, question
    return [{"photo": artist_photo_html(cfg, name, color), "title": artist_name_html(cfg, name, color), "chart": timeline_chart(timeline_data(cfg, name), st_["cutoff_year"], st_["future_years"], color, t, cfg), "question": f"<div style='font-size:{cfg['design']['photo']['question_font']}px;color:{pal['text']}'>{html.escape(q.format(artist=name))}</div>", } for name, color, q in cards]  # two cards


def story_halves(cfg: dict) -> list:  # numbers of story card 3
    """Per question (label, color, caption, k, b): top 10% rate and positive rate as whole numbers of n. Style reuses the dot chart panels of tab 5, Timing takes the last of the equal bins of the reliability file and the positive rate of the Timing holdout (the By chance bar)."""
    sc, pal, p = cfg["static_tabs"], cfg["design"]["palette"], cfg["poc"]["texts"]  # settings, palette and labels
    t, n = sc["texts"], sc["sizes"]["dots_side"] ** 2  # texts and artists per group
    style = dots_panels(cfg)[1]  # left grid is the random pick, right grid the top group
    rel = load_csv(sc["paths"]["calibration_timing"])  # ten bins of equal size
    top = rel.sort_values("bin").iloc[-1]  # highest-score bin
    chance = load_json(sc["paths"]["when_model"])["positive_rate"]  # same value as the By chance bar of tab 5
    return [(p["label_timing"], pal["timing"], t["t1_s3_cap_timing"].format(s=top_share(cfg)[1]), round(n * top["observed_rate"]), round(n * chance)), (p["label_style"], pal["style"], t["t1_s3_cap_style"].format(s=top_share(cfg)[1]), style[1]["k"], style[0]["k"])]  # Timing then Style

def tab2_headline_text(cfg: dict) -> str:  # headline text of tab 2
    """The peak headline when the peak year is before design.incomplete_from_year and every later year is below it, else the neutral headline."""
    t = cfg["static_tabs"]["texts"]  # texts
    d = works_per_year(cfg)  # year, works
    peak = d.loc[d["works"].idxmax()]  # peak year and value, from the data
    late = d[d["year"] >= cfg["design"]["incomplete_from_year"]]  # the years still being filled in
    if int(peak["year"]) < cfg["design"]["incomplete_from_year"] and not late.empty and (late["works"] < peak["works"]).all():  # peak is earlier and the last years are below it
        return t["t2_headline"].format(peak_year=int(peak["year"]))  # conclusion
    return t["t2_headline_neutral"]  # fallback

def dump_date_text(cfg: dict) -> str:  # dump date with the month name
    """The dump date of the config (day-month-year) as "1 September 2026"."""
    d = datetime.strptime(cfg["static_tabs"]["dump_date"], "%d-%m-%Y")  # parsed date
    return f"{d.day} {d.strftime('%B')} {d.year}"  # day, English month name, year

def tab2_head_html(cfg: dict) -> str:  # headline of tab 2
    """Headline and the dump date subtitle."""
    sc = cfg["static_tabs"]  # settings
    t = sc["texts"]  # texts
    return headline(tab2_headline_text(cfg), t["t2_source"].format(date=dump_date_text(cfg)))  # block

def works_per_year(cfg: dict) -> pd.DataFrame:  # chart table of tab 2
    """Dated cleaned Works per year, from works_per_year.csv."""
    return load_csv(static_path(cfg["static_tabs"], "works_per_year"))  # year, works

def tab2_chart_title_html(cfg: dict) -> str:  # title line of the chart card
    """Chart title."""
    t = cfg["static_tabs"]["texts"]  # texts
    return card_title(t["t2_chart_title"])  # title

def tab2_right_html(cfg: dict) -> str:  # right blocks of tab 2
    """The cleaning flow card, then the chip with the share of Works without a year."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    t, rw = sc["texts"], sc["rows"]  # texts and row names
    fig = figure_values(cfg)  # counts
    n = CONFIG["works"]["min_standalone_releases"]  # threshold of standalone releases
    counts = {"master": short_number(fig[rw['metric_master']]), "standalone": short_number(fig[rw['metric_standalone']]), "works": short_number(fig['works']), "artists": short_number(fig['artists'])}  # flow numbers
    undated = int(load_csv(static_path(sc, "data_quality")).set_index("metric").loc["works_undated", "value"])  # Works without a year
    pct = f"{undated / int(fig['works']) * 100:.1f}"  # share in percent
    return "<div class='ui-fill-right'>" + card(clean_flow(cfg, t, n, counts)) + "<div class='ui-chips' style='margin:0'>" + chip(t["t2_undated"].format(pct=pct), pal["muted"], t["t2_tip_undated"]) + "</div></div>"  # flow and chip

def final_models(cfg: dict) -> list:  # the three final-model tags of tab 4
    """Texts "Timing: X", "Style: X", "Style forecast (tab 7): X"; X is the models_compared entry that matches the name of the code building each final model."""
    sc = cfg["static_tabs"]  # settings
    t = sc["texts"]  # texts
    flat = lambda name: re.sub(r"[^a-z0-9]", "", name.lower())  # name without spaces, case or underscores
    shown = {flat(n): n for n in sc["models_compared"]}  # flat name -> name shown on the chips
    keys = (make_xgboost.__name__.removeprefix("make_"), make_what_xgboost.__name__.removeprefix("make_what_"), FINAL_MODEL)  # Timing builder, Style builder, forecast model
    return [t[k].format(model=shown[flat(key)]) for k, key in zip(("t4_final_timing", "t4_final_style", "t4_final_forecast"), keys)]  # three texts

def tab4_html(cfg: dict) -> str:  # method
    """Headline, the legend and the time infographic of the five practice runs and the final exam, one caption, the model chips and the final models."""
    sc = cfg["static_tabs"]  # settings
    t = sc["texts"]  # texts
    runs = method_spans()  # years of the folds and of the final split, from backtesting
    card_html = method_card(cfg, t, runs, runs[-1]["test_end"])  # six rows on one year axis
    caption = t["t4_caption"]  # one line under the card
    chips = method_chips(cfg, t, sc["models_compared"])  # release counts and the compared models
    finals = final_models_html(cfg, final_models(cfg))  # label and the three final-model tags
    return method_css(cfg) + "<div class='ui-stack'>" + headline(t["t4_headline"]) + method_legend(cfg, t) + card_html + f"<div class='ui-caption'>{html.escape(caption)} {info(t['t4_caption_tip'])}</div>" + chips + finals + "</div>"  # page

def tab8_html(cfg: dict) -> str:  # limits and next steps
    """Closing slide: three equal cards with a marker, a title and up to four items; numbers come from the tab 5 tags, the tab 1 card and the tab 7 test years."""
    from depth_views import models_data, trees_tie  # imported here because depth_views imports this module
    t, p = cfg["static_tabs"]["texts"], cfg["poc"]["texts"]  # texts and Timing / Style labels
    res = results_data(cfg)  # the table of tab 5, read by the closing block
    tim, sty = story_halves(cfg)  # Timing and Style numbers of the tab 1 third card: label, color, caption, top picks, random
    n = cfg["static_tabs"]["sizes"]["dots_side"] ** 2  # artists per group
    values = {"n": n, "t": tim[3], "tb": tim[4], "s": sty[3], "sb": sty[4], "m": n - sty[3]}  # every placeholder of the items
    supported = {("works", 1): bool(panel_beats_rule(res, cfg).all()), ("works", 2): tim[3] > tim[4], ("works", 3): sty[3] > sty[4]}  # line 1 needs the model clearly above release counts, lines 2 and 3 the highest-rated artists above random picking
    cards = ""  # html of the three cards
    for key in ("works", "mind", "next"):  # left to right
        texts = [t[f"t8_{key}_{i}"].format(**values) for i in range(1, 5) if f"t8_{key}_{i}" in t and supported.get((key, i), True)]  # items, as many as the config has (up to four)
        if key == "next" and not trees_tie(models_data(cfg), cfg):  # the tie claim is not supported by the data
            texts[0] = t["t8_next_1_plain"]  # first item without the tie clause
        items = "".join(f"<div class='ui-close-item'>{html.escape(x)}</div>" for x in texts)  # one-line items
        tip = info(t["t8_works_tip"]) if key == "works" else ""  # the What works title explains "highest rated" and "random"
        cards += card(f"<div class='ui-close-title'><span class='ui-close-dot'></span>{html.escape(t[f't8_{key}_title'])}{tip}</div>{items}")  # one card
    return close_css(cfg) + "<div class='ui-stack'>" + headline(t["t8_headline"]) + f"<div class='ui-row'>{cards}</div>" + closing_html(cfg, res) + "</div>"  # page


def closing_html(cfg: dict, res: pd.DataFrame) -> str:  # closing block of tab 8
    """Three blocks (Timing, Style, Trends) with a name, a question and a verdict chip each, then one sentence; res is the tab 5 table."""
    t, p, pal = cfg["static_tabs"]["texts"], cfg["poc"]["texts"], cfg["design"]["palette"]  # texts, Timing / Style labels, palette
    beats = panel_beats_rule(res, cfg)  # per question: model clearly above release counts
    status = lambda ok, yes: t[yes] if ok else t["t8_close_tested"]  # the positive status only when the data supports it
    trend = t["t8_close_trend_beats"] if forecast_beats_repeat(cfg) else t["t8_close_first"]  # forecast against repeating last year
    blocks = [("timing", status(beats[p["label_timing"]], "t8_close_beats"), pal["timing"]), ("style", status(beats[p["label_style"]], "t8_close_beats"), pal["style"]), ("trends", trend, pal["highlight"])]  # key, status, color
    row = "".join(f"<div class='ui-close-block'><div class='ui-close-name'>{html.escape(t[f't8_close_{k}_name'])}</div><div class='ui-close-q'>{html.escape(t[f't8_close_{k}_q'])}</div>{chip(s, c)}</div>" for k, s, c in blocks)  # name, question and verdict chip each
    return f"<div class='ui-band'><div class='ui-close-blocks'>{row}</div><div class='ui-close-item'>{html.escape(t['t8_close_line'])}</div></div>"  # band with the chips and the final sentence


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
        d = round(100 * (model[0] - rule[0]))  # advantage in whole points
        base = {"panel": panel, "title_color": color, "subtitle": lines[panel], "tag": None, "tag_tip": None, "d": None}  # shared fields
        rows.append({**base, "label": t["t5_col_chance"], "value": chance, "lo": None, "hi": None, "color": pal["chance"], "score_txt": f"{chance:.3f}", "range_txt": None})  # By chance
        rows.append({**base, "label": t["t5_col_rule"], "value": rule[0], "lo": rule[1], "hi": rule[2], "color": pal["rule"], "score_txt": f"{rule[0]:.3f}", "range_txt": t["t5_range_fmt"].format(lo=f"{rule[1]:.3f}", hi=f"{rule[2]:.3f}")})  # Release counts
        rows.append({**base, "label": t["t5_col_model"], "value": model[0], "lo": model[1], "hi": model[2], "color": color, "score_txt": f"{model[0]:.3f}", "range_txt": t["t5_range_fmt"].format(lo=f"{model[1]:.3f}", hi=f"{model[2]:.3f}"), "d": str(d), "tag": t["t5_advantage"].format(d=d), "tag_tip": t["t5_tip_advantage"]})  # Our model with its advantage tag
    return pd.DataFrame(rows)  # six rows


def panel_beats_rule(data: pd.DataFrame, cfg: dict) -> pd.Series:  # per panel condition of tab 5
    """Per panel: True when the lower bound of the model interval is above the upper bound of the rule interval."""
    t = cfg["static_tabs"]["texts"]  # bar labels
    rule = data[data["label"] == t["t5_col_rule"]].set_index("panel")["hi"]  # rule upper bounds
    model = data[data["label"] == t["t5_col_model"]].set_index("panel")["lo"]  # model lower bounds
    return model > rule.reindex(model.index)  # one flag per panel


def models_beat_rule(data: pd.DataFrame, cfg: dict) -> bool:  # headline condition of tab 5
    """True when, for every panel, the lower bound of the model interval is above the upper bound of the rule interval."""
    return bool(panel_beats_rule(data, cfg).all())  # every panel


def tab5_head_html(cfg: dict, data: pd.DataFrame) -> str:  # headline and subtitle of tab 5
    """The conclusion when the data supports it, the neutral headline otherwise, and the subtitle with its tooltip."""
    t, y = cfg["static_tabs"]["texts"], CONFIG["years"]  # texts and years
    text = t["t5_headline"] if models_beat_rule(data, cfg) else t["t5_headline_neutral"]  # headline rule
    return headline(text, t["t5_sub"].format(first=y["holdout_start"], last=y["holdout_end"]), t["t5_info"])  # block


def tab5_chips_html(cfg: dict) -> str:  # the three column labels with tooltips
    """By chance, Release counts, Our model: one chip each, the tooltip explains the bar."""
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
        out.append({"key": s["key"], "title": t[s["title"]], "chart": steps_chart(df, s["bucket"], s["rate"], s["count"], color, cfg), "line": f"<div class='ui-caption'>{html.escape(t[s['line']])}</div>"})  # card parts
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
    return {"title": card_title(t["t2g_title"]), "chart": chart, "note": note}  # card parts


def cards_table(cfg: dict) -> pd.DataFrame:  # rows of the Six tests chart
    """hypotheses_cards.csv with the short labels, sorted by effect size (strongest first), the answer filled from its stored values, the strength text and the bar color."""
    sc, pal = cfg["static_tabs"], cfg["design"]["palette"]  # settings and palette
    d = load_csv(static_path(sc, "hyp_cards")).merge(load_csv(static_path(sc, "hyp_labels")), on="id")  # cards with their short labels
    d = d.sort_values("effect_size", ascending=False).reset_index(drop=True)  # strongest first
    d["answer_text"] = [a.format(**json.loads(v)) for a, v in zip(d["answer"], d["values"])]  # placeholders filled
    d["strength_text"] = [f"{e:.2f} ({m})" + (" " + sc["texts"]["t3s_tip_h3"] if i == "H3" else "") for e, m, i in zip(d["effect_size"], d["effect_measure"], d["id"])]  # strength with its measure, H3 adds the rank test sentence
    d["color"] = pal["muted"]  # every bar neutral
    d.loc[0, "color"] = pal["highlight"]  # the strongest bar (first row) in the highlight color
    return d  # one row per hypothesis

def tab3s_head_html(cfg: dict) -> str:  # headline and muted line of the Six tests view
    """Headline with the short label of the strongest pattern, then the muted line with the honesty tooltip."""
    t = cfg["static_tabs"]["texts"]  # texts
    line = f"<div class='ui-caption'>{html.escape(t['t3s_line'])}{info(t['t3s_tip_honesty'])}</div>"  # muted line and info tooltip
    return headline(t["t3s_headline"].format(label=cards_table(cfg).loc[0, "short_label"])) + line  # block

@st.cache_data  # cache the table
def load_funnel(path: str) -> pd.DataFrame:  # cleaning_funnel.csv
    """Read the hand-typed funnel file (its first line is a # comment) given by a path relative to the project root."""
    return pd.read_csv(ROOT / path, comment="#")  # step, lane, rows_in, rows_out, source_line

def tab2c_html(cfg: dict) -> str:  # Cleaning view of tab 2
    """Headline, the two-lane cleaning flow and the chip with the releases without a year; every number comes from cleaning_funnel.csv."""
    sc = cfg["static_tabs"]  # settings
    t = sc["texts"]  # texts
    out = load_funnel(static_path(sc, "funnel")).set_index("step")["rows_out"]  # step -> rows left
    v = {k: short_number(n) for k, n in out.items()}  # counts as 2.5M or 120K
    flow = card(funnel_flow(cfg, t, v))  # flow card
    undated = chip(t["t2c_undated"].format(n=v["undated"]), cfg["design"]["palette"]["muted"])  # releases without a year
    return funnel_css(cfg) + "<div class='ui-stack'>" + headline(t["t2c_headline"], html.escape(t["t2c_sub"])) + flow + f"<div class='ui-chips' style='margin:0'>{undated}</div></div>"  # page

@st.cache_data  # cache the row
def example_row(path: str, artist_id: int, year: int, year_col: str, order: str) -> dict:  # one example row of a feature file
    """Last row (by `order`) of one artist and year as a dict; empty when the file, the columns or the row are missing."""
    try:  # a missing file or column only removes the example values
        df = pd.read_csv(ROOT / path)  # feature file
        rows = df[(df["artist_id"] == artist_id) & (df[year_col] == year)]  # the artist in the year
        return rows.sort_values(order).iloc[-1].to_dict() if not rows.empty else {}  # last row
    except (OSError, KeyError, ValueError):  # no file, no column, unreadable
        return {}  # cards without example values

def input_value(t: dict, flags: list, name: str, x: float) -> str:  # one example value as text
    """yes / no for the 0 / 1 facts, else a whole number with separators or two decimals."""
    if name in flags:  # a 0 / 1 fact
        return t["t4i_yes"] if x == 1 else t["t4i_no"]  # yes or no
    return f"{int(x):,}" if float(x).is_integer() else f"{x:.2f}"  # whole number or two decimals

def input_rows(cfg: dict, families: list, row: dict) -> list:  # (family, count, values) per family
    """Family name, number of facts and up to three "label: value" examples (genres: the names that are on); `families` is a list of (name, features)."""
    ic, t = cfg["static_tabs"]["inputs"], cfg["static_tabs"]["texts"]  # input settings and texts
    rows = []  # one tuple per family
    for name, feats in families:  # in display order
        shown = [f"{ic['labels'][f]}: {input_value(t, ic['flags'], f, row[f])}" for f in feats if f in row and not pd.isna(row[f])][:3]  # up to three examples
        rows.append((name, str(len(feats)), " \u00b7 ".join(shown)))  # family row
    on = [g for g in GENRES if row.get(f"genre_{g}") == 1]  # genres that are on
    rows.append((ic["genres_name"], str(len(GENRES)), " \u00b7 ".join(on)))  # genres row
    return rows  # all families

def tab4i_html(cfg: dict) -> str:  # "What the model sees" view of tab 4
    """Headline, the example line and two cards (Timing, Style) with their families; the example is the first artist and the year of the tab 6 default."""
    sc = cfg["static_tabs"]  # settings
    t, ic, pal = sc["texts"], sc["inputs"], cfg["design"]["palette"]  # texts, input settings, palette
    aid, year = int(cfg["route"]["artists"][0]), int(cfg["route"]["year"])  # tab 6 default artist and year
    artist = next(a["name"] for a in all_artists(cfg) if int(a["discogs_id"]) == aid)  # its name
    live = cfg["live"]  # stored feature files
    when = example_row(f"{live['folder']}/{live['when_features']}", aid, year, "snapshot_year", "works_so_far_this_year")  # last release of the year
    what = example_row(f"{live['folder']}/{live['what_features']}", aid, year, "year", "year")  # the artist-year
    timing = [(f["name"], f["features"]) for f in ic["timing"]]  # Timing families
    style = [(f["name"], ABLATION_GROUPS[f["group"]]) for f in ic["style"]]  # Style families
    left = inputs_card(pal["timing"], t["t4i_timing_title"], t["t4i_facts"].format(n=len(NUMERIC_FEATURES) + len(GENRES)), t["t4i_timing_question"], input_rows(cfg, timing, when))  # Timing card
    right = inputs_card(pal["style"], t["t4i_style_title"], t["t4i_facts"].format(n=len(WHAT_FEATURES)), t["t4i_style_question"], input_rows(cfg, style, what))  # Style card
    example = f"<div class='ui-caption'>{html.escape(t['t4i_example'].format(artist=artist, year=year))}</div>"  # example line
    return inputs_css(cfg) + "<div class='ui-stack'>" + headline(t["t4i_headline"], html.escape(t["t4i_sub"])) + example + f"<div class='ui-row'>{left}{right}</div><div class='ui-caption'>{html.escape(t['t4i_caption'])}</div></div>"  # page
