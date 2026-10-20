"""HTML and CSS helpers of the demo design system. Colors, sizes and shadows come from the design section of the demo config."""

import html  # escape of tooltip text


def base_css(cfg: dict) -> str:  # one-time style block
    """Style block with the type scale, cards, chips and tooltips; every rule is prefixed ui- so poc- rules are untouched."""
    d = cfg["design"]  # design settings
    c, ty, b, ch = d["palette"], d["type"], d["box"], d["chip"]  # palette, type scale, box and chip sizes
    hd, bt = d["header"], d["button"]  # header sizes and button style
    rules = [  # one rule per element
        f".ui-headline{{font-size:{ty['headline']}px;font-weight:700;color:{c['text']};line-height:1.2;margin:0}}",  # headline
        f".ui-sub{{font-size:{ty['subtitle']}px;color:{c['muted']};margin-top:6px}}",  # subtitle
        f".ui-head{{margin:0 0 {b['gap']}px 0}}",  # headline block
        f".ui-card{{background:{c['card']};border:{b['border']}px solid {c['line']};border-radius:{b['radius']}px;padding:{b['pad']}px;box-shadow:{d['shadow']};color:{c['text']};box-sizing:border-box}}",  # card
        f".ui-card-title{{font-size:{ty['subtitle']}px;font-weight:600;color:{c['text']};margin-bottom:{b['gap']}px}}",  # card title
        f".ui-big{{font-size:{ty['big']}px;font-weight:700;line-height:1.1}}",  # big number
        f".ui-label{{font-size:{ty['label']}px;color:{c['muted']}}}",  # label
        f".ui-caption{{font-size:{ty['caption']}px;color:{c['muted']}}}",  # caption
        f".ui-kpi{{margin-top:{b['gap']}px}}",  # one number block
        f".ui-chip{{display:inline-block;border:{b['border']}px solid;border-radius:999px;padding:{ch['pad_v']}px {ch['pad_h']}px;font-size:{ch['font']}px;color:{c['text']}}}",  # chip, border color inline
        f".ui-chips{{display:flex;flex-wrap:wrap;gap:{b['gap'] // 2}px;margin-top:{b['gap']}px;align-items:center}}",  # row of chips
        f".ui-info{{display:inline-block;position:relative;margin-left:6px;width:16px;height:16px;line-height:16px;text-align:center;border-radius:50%;border:{b['border']}px solid {c['muted']};color:{c['muted']};font-size:{ty['caption']}px;cursor:help;font-style:normal}}",  # small i
        f".ui-info:hover::after{{content:attr(data-tip);position:absolute;left:0;top:22px;z-index:10;width:260px;background:{c['background']};color:{c['text']};border:{b['border']}px solid {c['line']};border-radius:{b['radius'] // 2}px;padding:{b['gap']}px;font-size:{ty['caption']}px;line-height:1.4;text-align:left}}",  # tooltip bubble
        f".ui-stack{{display:flex;flex-direction:column;gap:{b['gap']}px}}",  # blocks one under the other
        f".ui-qtext{{font-size:{ty['label']}px;font-weight:700;color:{c['text']};line-height:1.3;min-height:{round(ty['label'] * 1.3 * 2)}px}}",  # bold question of a Six tests card, two lines
        f".ui-answer{{font-size:{ty['label']}px;color:{c['text']};line-height:1.35}}",  # answer of a Six tests card
        f".ui-split{{display:flex;justify-content:space-between;align-items:center;margin-top:{d['compact']['gap']}px}}",  # label left, value right
        f".ui-meter{{height:{d['compact']['meter_height']}px;border-radius:{d['compact']['meter_height']}px;background:{c['chance']};overflow:hidden;margin-top:4px}}",  # strength bar track
        f".ui-meter-fill{{height:100%}}",  # strength bar fill, width and color inline
        f"[class*='st-key-hyp-card'],[class*='st-key-hyp-card'] [data-testid='stVerticalBlock']{{gap:{d['compact']['gap']}px}}",  # compact Six tests cards: gap between parts
        f"[class*='st-key-hyp-card']{{padding:{d['compact']['pad']}px}}",  # compact Six tests cards: padding
        f"[class*='st-key-hyp-card'] .ui-chips{{margin-top:{d['compact']['gap']}px}}",  # chip row inside a compact card
        f"[class*='st-key-hyp-card'] .ui-chip{{padding:{d['compact']['chip_pad_v']}px {ch['pad_h']}px}}",  # smaller chip inside a compact card
        f"[data-testid='stBaseButton-segmented_controlActive']{{background:{c['timing']};border-color:{c['timing']};color:{c['background']}}}",  # selected view pill
        f"[data-testid='stBaseButton-segmented_controlActive'] p{{color:{c['background']};font-weight:{bt['weight']}}}",  # selected view pill label
        f".ui-row{{display:flex;gap:{b['gap']}px;align-items:stretch}}",  # cards side by side
        f".ui-row>.ui-card{{flex:1}}",  # equal cards in a row
        f".ui-strip{{display:flex;height:{d['strip']['height']}px;border-radius:{d['strip']['height']}px;overflow:hidden;background:{c['chance']}}}",  # sorted artists strip, muted rest
        f".ui-strip-top{{height:100%}}",  # highlighted first segment, width and color inline
        f".ui-strip-labels{{display:flex;justify-content:space-between;font-size:{ty['caption']}px;color:{c['muted']};margin-top:{d['strip']['gap']}px}}",  # end labels
        f".ui-strip-top-label{{font-size:{ty['caption']}px;margin-top:{d['strip']['gap']}px}}",  # label of the highlighted segment, color inline
        f".ui-flow{{display:flex;flex-direction:column;align-items:center}}",  # flow card, boxes and arrows in a column
        f".ui-flow-row{{display:flex;gap:{b['gap']}px;width:100%}}",  # the two inputs side by side
        f".ui-flow-box{{flex:1;box-sizing:border-box;border:{b['border']}px solid;border-radius:{b['radius'] // 2}px;padding:{d['flow']['box_pad']}px;text-align:center;background:{c['background']}}}",  # one box, border color inline
        f".ui-flow>.ui-flow-box{{width:60%;flex:none}}",  # single boxes narrower than the input row
        f".ui-flow-count{{font-size:{d['flow']['count_font']}px;font-weight:700;line-height:1.2}}",  # count in a box
        f".ui-flow-arrow{{position:relative;width:{d['flow']['arrow_width']}px;height:{d['flow']['arrow_len']}px;background:{c['muted']};margin-bottom:{d['flow']['arrow_head']}px}}",  # arrow shaft
        f".ui-flow-arrow::after{{content:'';position:absolute;left:50%;bottom:-{d['flow']['arrow_head']}px;transform:translateX(-50%);border-left:{d['flow']['arrow_head']}px solid transparent;border-right:{d['flow']['arrow_head']}px solid transparent;border-top:{d['flow']['arrow_head']}px solid {c['muted']}}}",  # arrow head made with borders
        f".ui-app-head{{display:flex;align-items:baseline;gap:{hd['gap']}px;white-space:nowrap;overflow:hidden}}",  # header, one line
        f".ui-app-title{{font-size:{hd['title_font']}px;font-weight:700;color:{c['text']}}}",  # project name
        f".ui-app-tag{{font-size:{hd['tagline_font']}px;color:{c['muted']};overflow:hidden;text-overflow:ellipsis}}",  # tagline
        f"[data-testid='stMainBlockContainer']{{padding-top:{hd['top_pad']}px}}",  # less space above the header
        f"[data-testid='stBaseButton-primary'],[data-testid='stBaseButton-secondary']{{font-weight:{bt['weight']}}}",  # same weight on every button
        f"[data-testid='stBaseButton-primary']{{background:{c['timing']};border-color:{c['timing']};color:{c['background']}}}",  # primary: dark label on the bright color
        f"[data-testid='stBaseButton-primary'] p{{color:{c['background']};font-weight:{bt['weight']}}}",  # primary label text
        f"[data-testid='stBaseButton-primary']:hover:not(:disabled){{background:{c['timing']};border-color:{c['timing']};color:{c['background']};filter:brightness({bt['hover_brightness']})}}",  # primary hover
        f"[data-testid='stBaseButton-primary']:hover:not(:disabled) p{{color:{c['background']}}}",  # primary hover label
        f"[data-testid='stBaseButton-secondary'] p{{font-weight:{bt['weight']}}}",  # secondary label weight
    ]
    return "<style>" + "".join(rules) + "</style>"  # the style element

def header_html(cfg: dict) -> str:  # app header above the tabs
    """Project name and one muted tagline on a single line."""
    h = cfg["header"]  # header texts
    return f"<div class='ui-app-head'><span class='ui-app-title'>{html.escape(h['title'])}</span><span class='ui-app-tag'>{html.escape(h['tagline'])}</span></div>"  # one line


def info(text: str) -> str:  # small i with a hover tooltip
    """A small i whose hover bubble shows `text`."""
    return f"<span class='ui-info' data-tip='{html.escape(text, quote=True)}'>i</span>"  # tooltip in a data attribute


def headline(text: str, subtitle: str = "", tip: str = "") -> str:  # page headline
    """Headline, optional subtitle and an optional info tooltip next to the subtitle."""
    sub = f"<div class='ui-sub'>{subtitle}{info(tip) if tip else ''}</div>" if subtitle else ""  # subtitle line
    return f"<div class='ui-head'><h1 class='ui-headline'>{html.escape(text)}</h1>{sub}</div>"  # block


def card_title(title: str, tip: str = "") -> str:  # title line alone
    """A card title with an optional tooltip, for cards that hold a chart."""
    return f"<div class='ui-card-title'>{html.escape(title)}{info(tip) if tip else ''}</div>"  # title line

def card(inner: str, title: str = "", tip: str = "") -> str:  # card around html
    """A card with an optional title and tooltip."""
    head = f"<div class='ui-card-title'>{html.escape(title)}{info(tip) if tip else ''}</div>" if title else ""  # title line
    return f"<div class='ui-card'>{head}{inner}</div>"  # card


def kpi_card(label: str, value: str, sub: str, color: str, info_text: str = "") -> str:  # one number block
    """Label (optional), a big number in `color` and a caption under it."""
    lab = f"<div class='ui-label'>{html.escape(label)}{info(info_text) if info_text else ''}</div>" if label else ""  # label line
    return f"<div class='ui-kpi'>{lab}<div class='ui-big' style='color:{color}'>{html.escape(value)}</div><div class='ui-caption'>{html.escape(sub)}</div></div>"  # block


def chip(text: str, color: str, tip: str = "") -> str:  # small pill
    """A pill with a colored border and an optional tooltip."""
    return f"<span class='ui-chip' style='border-color:{color}'>{html.escape(text)}{info(tip) if tip else ''}</span>"  # chip
