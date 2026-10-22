"""HTML and CSS helpers of the demo design system. Colors, sizes and shadows come from the design section of the demo config."""

import html  # escape of tooltip text


def base_css(cfg: dict) -> str:  # one-time style block
    """Style block with the type scale, cards, chips, tooltips, widgets and the two breakpoints; every rule is prefixed ui- so poc- rules are untouched."""
    d = cfg["design"]  # design settings
    c, ty, b, ch = d["palette"], d["type"], d["box"], d["chip"]  # palette, type scale, box and chip sizes
    hd, bt = d["header"], d["button"]  # header sizes and button style
    fo = d["font"]  # typefaces
    head_font = f"'{fo['heading']}',Georgia,serif"  # serif heading stack
    body_font = f"'{fo['body']}',{fo['fallback']}"  # body stack with system fallbacks
    fluid = lambda px, low, vw: f"clamp({round(px * low)}px,{vw}vw + 8px,{px}px)"  # size that shrinks on phones, full size from desktop width
    box = f"border:{b['border']}px solid {c['line']}"  # thin border shared by cards, tooltips and idle widgets
    pill = "[data-testid='stBaseButton-segmented_control'],[data-testid='stBaseButton-pills']"  # idle view pills
    pill_on = "[data-testid='stBaseButton-segmented_controlActive'],[data-testid='stBaseButton-pillsActive']"  # selected view pills
    rules = [  # one rule per element
        f".stApp{{font-family:{body_font};line-height:{ty['line']};overflow-x:clip}}",  # body font and line height everywhere, no sideways page scroll
        f"[data-testid='stTab'],[data-testid='stTab'] p{{font-family:{head_font};font-weight:{fo['tab_weight']}}}",  # tab labels in the serif
        f".ui-big,.ui-flow-count,table,td,th{{font-variant-numeric:tabular-nums}}",  # figures of equal width
        f".ui-eyebrow,.ui-label{{font-size:{ty['eyebrow']}px;text-transform:uppercase;letter-spacing:{ty['eyebrow_tracking']}}}",  # small uppercase labels
        f".ui-eyebrow .ui-info,.ui-chip .ui-info{{text-transform:none;letter-spacing:normal}}",  # the tooltip icon and bubble keep normal case
        f"[class*='st-key-ui-accent-timing']{{border-top:3px solid {c['timing']} !important}}",  # card of the Timing question only
        f"[class*='st-key-ui-accent-style']{{border-top:3px solid {c['style']} !important}}",  # card of the Style question only
        f".ui-headline{{font-family:{head_font};letter-spacing:{ty['head_tracking']};font-size:{fluid(ty['headline'], 0.75, 3.2)};font-weight:600;color:{c['text']};line-height:1.2;margin:0;text-align:left;overflow-wrap:anywhere}}",  # headline
        f".ui-sub{{font-size:{ty['subtitle']}px;color:{c['muted']};margin-top:6px;text-align:left}}",  # subtitle
        f".ui-head{{margin:0 0 {b['gap']}px 0}}",  # headline block
        f".ui-card{{background:{c['card']};{box};border-radius:{b['radius']}px;padding:{b['pad']}px;box-shadow:none;color:{c['text']};box-sizing:border-box;text-align:left;overflow-wrap:anywhere}}",  # flat card
        f".ui-card-title{{font-size:{ty['subtitle']}px;font-weight:600;color:{c['text']};margin-bottom:{b['gap']}px;text-align:left}}",  # card title
        f".ui-big{{font-size:{fluid(ty['big'], 0.64, 5)};font-family:{head_font};letter-spacing:{ty['head_tracking']};font-weight:600;line-height:1.1}}",  # big number
        f".ui-label{{color:{c['muted']}}}",  # label color
        f".ui-caption{{font-size:{ty['caption']}px;color:{c['muted']}}}",  # caption
        f".ui-kpi{{margin-top:{b['gap']}px}}",  # one number block
        f".ui-halves{{display:flex;gap:{cfg['static_tabs']['story']['pair_gap']}px}}",  # the two halves of the third story card
        f".ui-chip{{display:inline-block;max-width:100%;border:{b['border']}px solid;border-radius:{ch['radius']}px;padding:{ch['pad_v']}px {ch['pad_h']}px;font-size:{ch['font']}px;text-transform:uppercase;letter-spacing:{ty['eyebrow_tracking']};color:{c['text']};background:transparent}}",  # rectangular chip, border color inline
        f".ui-chips{{display:flex;flex-wrap:wrap;gap:{b['gap'] // 2}px;margin-top:{b['gap']}px;align-items:center}}",  # row of chips
        f".ui-info{{display:inline-block;position:relative;margin-left:6px;width:16px;height:16px;line-height:16px;text-align:center;border-radius:50%;border:{b['border']}px solid {c['muted']};color:{c['muted']};font-size:{ty['caption']}px;cursor:help;font-style:normal;font-weight:400}}",  # small i
        f".ui-info:hover::after{{content:attr(data-tip);position:absolute;left:0;top:22px;z-index:10;width:min(260px,calc(100vw - 4rem));background:{c['card']};color:{c['text']};{box};border-radius:{b['radius']}px;padding:{b['gap']}px;font-size:{ty['caption']}px;line-height:1.4;text-align:left;text-transform:none;letter-spacing:normal;font-weight:400;box-shadow:none}}",  # tooltip bubble, a white card with a thin border
        f".ui-stack{{display:flex;flex-direction:column;gap:{b['gap']}px}}",  # blocks one under the other
        f"{pill}{{background:transparent;{box};border-radius:{bt['radius']}px;color:{c['text']};box-shadow:none}}",  # idle view pill, flat
        f"{pill_on}{{background:{c['text']};border-color:{c['text']};border-radius:{bt['radius']}px;color:{c['background']};box-shadow:none}}",  # selected view pill, ink with paper text
        f"{pill_on.replace(',', ' p,')} p{{color:{c['background']};font-weight:{bt['weight']}}}",  # selected view pill label
        f".ui-row{{display:flex;gap:{b['gap']}px;align-items:stretch}}",  # cards side by side
        f".ui-row>.ui-card{{flex:1;min-width:0}}",  # equal cards in a row that never push past the page edge
        f"[data-testid='stColumn']{{min-width:0}}",  # columns (artist cards) shrink instead of overflowing
        f"[data-testid='stVerticalBlockBorderWrapper']{{box-shadow:none}}",  # native bordered containers are flat too
        f".ui-strip,.ui-strip-labels{{max-width:100%}}",  # the strip stays inside its card
        f".ui-strip{{display:flex;height:{d['strip']['height']}px;border-radius:{ch['radius']}px;overflow:hidden;background:{c['chance']}}}",  # sorted artists strip, muted rest
        f".ui-strip-top{{height:100%}}",  # highlighted first segment, width and color inline
        f".ui-strip-labels{{display:flex;justify-content:space-between;font-size:{ty['caption']}px;color:{c['muted']};margin-top:{d['strip']['gap']}px}}",  # end labels
        f".ui-strip-top-label{{font-size:{ty['caption']}px;margin-top:{d['strip']['gap']}px}}",  # label of the highlighted segment, color inline
        f".ui-flow{{display:flex;flex-direction:column;align-items:center}}",  # flow card, boxes and arrows in a column
        f".ui-flow-row{{display:flex;gap:{b['gap']}px;width:100%}}",  # the two inputs side by side
        f".ui-flow-box{{flex:1;box-sizing:border-box;border:{b['border']}px solid;border-radius:{b['radius']}px;padding:{d['flow']['box_pad']}px;text-align:center;background:{c['background']}}}",  # one box, border color inline, centered by design (a diagram)
        f".ui-flow>.ui-flow-box{{width:60%;flex:none}}",  # single boxes narrower than the input row
        f".ui-flow-count{{font-size:{fluid(d['flow']['count_font'], 0.7, 4)};font-family:{head_font};letter-spacing:{ty['head_tracking']};font-weight:600;line-height:1.2}}",  # count in a box
        f".ui-flow-arrow{{position:relative;width:{d['flow']['arrow_width']}px;height:{d['flow']['arrow_len']}px;background:{c['muted']};margin-bottom:{d['flow']['arrow_head']}px}}",  # arrow shaft
        f".ui-flow-arrow::after{{content:'';position:absolute;left:50%;bottom:-{d['flow']['arrow_head']}px;transform:translateX(-50%);border-left:{d['flow']['arrow_head']}px solid transparent;border-right:{d['flow']['arrow_head']}px solid transparent;border-top:{d['flow']['arrow_head']}px solid {c['muted']}}}",  # arrow head made with borders
        f".ui-flow-by{{font-size:{ty['caption']}px;color:{c['muted']};margin:{d['flow']['by_gap']}px 0}}",  # small muted connector word between two flow boxes
        f".ui-app-head{{display:flex;align-items:baseline;flex-wrap:wrap;gap:0 {hd['gap']}px;overflow:hidden}}",  # header, one line on desktop and wrapping on phones
        f".ui-app-title{{font-size:{hd['title_font']}px;font-family:{head_font};letter-spacing:{ty['head_tracking']};font-weight:600;color:{c['text']};white-space:nowrap}}",  # project name
        f".ui-app-tag{{font-size:{hd['tagline_font']}px;color:{c['muted']};min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}",  # tagline
        f"[data-testid='stMainBlockContainer']{{padding-top:{hd['top_pad']}px;max-width:100%}}",  # less space above the header
        f"[data-testid='stBaseButton-primary'],[data-testid='stBaseButton-secondary']{{font-weight:{bt['weight']};border-radius:{bt['radius']}px;box-shadow:none}}",  # same weight and flat corners on every button
        f"[data-testid='stBaseButton-primary']{{background:{c['text']};border-color:{c['text']};color:{c['background']}}}",  # primary: ink fill with paper text
        f"[data-testid='stBaseButton-primary'] p{{color:{c['background']};font-weight:{bt['weight']}}}",  # primary label text
        f"[data-testid='stBaseButton-primary']:hover:not(:disabled){{background:{c['timing']};border-color:{c['timing']};color:{c['background']}}}",  # primary hover
        f"[data-testid='stBaseButton-primary']:hover:not(:disabled) p{{color:{c['background']}}}",  # primary hover label
        f"[data-testid='stBaseButton-secondary']{{background:transparent;{box};color:{c['text']}}}",  # secondary: thin border, no fill
        f"[data-testid='stBaseButton-secondary'] p{{font-weight:{bt['weight']}}}",  # secondary label weight
        f".ui-scroll{{max-width:100%;overflow-x:auto}}",  # tables and schemas scroll inside their own box
        f"[data-testid='stVegaLiteChart'],[data-testid='stArrowVegaLiteChart']{{max-width:100%;overflow-x:auto}}",  # fixed-size charts scroll inside their own box
        f"@media (max-width:1023px){{[data-testid='stMainBlockContainer']{{padding-left:1.5rem !important;padding-right:1.5rem !important}}"  # tablet: less side padding
        f"[data-testid='stHorizontalBlock']{{flex-wrap:wrap}}[data-testid='stColumn']{{flex:1 1 calc(50% - 1rem) !important;min-width:calc(50% - 1rem) !important}}"  # tablet: at most two columns
        f".ui-row{{flex-wrap:wrap}}.ui-row>.ui-card{{flex:1 1 calc(50% - {b['gap']}px)}}"  # tablet: at most two cards per row
        f".ui-halves{{flex-direction:column}}.ui-m-row{{grid-template-columns:{d['method']['label_w'] * 2 // 3}px 1fr}}}}",  # tablet: story halves stack, narrower run labels
        f"@media (max-width:639px){{[data-testid='stMainBlockContainer']{{padding-left:1rem !important;padding-right:1rem !important}}"  # phone: smallest side padding
        f"[data-testid='stColumn']{{flex:1 1 100% !important;min-width:100% !important}}"  # phone: one column
        f".ui-row>.ui-card{{flex:1 1 100%}}.ui-flow>.ui-flow-box{{width:100%}}.ui-flow-row{{flex-wrap:wrap}}.ui-flow-row>.ui-flow-box{{flex:1 1 100%}}"  # phone: one card per row, wide flow boxes
        f"[class*='st-key-ui-accent'] [data-testid='stVerticalBlock']{{height:auto !important;max-height:none !important}}"  # phone: the artist cards grow with their stacked content
        f".ui-big{{white-space:normal !important}}.ui-m-row{{grid-template-columns:{d['method']['label_w'] // 2}px 1fr}}}}",  # phone: big numbers may wrap, narrowest run labels
    ]
    return "<style>" + "".join(rules) + "</style>"  # the style element

def header_html(cfg: dict) -> str:  # app header above the tabs
    """Project name and one muted tagline on a single line."""
    h = cfg["header"]  # header texts
    return f"<div class='ui-app-head'><span class='ui-app-title'>{html.escape(h['title'])}</span><span class='ui-app-tag'>{html.escape(h['tagline'])}</span></div>"  # one line


def accent_key(question: str, tab: str) -> str:  # container key that draws the top accent of one question
    """Key of a card that belongs to one question; the css matches it by the question name."""
    return f"ui-accent-{question}-{tab}"  # unique per tab, so two tabs never share a key


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
    lab = f"<div class='ui-label ui-eyebrow'>{html.escape(label)}{info(info_text) if info_text else ''}</div>" if label else ""  # label line
    return f"<div class='ui-kpi'>{lab}<div class='ui-big' style='color:{color}'>{html.escape(value)}</div><div class='ui-caption'>{html.escape(sub)}</div></div>"  # block


def chip(text: str, color: str, tip: str = "") -> str:  # small pill
    """A pill with a colored border and an optional tooltip."""
    return f"<span class='ui-chip' style='border-color:{color}'>{html.escape(text)}{info(tip) if tip else ''}</span>"  # chip
