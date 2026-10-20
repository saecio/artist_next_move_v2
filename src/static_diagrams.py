"""Inline SVG schemas of tabs 1 and 2. Sizes come from design.diagram, colors from the palette, texts from the caller."""

import html  # escape of text and tooltips
import textwrap  # label wrapping

ICONS = {  # 24 x 24 line icons, path data only
    "database": "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",  # cylinder
    "person": "M12 12a4 4 0 100-8 4 4 0 000 8zM4 21c0-4 3.6-7 8-7s8 3 8 7",  # head and shoulders
    "clock": "M12 21a9 9 0 100-18 9 9 0 000 18zM12 7v5l3 2",  # clock face
    "calendar": "M4 6h16v14H4zM4 10h16M8 3v4M16 3v4",  # calendar page
    "spark": "M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2z",  # four point star
    "check": "M5 13l4 4L19 7",  # tick
    "disc": "M12 21a9 9 0 100-18 9 9 0 000 18zM12 14a2 2 0 100-4 2 2 0 000 4z",  # record
    "box": "M4 8l8-4 8 4v8l-8 4-8-4zM4 8l8 4 8-4M12 12v8",  # box
}

def _lines(text: str, width: int) -> list:  # wrapped label lines
    """Label split into lines of at most `width` characters."""
    return textwrap.wrap(text, width) or [text]  # at least one line

def _node(x: float, y: float, w: float, h: float, d: dict, pal: dict, icon: str, head: str, body: str, width: int, tip: str, color: str, count: str = "") -> str:  # one rounded card
    """A card with an icon, a bold head, wrapped body lines, an optional big count and a hover tooltip."""
    s = d["icon"] / 24  # icon scale
    ic = f"<g transform='translate({x + 14:.1f},{y + 12:.1f}) scale({s:.3f})' fill='none' stroke='{color}' stroke-width='{d['stroke'] / s:.2f}' stroke-linecap='round' stroke-linejoin='round'><path d='{ICONS[icon]}'/></g>"  # icon
    hd = f"<text x='{x + 14 + d['icon'] + 8:.1f}' y='{y + 12 + d['icon'] * 0.72:.1f}' fill='{color}' font-size='{d['font']}' font-weight='700'>{html.escape(head)}</text>" if head else ""  # head beside the icon
    top = y + 12 + d["icon"] + 18  # first body line
    body_svg = "".join(f"<text x='{x + 14:.1f}' y='{top + i * (d['font'] + 5):.1f}' fill='{pal['text']}' font-size='{d['font']}'>{html.escape(line)}</text>" for i, line in enumerate(_lines(body, width))) if body else ""  # body lines
    cnt = f"<text x='{x + 14:.1f}' y='{y + h - 12:.1f}' fill='{color}' font-size='{d['font'] + 8}' font-weight='700'>{html.escape(count)}</text>" if count else ""  # big count at the bottom
    box = f"<rect x='{x:.1f}' y='{y:.1f}' width='{w:.1f}' height='{h:.1f}' rx='{d['radius']}' fill='{pal['card']}' stroke='{color}' stroke-width='{d['stroke']}'/>"  # the card
    return f"<g><title>{html.escape(tip)}</title>{box}{ic}{hd}{body_svg}{cnt}</g>"  # node with tooltip

def _arrow(x1: float, y1: float, x2: float, y2: float, d: dict, color: str) -> str:  # elbow arrow with a chevron head
    """Line from (x1,y1) to (x2,y2): straight, or a horizontal-vertical-horizontal elbow, ending in a chevron."""
    mid = (x1 + x2) / 2  # elbow position
    path = f"M{x1:.1f},{y1:.1f}L{x2:.1f},{y2:.1f}" if abs(y1 - y2) < 1 else f"M{x1:.1f},{y1:.1f}L{mid:.1f},{y1:.1f}L{mid:.1f},{y2:.1f}L{x2:.1f},{y2:.1f}"  # shaft
    head = f"M{x2 - 8:.1f},{y2 - 5:.1f}L{x2:.1f},{y2:.1f}L{x2 - 8:.1f},{y2 + 5:.1f}"  # chevron
    return f"<path d='{path}{head}' fill='none' stroke='{color}' stroke-width='{d['stroke']}' stroke-linecap='round' stroke-linejoin='round'/>"  # arrow

def _svg(w: float, h: float, inner: str) -> str:  # scalable wrapper
    """SVG that scales to the width of its container through the view box."""
    return f"<svg viewBox='0 0 {w} {h}' width='100%' xmlns='http://www.w3.org/2000/svg' role='img'>{inner}</svg>"  # no pixel width

def flow_svg(cfg: dict, t: dict, poc: dict) -> str:  # tab 1 schema
    """Source -> clean histories -> history at a past date -> Timing and Style branches -> one check node."""
    d, pal = cfg["design"]["diagram"], cfg["design"]["palette"]  # sizes and colors
    cw, chars = d["col_w"], d["chars"]  # column widths and wrap widths
    gap = (d["width"] - sum(cw)) / (len(cw) - 1)  # space between columns
    xs = [sum(cw[:i]) + gap * i for i in range(len(cw))]  # left edge of each column
    cy, nh, off = d["height"] / 2, d["node_h"], d["node_h"] / 2 + 10  # row center, node height, branch offset
    mono = [  # the straight nodes: column, icon, text, tip
        (0, "database", t["t1_node_source"], t["t1_tip_source"]), (1, "person", t["t1_node_clean"], t["t1_tip_clean"]),  # source and clean
        (2, "clock", t["t1_node_past"], t["t1_tip_past"]), (4, "check", t["t1_node_check"], t["t1_tip_check"]),  # past date and check
    ]
    parts = [_node(xs[c], cy - nh / 2, cw[c], nh, d, pal, ic, "", txt, chars[c], tip, pal["muted"]) for c, ic, txt, tip in mono]  # straight nodes
    branches = [(cy - off, "calendar", poc["label_timing"], poc["q_timing"], t["t1_tip_timing"], pal["timing"]), (cy + off, "spark", poc["label_style"], poc["q_style"], t["t1_tip_style"], pal["style"])]  # top and bottom branch
    for by, ic, head, q, tip, col in branches:  # the two branches
        parts.append(_node(xs[3], by - nh / 2, cw[3], nh, d, pal, ic, head, q, chars[3], tip, col))  # branch node
        parts.append(_arrow(xs[2] + cw[2], cy, xs[3], by, d, col))  # split arrow
        parts.append(_arrow(xs[3] + cw[3], by, xs[4], cy, d, col))  # join arrow
    for a, b in ((0, 1), (1, 2)):  # the two straight arrows
        parts.append(_arrow(xs[a] + cw[a], cy, xs[b], cy, d, pal["muted"]))  # arrow
    return _svg(d["width"], d["height"], "".join(parts))  # drawing

def clean_svg(cfg: dict, t: dict, n: int, counts: dict) -> str:  # tab 2 cleaning schema
    """Two inputs with their counts flowing into one Works box; counts holds master, standalone and works as text."""
    d, pal = cfg["design"]["diagram"], cfg["design"]["palette"]  # sizes and colors
    w, h = d["clean_width"], d["clean_height"]  # view box
    iw, ow, nh = w * 0.6, w * 0.3, h / 2 - 8  # input width, output width, node height
    ys = [0, h - nh]  # top of the two inputs
    ins = [("disc", t["t2_in_master"], t["t2_tip_master"], counts["master"]), ("box", t["t2_in_standalone"].format(n=n), t["t2_tip_standalone"], counts["standalone"])]  # inputs
    parts = [_node(0, y, iw, nh, d, pal, ic, "", txt, d["clean_chars"], tip, pal["highlight"], cnt) for y, (ic, txt, tip, cnt) in zip(ys, ins)]  # input cards
    ox, oy = w - ow, h / 2 - nh / 2  # output position
    parts.append(_node(ox, oy, ow, nh, d, pal, "check", "", t["t2_out"], d["clean_chars"], t["t2_tip_out"], pal["highlight"], counts["works"]))  # output card
    parts += [_arrow(iw, y + nh / 2, ox, h / 2, d, pal["muted"]) for y in ys]  # two arrows into the output
    return _svg(w, h, "".join(parts))  # drawing
