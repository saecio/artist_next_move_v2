"""HTML and CSS flow card of tab 2 (inline SVG does not render through st.html). Texts come from the caller, colors from the palette."""

import html  # escape of text

from ui_components import info  # small i with a tooltip


def _box(label: str, count: str, tip: str, color: str) -> str:  # one flow box
    """A box with a label (and its tooltip) and a big count in `color`."""
    return f"<div class='ui-flow-box' style='border-color:{color}'><div class='ui-label'>{html.escape(label)}{info(tip)}</div><div class='ui-flow-count' style='color:{color}'>{html.escape(count)}</div></div>"  # box


def clean_flow(cfg: dict, t: dict, n: int, counts: dict) -> str:  # tab 2 cleaning flow
    """Two input boxes, an arrow to the Works box, an arrow to the Artists box; counts holds master, standalone, works and artists as text."""
    color = cfg["design"]["palette"]["highlight"]  # box color
    arrow = "<div class='ui-flow-arrow'></div>"  # CSS arrow pointing down
    inputs = _box(t["t2_in_master"], counts["master"], t["t2_tip_master"], color) + _box(t["t2_in_standalone"].format(n=n), counts["standalone"], t["t2_tip_standalone"], color)  # the two sources
    return f"<div class='ui-flow'><div class='ui-flow-row'>{inputs}</div>{arrow}{_box(t['t2_out'], counts['works'], t['t2_tip_out'], color)}{arrow}{_box(t['t2_artists'], counts['artists'], t['t2_tip_artists'], color)}</div>"  # whole flow
