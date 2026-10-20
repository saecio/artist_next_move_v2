"""Pure functions of the tab 6 game: playable years, verdicts, scoreboard, card status. No Streamlit here."""

import json  # list columns stored as JSON text

import pandas as pd  # missing-value checks

from config import CONFIG  # project settings

LAST_COUNTED = CONFIG["years"]["last_complete"]  # last year whose releases are fully catalogued, a reality after it is not counted


def scored_years(timeline: pd.DataFrame, first_year: int) -> list:  # playable years of an artist
    """Years from first_year on whose representative Work has a live next_1y score."""
    ok = timeline[(timeline["year"] >= first_year) & timeline["next_1y"].notna()]  # playable rows
    return [int(y) for y in ok["year"]]  # ascending years


def verdict(score: float, threshold: float) -> str:  # model answer from the score
    """'yes' when the score reaches the threshold, else 'no'."""
    return "yes" if score >= threshold else "no"  # the model rule


def flag(value) -> bool:  # safe boolean of a possibly missing flag
    """True only for a present, true flag (missing values count as false)."""
    return bool(pd.notna(value) and value)  # NaN, None and False give False


def reality_status(label: str | None, reality_year: int | None) -> str:  # counting rule by reality year, the only place it lives
    """'none' without a label or a reality year, 'counted' when the reality year is LAST_COUNTED or earlier, 'open' (asterisk, not counted) for a later year."""
    if label is None or reality_year is None:  # no reality to compare with
        return "none"  # grey, never counted
    return "counted" if reality_year <= LAST_COUNTED else "open"  # complete year counts, the year still running does not


def is_counted(row: pd.Series) -> bool:  # whether the year enters the scoreboard
    """True when the stored next_1y label exists and its reality year (year + 1) is LAST_COUNTED or earlier."""
    label = None if pd.isna(row["label_next_1y"]) else "yes"  # only presence matters for the rule
    return reality_status(label, int(row["year"]) + 1) == "counted"  # reality year of next_1y is year + 1


def play(row: pd.Series, answer: str | None, threshold: float) -> dict:  # record one played year
    """One played year: user answer (None when not played), model verdict, stored reality and whether it counts."""
    label = None if pd.isna(row["label_next_1y"]) else ("yes" if row["label_next_1y"] == 1 else "no")  # stored reality
    return {"year": int(row["year"]), "user": answer, "model": verdict(row["next_1y"], threshold), "label": label, "counted": is_counted(row)}  # history entry


def scoreboard(history: list) -> dict:  # score of a history
    """Counted years only; user fields are None when no counted year had a user answer."""
    rows = [h for h in history if h["counted"]]  # years that enter the score
    users = [h for h in rows if h["user"] is not None]  # years where the user predicted
    out = {"n": len(rows), "model_right": sum(h["model"] == h["label"] for h in rows)}  # model vs reality
    out["user_n"] = len(users) if users else None  # None when the user never played
    out["user_right"] = sum(h["user"] == h["label"] for h in users) if users else None  # user vs reality
    out["agree"] = sum(h["user"] == h["model"] for h in users) if users else None  # user vs model
    return out  # scoreboard values


def session_total(histories: list) -> dict:  # score over all artists
    """Scoreboard over the histories of all artists of the session."""
    return scoreboard([h for hist in histories for h in hist])  # flatten and score


def hidden_counts(works: pd.DataFrame, artist_id: int) -> dict:  # hidden re-recordings per year
    """Number of Works flagged is_rerecording per year for one artist."""
    flagged = works[(works["artist_id"] == artist_id) & works["is_rerecording"]]  # flagged rows
    return {int(y): int(n) for y, n in flagged.groupby("year").size().items()}  # year -> count


def year_cards(artist_works: pd.DataFrame, year: int, hide: bool, cap: int) -> dict:  # Works drawn for one year
    """Works drawn for one year: visible Works up to cap, how many more, and how many are in the file."""
    rows = artist_works[artist_works["year"] == year]  # every Work of the year in the file
    shown = rows[~rows["is_rerecording"]] if hide else rows  # hidden re-recordings excluded when the key is on
    return {"total": len(rows), "cards": shown.head(cap).to_dict("records"), "more": max(len(shown) - cap, 0)}  # counts and capped cards


def card_status(total: int, cards: list, year: int, visible_until: int) -> str:  # kind of card of one year
    """'future', 'gap' (no Work in the file), 'hidden' (only hidden re-recordings) or 'card' for one timeline year."""
    if year > visible_until:  # not revealed yet
        return "future"  # shown as ?
    if total == 0:  # no Work in the data
        return "gap"  # empty year
    return "card" if cards else "hidden"  # drawn Works, or only re-recordings that are hidden


def reliability(top: dict, scale: int) -> tuple:  # x and y of the reliability line
    """Whole numbers out of `scale` for the top group and for any artist, from the holdout precision and positive rate."""
    return round(top["precision_top"] * scale), round(top["holdout_positive_rate"] * scale)  # nearest integers


def outside_rate(scores: pd.DataFrame, cutoff: float, scale: int) -> dict:  # counts and rate below the cutoff
    """Rows below the cutoff, their positives, their positive rate and the rate rounded out of `scale`."""
    below = scores[scores["stage0_score"] < cutoff]  # holdout rows below the cutoff
    rate = float(below["label"].mean())  # share of positives among them
    return {"rows": len(below), "positives": int(below["label"].sum()), "rate": rate, "z": round(rate * scale)}  # counts, rate and z


def outside_text_key(z: int, y: int, similar_gap: int) -> str:  # wording variant of the No-verdict line
    """Config key of the No-verdict reliability line: 'almost the same' only when z and y differ by at most similar_gap (game.similar_gap of the config)."""
    return "what_reliability_no_similar" if abs(z - y) <= similar_gap else "what_reliability_no_other"  # wording variant


def names(text) -> str:  # readable list from JSON text
    """Comma-separated names from a JSON list stored as text; empty when missing."""
    return "" if pd.isna(text) else ", ".join(json.loads(text))  # genres or styles


def next_year(years: list, current: int) -> int | None:  # next playable year
    """Next playable year after current, or None when there is none."""
    later = [y for y in years if y > current]  # years still to play
    return later[0] if later else None  # first one or none


def new_state(years: list, pred: bool = False) -> dict:  # fresh state of one artist
    """Fresh game state of one artist: first playable year, question step, empty history."""
    return {"year": years[0] if years else None, "step": "ask", "answer": None, "history": [], "pred": pred}  # state dict
