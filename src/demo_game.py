"""Pure functions of the tab 6 game: playable years, verdicts, scoreboard, card status. No Streamlit here."""

import json  # list columns stored as JSON text

import pandas as pd  # missing-value checks


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


def is_counted(row: pd.Series) -> bool:  # whether the year enters the scoreboard
    """True when the stored next_1y label exists and is neither projection nor after the holdout."""
    if pd.isna(row["label_next_1y"]):  # no stored label
        return False  # partial truth
    return not (flag(row["is_projection_next_1y"]) or flag(row["label_after_holdout_next_1y"]))  # flags of next_1y


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
