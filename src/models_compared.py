"""Models compared: mean development PR-AUC of each model, read from the saved outputs of the model notebooks. Nothing is retrained or typed by hand."""

import json  # notebook files
import re  # table rows of the saved outputs

import pandas as pd  # tables

from config import ROOT  # project root

NUM = r"(\d\.\d+)"  # one decimal number of a printed table
FIRST = re.compile(rf"^(\d)\s+\d{{4}}-\d{{4}}\s+\d+\s+\d+\s+{NUM}\s+{NUM}\s+{NUM}\s*\\?$")  # per-model report, first block: fold, years, train rows, test rows, positive rate, PR-AUC, low
SECOND = re.compile(rf"^(\d)\s+{NUM}\s+{NUM}$")  # per-model report, second block: fold, PR-AUC high, ROC-AUC
WIDE = re.compile(rf"^(\d)\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}$")  # two models side by side: fold, then PR-AUC, low, high, ROC-AUC twice
STYLE = re.compile(rf"^(?:(.+?)\s+)?(\d)\s+\d{{4}}-\d{{4}}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}$")  # Style table: optional model label, fold, years, PR-AUC, low, high, ROC-AUC


def cells_of(path: str) -> list:  # code cells of a notebook
    """(index, source, printed output) of every code cell of a notebook given by a path relative to the project root."""
    with open(ROOT / path, encoding="utf-8") as f:  # saved notebook
        nb = json.load(f)  # notebook json
    out = []  # one tuple per code cell
    for i, c in enumerate(nb["cells"]):  # in notebook order
        if c["cell_type"] == "code":  # only code cells
            text = "".join("".join(o.get("text", [])) + "".join(o.get("data", {}).get("text/plain", [])) for o in c.get("outputs", []))  # printed output
            out.append((i, "".join(c["source"]), text))  # index, source, output
    return out  # all code cells


def summarize(scores: list, lows: list, highs: list) -> tuple:  # mean over the folds
    """Mean score and mean 95% limits over the five folds."""
    return sum(scores) / len(scores), sum(lows) / len(lows), sum(highs) / len(highs)  # means


def timing_rows(cells: list, keys: dict, rule_label: str) -> list:  # Timing rows from the practice-run reports
    """One (model, mean, low, high, cell) per model of `keys` (name -> key of the notebook reports dict), plus the release-counts row from the first side-by-side table."""
    rows = []  # result rows
    for name, key in keys.items():  # in the configured order
        hit = [(i, o) for i, s, o in cells if f"reports_1y['{key}'] = evaluate_folds" in s]  # cell that fits this model
        if not hit:  # not in the notebook
            continue  # left out
        i, text = hit[0]  # its cell
        first = {m.group(1): m.groups()[1:] for m in map(FIRST.match, map(str.strip, text.splitlines())) if m}  # fold -> (rate, PR-AUC, low)
        second = {m.group(1): m.group(2) for m in map(SECOND.match, map(str.strip, text.splitlines())) if m}  # fold -> high
        if len(first) != 5 or len(second) != 5:  # a block is missing or cut
            continue  # left out
        folds = sorted(first)  # fold numbers
        rows.append((name, *summarize([float(first[f][1]) for f in folds], [float(first[f][2]) for f in folds], [float(second[f]) for f in folds]), i))  # model row
    side = [(i, o) for i, s, o in cells if "reports_1y" in s and "report[metrics]" in s]  # side-by-side tables
    if side:  # the first one starts with the release counts
        i, text = side[0]  # its cell
        wide = [WIDE.match(x.strip()) for x in text.splitlines()]  # data rows
        wide = [m for m in wide if m][:5]  # five folds
        if len(wide) == 5:  # complete
            rows.append((rule_label, *summarize([float(m.group(2)) for m in wide], [float(m.group(3)) for m in wide], [float(m.group(4)) for m in wide]), i))  # release counts row
    return rows  # Timing rows


def style_rows(cells: list, names: list, rule_prefix: str, rule_label: str) -> list:  # Style rows from the comparison table
    """One (model, mean, low, high, cell) per name found in the comparison table, plus the release-counts row; a name matches the table label that starts with it."""
    flat = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())  # name without spaces or case
    hit = [(i, o) for i, s, o in cells if "comparison = pd.concat" in s]  # comparison cell
    if not hit:  # not in the notebook
        return []  # nothing to read
    i, text = hit[0]  # its cell
    table, label = {}, None  # label -> rows
    for line in text.splitlines():  # printed table
        m = STYLE.match(line.strip())  # one row
        if m:  # a data row
            label = m.group(1) or label  # the label is printed on the first row of a block only
            table.setdefault(label, []).append([float(v) for v in m.groups()[2:5]])  # PR-AUC, low, high
    rows = []  # result rows
    for name in names + [rule_prefix]:  # models, then release counts
        found = [k for k in table if flat(k).startswith(flat(name))]  # matching table label
        if found and len(table[found[0]]) == 5:  # five folds read
            cols = list(zip(*table[found[0]]))  # PR-AUC, low, high lists
            rows.append((rule_label if name == rule_prefix else name, *summarize(*map(list, cols)), i))  # model or release counts row
    return rows  # Style rows


def build_models_compared(cfg: dict) -> tuple:  # table of the Models compared view
    """(table, missing): question, model, mean_score, ci_low, ci_high, source; missing lists the models whose scores were not found."""
    sc = cfg["static_tabs"]  # settings
    pa, t, p = sc["paths"], sc["texts"], cfg["poc"]["texts"]  # paths, texts and question labels
    names = sc["models_compared"]  # models of the comparison
    parts = [  # (question, notebook path, rows)
        (p["label_timing"], pa["notebook_timing"], timing_rows(cells_of(pa["notebook_timing"]), sc["timing_report_keys"], t["t5_col_rule"])),  # Timing
        (p["label_style"], pa["notebook_style"], style_rows(cells_of(pa["notebook_style"]), names, sc["style_rule_label"], t["t5_col_rule"])),  # Style
    ]
    rows, missing = [], []  # table rows and models left out
    for question, nb, found in parts:  # per question
        got = {r[0] for r in found}  # models read
        missing += [f"{question}: {n}" for n in names if n not in got]  # left out
        rows += [{"question": question, "model": m, "mean_score": s, "ci_low": lo, "ci_high": hi, "source": f"{nb} cell {i}"} for m, s, lo, hi, i in found]  # one row each
    return pd.DataFrame(rows), missing  # table and the list of omissions
