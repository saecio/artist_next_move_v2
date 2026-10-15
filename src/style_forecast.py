"""Style growth forecast: panel, baselines, models, walk-forward evaluation on development years.

Design fixed in docs/decisions.md (entries "Style trend forecast"). Only development
target years are scored here. Nothing is saved to disk by this module.

Usage:
    from style_forecast import build_panel, walk_forward, hits_table
    panel = build_panel(kept, last_year=2017)
    results = walk_forward(panel)
"""

import numpy as np  # arrays and the seeded bootstrap
import pandas as pd  # tables
from scipy.stats import rankdata, spearmanr  # ranks and rank correlation
from sklearn.ensemble import RandomForestRegressor  # random forest model
from sklearn.linear_model import BayesianRidge  # bayesian ridge model
from xgboost import XGBRegressor  # gradient boosting model

from style_series import style_year_counts, style_year_table  # series builders from the first phase

FIRST_YEAR = 1890  # first calendar column of the style by year table
DEV_FIRST_TARGET = 1993  # first development target year
DEV_LAST_TARGET = 2017  # last development target year, nothing later is ever read
MIN_BASE = 100  # a style is eligible when the mean of the three latest years is at least this
TOP_K = 10  # size of the ranking that is scored
SEED = 1  # one seed for every model and for the bootstrap
N_BOOT = 2000  # bootstrap draws over years
MODELS = ["bayes_ridge", "xgboost", "random_forest"]  # model names, in report order
# Locked Random Forest settings, fixed in the decisions entry "Style trend forecast, adoption and pass rule results, final test, 2024 projection and output procedure" (trial m_trees600 of results/trials/rf_style_forecast.jsonl).
LOCKED_RF_PARAMS = {"n_estimators": 600, "max_depth": 13, "min_samples_leaf": 7, "max_features": 0.5, "bootstrap": True, "max_samples": 0.3253691715004556, "random_state": 1}  # same keys as the tuning trials
BASELINES = ["repeat_last", "mean_growth"]  # ranking baselines, in report order
TRANCHES = [(1993, 1997), (1998, 2002), (2003, 2007), (2008, 2012), (2013, 2017)]  # project tranches of target years


def style_table(kept: pd.DataFrame, last_year: int) -> pd.DataFrame:
    """Style by year counts with a fixed column range FIRST_YEAR to last_year."""
    table = style_year_table(style_year_counts(kept[kept["year"] <= last_year]))  # only Works dated up to last_year
    return table.reindex(columns=range(FIRST_YEAR, last_year + 1), fill_value=0)  # same columns whatever the data holds


def primary_genre_codes(kept: pd.DataFrame, table: pd.DataFrame, last_year: int, genres: list = None) -> tuple:
    """Index of the most frequent genre of each style up to each year, and the genre names.

    Counts use only Works dated up to the column year, so a code never sees later Works.
    The literal tag "1" is ignored (8 Works), which leaves the 11 real genres.
    `genres` fixes the genre list (a truncated rebuild may not hold every genre yet).
    """
    pairs = kept.loc[kept["year"] <= last_year, ["year", "styles", "genres"]].explode("styles").explode("genres").dropna()  # one row per Work, style and genre
    pairs = pairs[pairs["genres"] != "1"]  # drop the literal tag "1"
    names = genres if genres is not None else sorted(pairs["genres"].unique())  # the real genres in a fixed order
    counts = pairs.groupby(["genres", "styles", "year"]).size()  # Works per genre, style and year
    cube = np.stack([counts[g].unstack("year").reindex(index=table.index, columns=table.columns).fillna(0).to_numpy() if g in counts.index.get_level_values(0) else np.zeros(table.shape) for g in names])  # genre by style by year, zeros for a genre not seen yet
    return np.cumsum(cube, axis=2).argmax(axis=0), names  # cumulative counts, then the top genre per style and year


def build_panel(kept: pd.DataFrame, last_year: int, genres: list = None) -> pd.DataFrame:
    """One row per (style, origin year) with base at least MIN_BASE.

    Features use only years up to the origin. The label (log growth of the next
    year over the base) is missing when the next year is after `last_year`.
    """
    table = style_table(kept, last_year)  # style by year counts
    counts = table.to_numpy().astype(float)  # matrix styles by years
    years = np.array(table.columns)  # calendar years of the columns
    codes, genres = primary_genre_codes(kept, table, last_year, genres)  # primary genre per style and year
    first10 = np.where((counts >= 10).any(axis=1), years[(counts >= 10).argmax(axis=1)], 10**6)  # first year with at least 10 Works
    totals = counts.sum(axis=0)  # all style-year cells of each year
    frames = []  # one frame per origin year
    for j in range(5, len(years)):  # origins that have five earlier columns
        c = [counts[:, j - i] for i in range(6)]  # c[0] is the origin year, c[5] five years before
        base = (c[0] + c[1] + c[2]) / 3  # mean of the three latest years
        keep = base >= MIN_BASE  # eligible styles at this origin
        if not keep.any():  # nothing eligible, skip the year
            continue  # next origin
        gr = [c[i] / np.where((c[i + 1] + c[i + 2] + c[i + 3]) > 0, (c[i + 1] + c[i + 2] + c[i + 3]) / 3, np.nan) - 1 for i in range(3)]  # smoothed growth at t, t-1, t-2
        frame = pd.DataFrame({
            "style": table.index[keep],  # style name
            "origin": years[j],  # origin year t
            "c_t": c[0][keep],  # count in the origin year
            "base": base[keep],  # mean of the three latest years
            "lc0": np.log1p(c[0][keep]), "lc1": np.log1p(c[1][keep]), "lc2": np.log1p(c[2][keep]),  # log1p of the three latest counts
            "lg0": np.log((c[0][keep] + 1) / (c[1][keep] + 1)), "lg1": np.log((c[1][keep] + 1) / (c[2][keep] + 1)), "lg2": np.log((c[2][keep] + 1) / (c[3][keep] + 1)),  # one-year log growth at t, t-1, t-2
            "age": years[j] - first10[keep],  # years since the style first had 10 Works
            "share": c[0][keep] / totals[j],  # share of the year's total
            "gr0": gr[0][keep], "gr1": gr[1][keep], "gr2": gr[2][keep],  # smoothed growth at t, t-1, t-2 for the baselines
        })
        onehot = np.eye(len(genres))[codes[keep, j]]  # primary genre one-hot
        for k, name in enumerate(genres):  # one column per genre
            frame["genre_" + name] = onehot[:, k]  # 1 when this is the primary genre
        if j + 1 < len(years):  # the next year exists in the table
            frame["c_next"] = counts[keep, j + 1]  # count in the target year
            frame["label"] = np.log(np.maximum(frame["c_next"], 1) / frame["base"])  # log growth of the target year over the base
        else:  # last origin, no label
            frame["c_next"] = np.nan  # unknown
            frame["label"] = np.nan  # unknown
        frames.append(frame)  # keep this origin
    return pd.concat(frames, ignore_index=True)  # all origins stacked


def feature_columns(panel: pd.DataFrame) -> list:
    """Model input columns: log counts, lagged growth, age, share and genre dummies."""
    return ["lc0", "lc1", "lc2", "lg0", "lg1", "lg2", "age", "share"] + [c for c in panel.columns if c.startswith("genre_")]  # fixed order


def make_model(name: str, rf_params: dict = None):
    """A fresh model with the settings fixed in the design entry. `rf_params` replaces the forest settings of a tuning trial."""
    if name == "bayes_ridge":  # linear model
        return BayesianRidge()  # default settings
    if name == "xgboost":  # boosted trees
        return XGBRegressor(max_depth=3, n_estimators=200, random_state=SEED, n_jobs=1)  # other settings default
    if rf_params is not None:  # a tuning trial brings its own forest settings
        return RandomForestRegressor(**rf_params, n_jobs=1)  # forest with the trial settings
    return RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=10, max_features="sqrt", random_state=SEED, n_jobs=1)  # forest


def fit_model(name: str, train: pd.DataFrame, cols: list, rf_params: dict = None) -> dict:
    """Fit one model on `train` and return it with the columns and the scaling statistics it needs."""
    x_train = train[cols]  # inputs
    mean, std = None, None  # tree models take unscaled inputs
    if name == "bayes_ridge":  # only this model gets scaled inputs
        mean, std = x_train.mean(), x_train.std().replace(0, 1)  # train-only statistics, constant columns left unscaled
        x_train = (x_train - mean) / std  # scaled training inputs
    model = make_model(name, rf_params)  # new model, default settings unless a trial passes its own
    model.fit(x_train, train["label"])  # fit on the training rows
    return {"model": model, "columns": cols, "mean": mean, "std": std}  # everything needed to predict later


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predicted log growth of `frame` from a bundle made by `fit_model`."""
    x = frame[bundle["columns"]]  # inputs in the training order
    if bundle["mean"] is not None:  # scaled model
        x = (x - bundle["mean"]) / bundle["std"]  # same transform as in training
    return bundle["model"].predict(x)  # predicted log growth


def fit_predict(name: str, train: pd.DataFrame, test: pd.DataFrame, cols: list, rf_params: dict = None) -> np.ndarray:
    """Fit one model on `train` and predict the label of `test`. `rf_params` replaces the forest settings only."""
    return predict_bundle(fit_model(name, train, cols, rf_params), test)  # fit, then predict


def walk_forward(panel: pd.DataFrame, first: int = DEV_FIRST_TARGET, last: int = DEV_LAST_TARGET, allow_test: bool = False, rf_params: dict = None) -> pd.DataFrame:
    """Score every method for each development target year, one fit per model and year.

    For target year T the origin is T - 1 and training rows are origins up to T - 2.
    """
    if last > DEV_LAST_TARGET and not allow_test:  # the test years are only scored when asked explicitly
        raise ValueError("only development target years are allowed")  # hard stop unless allow_test is set
    cols = feature_columns(panel)  # inputs
    out = []  # one frame per target year
    for target in range(first, last + 1):  # one target year at a time
        train = panel[(panel["origin"] <= target - 2) & panel["label"].notna()]  # labels known at the origin
        test = panel[panel["origin"] == target - 1].copy()  # styles eligible at the origin
        test["target"] = target  # target year
        test["repeat_last"] = test["gr0"]  # growth of year t repeated
        test["mean_growth"] = test[["gr0", "gr1", "gr2"]].mean(axis=1)  # mean growth of t, t-1, t-2
        for name in MODELS:  # the three models
            test[name] = fit_predict(name, train, test, cols, rf_params)  # predicted log growth, forest settings replaced when given
        out.append(test)  # keep
    return pd.concat(out, ignore_index=True)  # all development years


def top_hits(score: np.ndarray, label: np.ndarray, k: int) -> int:
    """Overlap between the k best by score and the k best by actual label. Missing scores rank last."""
    s = np.where(np.isnan(score), -np.inf, score)  # missing scores go to the bottom
    predicted = set(np.argsort(-s, kind="stable")[:k])  # predicted top k
    actual = set(np.argsort(-label, kind="stable")[:k])  # actual top k
    return len(predicted & actual)  # number of hits


def hits_table(results: pd.DataFrame, k: int) -> pd.DataFrame:
    """Hits@k per target year for random (analytic), the baselines and the models."""
    rows = {}  # year -> dict of method -> hits
    for year, g in results.groupby("target"):  # one year at a time
        row = {"eligible": len(g), "random": k * k / len(g)}  # analytic expectation for a random ranking
        for m in BASELINES + MODELS:  # every scored method
            row[m] = top_hits(g[m].to_numpy(), g["label"].to_numpy(), k)  # hits of this method
        rows[year] = row  # keep
    return pd.DataFrame(rows).T  # years as rows


def spearman_table(results: pd.DataFrame) -> pd.DataFrame:
    """Spearman between score and actual label per year, on the styles scored by every method."""
    methods = BASELINES + MODELS  # all scored methods
    rows = {}  # year -> dict
    for year, g in results.groupby("target"):  # one year at a time
        common = g.dropna(subset=methods)  # styles with a score from every method
        rows[year] = {m: spearmanr(common[m], common["label"])[0] for m in methods}  # rank correlation per method
        rows[year]["common"] = len(common)  # number of styles used
    return pd.DataFrame(rows).T  # years as rows


def boot_ci(values: np.ndarray, level: float = 0.95) -> tuple:
    """Mean and percentile interval of the mean over years, bootstrap over years."""
    rng = np.random.default_rng(SEED)  # fixed seed
    draws = rng.integers(0, len(values), size=(N_BOOT, len(values)))  # resample years with replacement
    means = values[draws].mean(axis=1)  # mean of each draw
    low, high = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])  # interval bounds
    return values.mean(), low, high  # point estimate and bounds


def pass_rule(hits10: pd.DataFrame, spearman: pd.DataFrame) -> dict:
    """The five pass-rule conditions for the model with the highest mean hits@10."""
    candidate = hits10[MODELS].mean().idxmax()  # model with the highest mean hits@10
    baseline = "repeat_last" if hits10["repeat_last"].mean() >= hits10["mean_growth"].mean() else "mean_growth"  # best baseline, tie to repeat last
    cand, base, rand = (hits10[c].to_numpy(dtype=float) for c in (candidate, baseline, "random"))  # per-year hits
    diff_base = boot_ci(cand - base)  # paired difference against the best baseline
    diff_rand = boot_ci(cand - rand)  # paired difference against random
    conditions = {
        "1 mean hits@10 at least 0.5 above best baseline": (cand.mean() - base.mean() >= 0.5, cand.mean() - base.mean()),  # gap in mean hits
        "2 at least as many hits as best baseline in 15+ years": ((cand >= base).sum() >= 15, int((cand >= base).sum())),  # years won or tied
        "3 lower bound of paired difference above 0": (diff_base[1] > 0, diff_base[1]),  # bootstrap lower bound
        "4 at least 1.0 above random and lower bound above 0": (cand.mean() - rand.mean() >= 1.0 and diff_rand[1] > 0, (cand.mean() - rand.mean(), diff_rand[1])),  # gap and lower bound
        "5 mean Spearman not below best baseline": (spearman[candidate].mean() >= spearman[baseline].mean(), (spearman[candidate].mean(), spearman[baseline].mean())),  # mean correlations
    }
    recommended = candidate if all(ok for ok, _ in conditions.values()) else baseline  # candidate only when all five hold
    return {"candidate": candidate, "baseline": baseline, "conditions": conditions, "recommended": recommended, "diff_base": diff_base, "diff_rand": diff_rand}  # everything to print


def count_table(results: pd.DataFrame) -> pd.DataFrame:
    """Expected Works, error and range per row for the three models and the two count baselines.

    Range = expected x exp(10th and 90th percentile of log residuals from earlier target years, pooled).
    The first target year has no earlier residuals and is left out.
    """
    out = results[["target", "style", "c_t", "base", "c_next"]].copy()  # identifiers and actual counts
    expected = {m: out["base"] * np.exp(results[m]) for m in MODELS}  # model counts, base x exp(prediction)
    expected["repeat_ct"] = out["c_t"].clip(lower=1)  # count baseline: last year's count
    expected["mean3"] = out["base"]  # count baseline: mean of the three latest years
    actual = out["c_next"].clip(lower=1)  # actual count, at least 1 as in the label
    for m, e in expected.items():  # every count method
        out[m + "_exp"] = e  # expected Works
        out[m + "_eps"] = np.log(actual / e)  # log residual
    for m in expected:  # ranges need earlier residuals only
        lo, hi = np.full(len(out), np.nan), np.full(len(out), np.nan)  # empty bounds
        for year in out["target"].unique()[1:]:  # skip the first year
            eps = out.loc[out["target"] < year, m + "_eps"]  # residuals of earlier folds
            q10, q90 = eps.quantile(0.1), eps.quantile(0.9)  # their 10th and 90th percentiles
            rows = (out["target"] == year).to_numpy()  # rows of this year
            lo[rows], hi[rows] = out.loc[rows, m + "_exp"] * np.exp(q10), out.loc[rows, m + "_exp"] * np.exp(q90)  # range
        out[m + "_lo"], out[m + "_hi"] = lo, hi  # store the range
        out[m + "_ape"] = (out[m + "_exp"] - actual).abs() / actual  # absolute percentage error
        out[m + "_in"] = (actual >= lo) & (actual <= hi)  # inside the range
    return out[out["target"] > out["target"].min()]  # first year left out


def count_metrics(counts: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Median absolute percentage error and range coverage, all eligible styles and each model's top 10."""
    methods = MODELS + ["repeat_ct", "mean3"]  # model counts and the two baselines
    rows = []  # one row per set and count method
    sets = {"all eligible": np.ones(len(counts), dtype=bool)}  # every eligible style
    for m in MODELS:  # predicted top 10 of each model, per year
        picked = np.zeros(len(results), dtype=bool)  # mask over results rows
        for _, idx in results.groupby("target").indices.items():  # rows of one year
            picked[idx[np.argsort(-results[m].to_numpy()[idx], kind="stable")[:TOP_K]]] = True  # best TOP_K by this model
        sets["top 10 of " + m] = picked[results["target"].isin(counts["target"]).to_numpy()]  # same rows as counts
    for set_name, mask in sets.items():  # each style set
        for m in methods:  # each count method
            sub = counts[mask]  # rows of the set
            rows.append({"set": set_name, "method": m, "n": len(sub), "median_ape": sub[m + "_ape"].median(), "coverage": sub[m + "_in"].mean()})  # metrics
    return pd.DataFrame(rows)


def truncation_test(kept: pd.DataFrame, full_panel: pd.DataFrame, years: list) -> pd.DataFrame:
    """Rebuild the features of origin t after removing every Work dated after t and compare with the full build."""
    cols = ["c_t", "base", "lc0", "lc1", "lc2", "lg0", "lg1", "lg2", "age", "share", "gr0", "gr1", "gr2"] + [c for c in full_panel.columns if c.startswith("genre_")]  # all feature columns
    rows = []  # one row per tested year
    for t in years:  # origins to test
        cut = build_panel(kept[kept["year"] <= t], last_year=t, genres=[c[6:] for c in full_panel.columns if c.startswith("genre_")])  # data after t removed, same genre columns
        cut = cut[cut["origin"] == t].set_index("style")  # features of origin t
        full = full_panel[full_panel["origin"] == t].set_index("style")  # same origin from the full build
        same_styles = set(cut.index) == set(full.index)  # same eligible styles
        diff = (cut[cols] - full.loc[cut.index, cols]).abs().max().max() if same_styles else np.nan  # largest absolute difference
        rows.append({"origin": t, "rows": len(cut), "same_styles": same_styles, "max_abs_diff": diff})  # result
    return pd.DataFrame(rows)


def spearman_boot(results: pd.DataFrame) -> pd.DataFrame:
    """Mean Spearman over years and its 95% interval, bootstrap over styles within each year.

    Each draw resamples the styles of a year with replacement (same draw for every method),
    recomputes the rank correlation per year, and averages over years.
    """
    methods = BASELINES + MODELS  # scored methods
    rng = np.random.default_rng(SEED)  # fixed seed
    draws = {m: [] for m in methods}  # per method: one array of N_BOOT correlations per year
    point = {m: [] for m in methods}  # per method: observed correlation per year
    for _, g in results.groupby("target"):  # one year at a time
        common = g.dropna(subset=methods)  # styles with a score from every method
        idx = rng.integers(0, len(common), size=(N_BOOT, len(common)))  # resampled style positions
        y = rankdata(common["label"].to_numpy()[idx], axis=1)  # ranks of the resampled actual growth
        for m in methods:  # each method
            x = rankdata(common[m].to_numpy()[idx], axis=1)  # ranks of the resampled scores
            draws[m].append(np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(x, y)]))  # Spearman of each draw
            point[m].append(spearmanr(common[m], common["label"])[0])  # observed Spearman
    rows = {}  # method -> summary
    for m in methods:  # each method
        means = np.mean(draws[m], axis=0)  # mean over years for each draw
        rows[m] = {"mean": np.mean(point[m]), "low": np.percentile(means, 2.5), "high": np.percentile(means, 97.5)}  # point estimate and interval
    return pd.DataFrame(rows).T  # one row per method


def expected_counts(frame: pd.DataFrame) -> dict:
    """Expected Works per count method: the three models and the two count baselines."""
    expected = {m: frame["base"] * np.exp(frame[m]) for m in MODELS}  # model counts, base x exp(prediction)
    expected["repeat_ct"] = frame["c_t"].clip(lower=1)  # count baseline: last year's count
    expected["mean3"] = frame["base"]  # count baseline: mean of the three latest years
    return expected  # method -> Series


def count_bounds(results: pd.DataFrame) -> dict:
    """10th and 90th percentile of the log residuals of every count method, pooled over all given folds."""
    actual = results["c_next"].clip(lower=1)  # actual count, at least 1 as in the label
    return {m: tuple(np.log(actual / e).quantile([0.1, 0.9])) for m, e in expected_counts(results).items()}  # (q10, q90) per method


def projection(panel: pd.DataFrame, results: pd.DataFrame, base_year: int, rf_params: dict = None) -> tuple:
    """Fit the three models on origins up to base_year - 1 and predict origin base_year.

    The labels of origin base_year - 1 (target year base_year) are in the training rows.
    Ranges use the residual percentiles of every fold in `results`.
    Returns the predicted frame (target year base_year + 1) and the fitted bundles.
    """
    cols = feature_columns(panel)  # inputs
    train = panel[(panel["origin"] <= base_year - 1) & panel["label"].notna()]  # rows with a known label
    frame = panel[panel["origin"] == base_year].copy()  # styles eligible at the base year
    frame["target"] = base_year + 1  # projected year
    frame["repeat_last"] = frame["gr0"]  # recommended ranking score
    bundles = {m: fit_model(m, train, cols, rf_params) for m in MODELS}  # one fit per model, forest settings replaced when given
    for m in MODELS:  # predicted log growth of each model
        frame[m] = predict_bundle(bundles[m], frame)  # prediction
    bounds = count_bounds(results)  # residual percentiles from the scored folds
    for m, e in expected_counts(frame).items():  # expected Works and range per count method
        frame[m + "_exp"], frame[m + "_lo"], frame[m + "_hi"] = e, e * np.exp(bounds[m][0]), e * np.exp(bounds[m][1])  # expected, low, high
    return frame, bundles  # table input and fitted models


def long_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per style, target year and method, with growth, expected Works, range and rank (1 = best score)."""
    parts = []  # one block per method
    for m in MODELS + ["repeat_last"]:  # the three models and the recommended baseline
        score = frame[m]  # ranking score
        count = "repeat_ct" if m == "repeat_last" else m  # count columns that go with the method
        part = pd.DataFrame({
            "style": frame["style"], "target_year": frame["target"], "method": m,  # identifiers
            "expected_growth": frame["gr0"] if m == "repeat_last" else np.exp(score) - 1,  # growth over the base (the score itself for repeat last)
            "expected_works": frame[count + "_exp"], "range_low": frame[count + "_lo"], "range_high": frame[count + "_hi"],  # count and range
        })
        part["rank"] = score.groupby(frame["target"]).rank(ascending=False, method="first", na_option="bottom").astype(int)  # rank within the target year
        parts.append(part)  # keep
    return pd.concat(parts, ignore_index=True)  # all methods stacked


def walk_forward_tuned(panel: pd.DataFrame, rf_params: dict) -> pd.DataFrame:
    """Same walk-forward as `walk_forward`, with the Random Forest column refitted using `rf_params`."""
    results = walk_forward(panel)  # every method with the design settings, development years only
    cols = feature_columns(panel)  # same inputs as the other models
    for target in sorted(results["target"].unique()):  # one target year at a time
        train = panel[(panel["origin"] <= target - 2) & panel["label"].notna()]  # same training rows as the walk-forward
        rows = results["target"] == target  # rows of this target year
        results.loc[rows, "random_forest"] = fit_predict_params(train, results[rows], cols, rf_params)  # forest with the locked settings
    return results  # all methods, forest tuned

def fit_predict_params(train: pd.DataFrame, test: pd.DataFrame, cols: list, rf_params: dict) -> np.ndarray:
    """Fit the Random Forest with `rf_params` on `train` and predict `test`."""
    return predict_bundle(fit_model("random_forest", train, cols, rf_params), test)  # fit, then predict


def paired_differences(results: pd.DataFrame) -> pd.DataFrame:
    """Each model minus repeat last year in hits@10 and in Spearman: mean difference and year-bootstrap 95% interval."""
    hits = hits_table(results, 10)  # hits@10 per year
    spearman = spearman_table(results)  # Spearman per year on the common styles
    rows = {}  # row label -> mean, low, high
    for m in MODELS:  # each model against the recommended baseline
        rows[f"{m} hits@10"] = boot_ci((hits[m] - hits["repeat_last"]).to_numpy(dtype=float))  # paired by year
        rows[f"{m} Spearman"] = boot_ci((spearman[m] - spearman["repeat_last"]).to_numpy(dtype=float))  # paired by year
    return pd.DataFrame(rows, index=["mean", "low", "high"]).T  # one row per model and metric
