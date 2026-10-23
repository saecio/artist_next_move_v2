"""
Feature allowlist and fit/score pairs for the WHEN models.

Every pair has the shape `evaluate_folds` (src/backtesting.py) expects:
`fit(train)` learns only from the train rows it is given and returns the
fitted object, `score(test, fitted)` returns one score per test row (higher
= more likely the target is 1). The target is not passed in, the pairs are
built with `make_<model>(target)`.

Usage:
    from models import make_logistic_regression
    fit_fn, score_fn = make_logistic_regression("next_1y")
    report = evaluate_folds(folds, "next_1y", fit_fn, score_fn)
"""

from __future__ import annotations  # allow type hints to be read lazily

import pandas as pd  # dataframe library used throughout
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # boosted trees (native NaN) and bagged trees
from sklearn.impute import SimpleImputer  # fills the NaN columns with a train statistic
from sklearn.linear_model import LogisticRegression  # linear model on the log-odds
from sklearn.naive_bayes import GaussianNB  # probabilistic model with independent Gaussian features
from sklearn.pipeline import Pipeline  # chains imputer, scaler and model so all are fitted on train only
from sklearn.preprocessing import StandardScaler  # puts features on a common scale for the linear model
from xgboost import XGBClassifier  # gradient boosted trees with native NaN handling

from config import CONFIG  # project settings read from config.yaml

# Seed for every model with randomness, the project seed of config.yaml.
SEED = CONFIG["project"]["SEED"]  # project.SEED

# The only columns allowed as model inputs for WHEN (docs/decisions.md,
# "Feature allowlist for WHEN modeling"). genres is expanded into one 0/1
# column per genre by build_features.
NUMERIC_FEATURES = [
    "works_so_far_this_year", "albums_so_far_this_year", "singles_so_far_this_year",
    "is_album_like", "is_single_like",
    "history_count", "years_since_last", "years_active_so_far", "historical_rate",
]

# Fixed genre vocabulary, so every fold gets the same columns. The stray "1"
# tag (4 Works, each of which also carries a real genre) is not encoded.
GENRES = [
    "Blues", "Electronic", "Folk, World, & Country", "Funk / Soul", "Hip Hop", "Jazz",
    "Latin", "Pop", "Reggae", "Rock", "Stage & Screen",
]


def build_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Model input matrix: the numeric allowlist plus one 0/1 column per genre."""
    numeric = snapshots[NUMERIC_FEATURES].astype(float)  # booleans become 0.0 / 1.0, NaN stays NaN
    dummies = pd.get_dummies(snapshots["genres"].explode()).groupby(level=0).max()  # one row per snapshot, True if the Work carries the genre
    genres = dummies.reindex(index=snapshots.index, columns=GENRES, fill_value=False).astype(float)  # fixed columns, 0.0 for absent genres
    genres.columns = [f"genre_{g}" for g in GENRES]  # prefix to keep names apart from the numeric features
    return pd.concat([numeric, genres], axis=1)  # numeric allowlist first, then genres


def _make_pair(target: str, make_model):
    """Fit/score pair around any model builder: fit on the train rows, score the test rows.

    `make_model()` returns a fresh, unfitted sklearn estimator or Pipeline.
    The imputation (when the model has one) lives inside the Pipeline, so
    its statistics are learned on the fold's train rows only.
    """
    def fit(train: pd.DataFrame):
        """Fit a fresh model on the train rows of a fold. Input: the train dataframe. Output: the fitted estimator."""
        return make_model().fit(build_features(train), train[target])  # fitted on the fold's train rows only

    def score(test: pd.DataFrame, fitted):
        """Score test rows with a fitted model. Inputs: the test dataframe and the fitted estimator. Output: the probability of target == 1 for each row."""
        return fitted.predict_proba(build_features(test))[:, 1]  # probability of target == 1

    return fit, score  # return both functions for the folds


def make_logistic_regression(target: str):
    """Untuned Logistic Regression pair.

    NaN in years_since_last and historical_rate (an artist's first Work, or
    no years elapsed yet) is filled with the median of the fold's train rows,
    the same rule fit_when_baseline uses for its fallback. No missing-value
    indicator columns: history_count and years_active_so_far already tell
    the model which rows were imputed. Features are only standardized, no
    log transform of the heavy-tailed counts, a known limit of this untuned
    linear model.
    """
    return _make_pair(target, lambda: Pipeline([  # model: median fill, scaling, logistic regression
        ("impute", SimpleImputer(strategy="median")),  # median, not mean: the counts are heavy-tailed
                                                       # and the baseline uses the same fallback rule
        ("scale", StandardScaler()),  # zero mean and unit variance from train. A linear model with L2
                                      # needs this so no feature is penalized just for its units
        ("model", LogisticRegression(
            C=1.0,            # Inverse regularization strength. Smaller = stronger shrinkage of the
                              # coefficients. 1.0 is the sklearn default, kept because this step is
                              # deliberately untuned.
            solver="lbfgs",   # Default solver, handles millions of rows and L2 well.
            max_iter=1000,    # Iteration cap for the solver. The 100 default can stop before
                              # converging on 1M+ rows, 1000 leaves room to converge.
        )),
    ]))


def make_naive_bayes(target: str):
    """Untuned Gaussian Naive Bayes pair, a cheap probabilistic reference point.

    Known mismatch: GaussianNB models every feature as an independent
    Gaussian, but 13 of the 20 columns after encoding are 0/1 (the two
    format flags and the 11 genre columns), and the counts are skewed and
    correlated. It is kept as a reference, not as a competitive candidate,
    so the mismatch is documented in docs/decisions.md and not fixed with a
    mixed Gaussian/Bernoulli model. Same median imputation as the others.
    """
    return _make_pair(target, lambda: Pipeline([  # model: median fill, Gaussian naive Bayes
        ("impute", SimpleImputer(strategy="median")),  # same imputation as the other non-native-NaN models
        ("model", GaussianNB(
            var_smoothing=1e-9,  # Fraction of the largest feature variance added to every variance
                                 # for numerical stability, so a near-constant column cannot give a
                                 # zero variance. 1e-9 is the sklearn default, kept (no tuning).
                                 # No scaler: NB fits each feature's mean and variance on its own,
                                 # so feature units do not matter.
        )),
    ]))


def make_random_forest(target: str):
    """Untuned Random Forest pair, the standard tree-ensemble reference.

    Same median imputation as the other non-native-NaN models. Trees do not
    need scaling. The values below are common starting points for tabular
    data with a few dozen features, not the result of any search.
    """
    return _make_pair(target, lambda: Pipeline([  # model: median fill, random forest
        ("impute", SimpleImputer(strategy="median")),  # same imputation as the other non-native-NaN models
        ("model", RandomForestClassifier(
            n_estimators=200,      # Number of trees. More trees = more stable average prediction and
                                   # lower variance, at higher compute cost. 200-500 is a common
                                   # range, gains beyond that are marginal.
            max_depth=12,          # Main lever against overfitting. Unlimited depth lets trees
                                   # memorize the train data. 8-15 is typical for tabular data with
                                   # a few dozen features.
            min_samples_leaf=50,   # Minimum rows in a leaf. With 1M+ train rows a leaf of a few rows
                                   # is noise, 50 keeps each leaf's probability estimate usable.
            max_features="sqrt",   # Features tried per split (about 4 of 20). The decorrelation that
                                   # makes averaging many trees work. "sqrt" is the standard default.
            max_samples=0.25,      # Each tree trains on a random 25% of the rows. With millions of
                                   # rows this barely changes the fit and cuts training time a lot.
            n_jobs=-1,             # Use every CPU core, trees are independent.
            random_state=SEED,     # Reproducible runs.
        )),
    ]))


def make_hist_gradient_boosting(target: str):
    """Untuned HistGradientBoostingClassifier pair, boosting before XGBoost.

    No imputer and no scaler: the model sends NaN (years_since_last and
    historical_rate for an artist's first Work) down a learned branch at
    every split, and trees do not need scaling. The values below are common
    starting points, not the result of any search.
    """
    return _make_pair(target, lambda: HistGradientBoostingClassifier(  # model: gradient boosting that handles NaN itself
        learning_rate=0.1,      # Step size of each new tree. Smaller = slower, more careful
                                # learning that needs more trees. 0.1 is the usual starting point.
        max_iter=200,           # Number of boosting rounds (trees). With learning_rate 0.1, 200
                                # rounds is enough for the loss to flatten on this data size
                                # without a long fit.
        max_leaf_nodes=31,      # Leaves per tree, the main control of how complex each tree is.
                                # 31 is the sklearn default, a moderate size for tabular data.
        min_samples_leaf=50,    # Minimum rows in a leaf. With 1M+ train rows a leaf of a few
                                # rows is noise, 50 keeps each leaf estimate usable (same value
                                # as the Random Forest).
        early_stopping=False,   # Off, so the model does not carve a validation split out of the
                                # fold's train rows: all of them are used and the number of
                                # rounds is fixed by max_iter.
        random_state=SEED,      # Reproducible runs.
    ))


def make_xgboost(target: str, **overrides):
    """Untuned XGBoost pair, the starting point of the staged tuning.

    No imputer and no scaler: XGBoost learns a default direction for NaN at
    every split (years_since_last and historical_rate for an artist's first
    Work), and trees do not need scaling. The values below are the stage 0
    starting point, not the result of any search. Tuning happens in three
    phases (random search, focused grid, manual refinement, see
    src/tuning.py and docs/decisions.md). `overrides` replaces any stage 0
    value, which is how the tuned configuration is evaluated.
    """
    return _make_pair(target, lambda: XGBClassifier(  # model: XGBoost, tuned values replace the defaults
        n_estimators=300,        # Number of boosting rounds (trees). With learning_rate 0.1,
                                 # 300 rounds is enough for the loss to flatten on this data size.
        learning_rate=0.1,       # Step size of each new tree. Smaller = slower, more careful
                                 # learning that needs more trees. 0.1 is the usual starting point.
        max_depth=6,             # Depth of each tree, the main control of how complex it is.
                                 # 6 is the XGBoost default, a moderate depth for tabular data.
        min_child_weight=50,     # Minimum sum of hessian weight in a leaf. For the logistic loss a
                                 # row weighs at most 0.25 (p * (1 - p)), so 50 means at least 200
                                 # rows at p = 0.5 and more where predictions are confident. It is
                                 # the XGBoost counterpart of the 50-row leaf floor of the other
                                 # tree models, not an exact match.
        subsample=0.8,           # Fraction of rows each tree sees. Below 1.0 adds randomness that
                                 # reduces overfitting and speeds up the fit.
        colsample_bytree=0.8,    # Fraction of features each tree sees. Same idea as subsample,
                                 # for columns.
        reg_lambda=1.0,          # L2 penalty on the leaf values. Larger = smoother, more cautious
                                 # trees. 1.0 is the XGBoost default.
        tree_method="hist",      # Histogram-based split search, the fast method for millions of rows.
        eval_metric="aucpr",     # Metric XGBoost would report on an evaluation set. No eval set is
                                 # passed here, so it does not change the fit; it matches the
                                 # project's main metric for when one is used.
        n_jobs=-1,               # Use every CPU core.
        random_state=SEED,       # Reproducible runs.
    ).set_params(**overrides))  # tuned values replace the stage 0 ones


# Locked XGBoost configurations (docs/decisions.md, "XGBoost tuning results"):
# the best grid combination per target, trials g8 and g10. They replace the
# stage 0 values of make_xgboost; everything not listed keeps the stage 0 value.
FINAL_XGB_PARAMS = {
    "next_1y": {"max_depth": 7, "reg_lambda": 15, "learning_rate": 0.05, "n_estimators": 500, "subsample": 0.8, "colsample_bytree": 0.55, "min_child_weight": 50, "reg_alpha": 0.01},  # tuned values for the 1-year target
    "next_2y": {"max_depth": 9, "reg_lambda": 2, "learning_rate": 0.03, "n_estimators": 500, "subsample": 0.8, "colsample_bytree": 0.55, "min_child_weight": 50, "reg_alpha": 0.01},  # tuned values for the 2-year target
}


# WHAT inputs that come from build_what_features (src/what_target.py): the numeric
# and share columns, then one True / False column per genre of the WHEN vocabulary.
WHAT_FEATURE_COLUMNS = [
    "n_styles_seen", "n_styles_year", "n_new_styles_year", "styles_per_work", "new_style_rate_so_far",
    "years_since_new_style", "years_since_prev_active_year", "years_active_so_far", "top_style_share",
    "n_genres_seen", "albums_share_year", "singles_share_year",
] + [f"genre_{g}" for g in GENRES]

# WHAT inputs that already sit in the rows of build_what_table, so they are not repeated in the features table.
WHAT_TABLE_INPUTS = ["history_count", "n_works_year"]

# Full list of WHAT model inputs, without duplicates (checked by make_what_xgboost).
WHAT_FEATURES = WHAT_TABLE_INPUTS + WHAT_FEATURE_COLUMNS

# Columns that must never be a WHAT model input: keys, and everything that describes the next
# active year or the label itself (docs/decisions.md, "WHAT feature set and point-in-time audit").
WHAT_FORBIDDEN_INPUTS = ["artist_id", "year", "next_active_year", "new_style_year", "n_works_next_year"]


def _what_matrix(rows: pd.DataFrame, features: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """Model input matrix of WHAT rows: the requested columns, as floats, in the order of `feature_columns`.

    `rows` is a train or test table from build_what_table (with `artist_id` and `year`),
    `features` the output of build_what_features. A requested column is taken from `features`
    when it is there, joined on (artist_id, year), and from `rows` otherwise. The join is a
    reindex on the key, so it can neither add nor reorder rows, and it is asserted.
    """
    keys = ["artist_id", "year"]  # row identity shared by both tables
    lookup = features.set_index(keys)  # one row per artist-year
    assert lookup.index.is_unique, "the features table has repeated (artist_id, year) keys"
    wanted = pd.MultiIndex.from_frame(rows[keys])  # the keys of the rows, in their order
    joined = lookup.reindex(wanted)  # features of each row, NaN row if the key is missing
    assert len(joined) == len(rows), "the join changed the row count"
    assert wanted.isin(lookup.index).all(), "some rows have no features"  # a missing key, not a feature that is NaN
    joined.index = rows.index  # back to the index of the rows
    assert joined.index.equals(rows.index), "the join changed the index"
    from_features = [c for c in feature_columns if c in lookup.columns]  # columns that come from the features table
    from_rows = [c for c in feature_columns if c not in lookup.columns]  # columns that come from the rows themselves
    matrix = pd.concat([joined[from_features], rows[from_rows]], axis=1)[feature_columns]  # requested order
    return matrix.astype(float)  # booleans become 0.0 / 1.0, NaN stays NaN


def make_what_xgboost(features: pd.DataFrame, feature_columns: list = WHAT_FEATURES, target: str = "new_style_year", **overrides):
    """WHAT sibling of the WHEN XGBoost stage 0: untuned reference model, one configuration.

    Same hyperparameters as `make_xgboost` (stage 0), copied here because that function
    builds its input matrix from the WHEN snapshot columns. No imputation (NaN is handled
    natively), no class weights, binary objective on `target` (the XGBClassifier default, as in
    WHEN, so there is no intended difference in the settings). `features` is the output of
    `build_what_features`; `feature_columns` must not contain any of `WHAT_FORBIDDEN_INPUTS`
    or a repeated name. `overrides` replaces any stage 0 value through `set_params`, which is how
    a tuned configuration is evaluated; with no overrides the model is the stage 0 one. Returns
    `(fit, score)` with the signatures `evaluate_folds` expects: `fit(train)` returns the fitted
    model, `score(test, fitted)` the probability of `target` == 1.
    """
    assert not set(feature_columns) & set(WHAT_FORBIDDEN_INPUTS), "a forbidden column is in the feature list"  # stop if a forbidden column is requested
    assert len(set(feature_columns)) == len(feature_columns), "the feature list has repeated names"  # stop if a column name is repeated

    def fit(train: pd.DataFrame):
        """Fit a fresh XGBoost with the stage 0 settings plus `overrides` on the WHAT matrix of the train rows. Input: the train dataframe. Output: the fitted model."""
        model = XGBClassifier(  # fresh XGBoost with the stage 0 settings
            n_estimators=300,        # Same as the WHEN stage 0: boosting rounds.
            learning_rate=0.1,       # Same: step size of each new tree.
            max_depth=6,             # Same: depth of each tree.
            min_child_weight=50,     # Same: minimum hessian weight in a leaf (at least 200 rows at p = 0.5).
            subsample=0.8,           # Same: fraction of rows each tree sees.
            colsample_bytree=0.8,    # Same: fraction of features each tree sees.
            reg_lambda=1.0,          # Same: L2 penalty on the leaf values.
            tree_method="hist",      # Same: histogram split search.
            eval_metric="aucpr",     # Same: matches the project's main metric, no eval set is passed.
            n_jobs=-1,               # Same: every CPU core.
            random_state=SEED,       # Same: reproducible runs.
        ).set_params(**overrides)  # tuned values replace the stage 0 ones, nothing changes when empty
        return model.fit(_what_matrix(train, features, feature_columns), train[target])  # fitted on the fold's train rows only

    def score(test: pd.DataFrame, fitted):
        """Score test rows on the WHAT matrix with a fitted model. Inputs: the test dataframe and the fitted model. Output: the probability of target == 1 for each row."""
        return fitted.predict_proba(_what_matrix(test, features, feature_columns))[:, 1]  # probability target == 1

    return fit, score  # return both functions for the folds


def make_what_logistic_regression(features: pd.DataFrame, feature_columns: list = WHAT_FEATURES, target: str = "new_style_year"):
    """WHAT sibling of the WHEN Logistic Regression: untuned linear reference, one configuration.

    Same Pipeline and settings as `make_logistic_regression`, copied here because that function
    builds its input matrix from the WHEN snapshot columns. NaN is filled with the median of the
    fold's train rows (same rule as WHEN, no missing-value indicator columns). Limit: years_since_new_style
    is 17.07% NaN (fold 5 train) and was not checked as an exact identifier of the imputed rows, unlike
    history_count and years_active_so_far in WHEN. Features are only standardized, no log transform of the
    heavy-tailed counts, a known limit of this untuned linear model. `features` is the output of
    `build_what_features`; `feature_columns` must not contain any of `WHAT_FORBIDDEN_INPUTS`
    or a repeated name. Returns `(fit, score)` with the signatures `evaluate_folds` expects:
    `fit(train)` returns the fitted model, `score(test, fitted)` the probability of `target` == 1.
    """
    assert not set(feature_columns) & set(WHAT_FORBIDDEN_INPUTS), "a forbidden column is in the feature list"  # stop if a forbidden column is requested
    assert len(set(feature_columns)) == len(feature_columns), "the feature list has repeated names"  # stop if a column name is repeated

    def fit(train: pd.DataFrame):
        """Fit a fresh Logistic Regression pipeline (median imputation, scaling) on the WHAT matrix of the train rows. Input: the train dataframe. Output: the fitted pipeline."""
        model = Pipeline([  # pipeline: median fill, scaling, logistic regression
            ("impute", SimpleImputer(strategy="median")),  # Same: median, not mean, the counts are heavy-tailed.
            ("scale", StandardScaler()),  # Same: zero mean and unit variance from train, so L2 does not penalize units.
            ("model", LogisticRegression(
                C=1.0,            # Same: inverse regularization strength, sklearn default, untuned.
                solver="lbfgs",   # Same: default solver.
                max_iter=1000,    # Same: iteration cap, leaves room to converge.
            )),
        ])
        return model.fit(_what_matrix(train, features, feature_columns), train[target])  # fitted on the fold's train rows only

    def score(test: pd.DataFrame, fitted):
        """Score test rows on the WHAT matrix with a fitted model. Inputs: the test dataframe and the fitted model. Output: the probability of target == 1 for each row."""
        return fitted.predict_proba(_what_matrix(test, features, feature_columns))[:, 1]  # probability target == 1

    return fit, score  # return both functions for the folds


def make_what_naive_bayes(features: pd.DataFrame, feature_columns: list = WHAT_FEATURES, target: str = "new_style_year"):
    """WHAT sibling of the WHEN Gaussian Naive Bayes: untuned probabilistic reference point, one configuration.

    Same Pipeline and settings as `make_naive_bayes`, copied here because that function builds
    its input matrix from the WHEN snapshot columns. Known mismatch: GaussianNB models every
    feature as an independent Gaussian, but 11 of the 25 columns are 0/1 (the genre columns)
    and the counts and shares are skewed, bounded and correlated. It is kept as a reference, not
    as a competitive candidate, so the mismatch is documented and not fixed with a mixed
    Gaussian/Bernoulli model. Its scores are overconfident (calibration), which does not affect
    the ranking metrics. NaN is filled with the median of the fold's train rows (same rule as
    WHEN, no missing-value indicator columns). Limit: years_since_new_style is 17.07% NaN (fold 5
    train) and was not checked as an exact identifier of the imputed rows. `features` is the output
    of `build_what_features`; `feature_columns` must not contain any of `WHAT_FORBIDDEN_INPUTS`
    or a repeated name. Returns `(fit, score)` with the signatures `evaluate_folds` expects:
    `fit(train)` returns the fitted model, `score(test, fitted)` the probability of `target` == 1.
    """
    assert not set(feature_columns) & set(WHAT_FORBIDDEN_INPUTS), "a forbidden column is in the feature list"  # stop if a forbidden column is requested
    assert len(set(feature_columns)) == len(feature_columns), "the feature list has repeated names"  # stop if a column name is repeated

    def fit(train: pd.DataFrame):
        """Fit a fresh Gaussian Naive Bayes pipeline (median imputation) on the WHAT matrix of the train rows. Input: the train dataframe. Output: the fitted pipeline."""
        model = Pipeline([  # pipeline: median fill, Gaussian naive Bayes
            ("impute", SimpleImputer(strategy="median")),  # Same: median imputation as the other non-native-NaN models.
            ("model", GaussianNB(
                var_smoothing=1e-9,  # Same: sklearn default, fraction of the largest variance added to every variance.
            )),
        ])
        return model.fit(_what_matrix(train, features, feature_columns), train[target])  # fitted on the fold's train rows only

    def score(test: pd.DataFrame, fitted):
        """Score test rows on the WHAT matrix with a fitted model. Inputs: the test dataframe and the fitted model. Output: the probability of target == 1 for each row."""
        return fitted.predict_proba(_what_matrix(test, features, feature_columns))[:, 1]  # probability target == 1

    return fit, score  # return both functions for the folds


def make_what_random_forest(features: pd.DataFrame, feature_columns: list = WHAT_FEATURES, target: str = "new_style_year"):
    """WHAT sibling of the WHEN Random Forest: untuned tree-ensemble reference, one configuration.

    Same Pipeline and settings as `make_random_forest`, copied here because that function
    builds its input matrix from the WHEN snapshot columns. NaN is filled with the median of the
    fold's train rows (same rule as WHEN, no missing-value indicator columns). Trees do not need
    scaling. Limit: years_since_new_style is 17.07% NaN (fold 5 train) and was not checked as an
    exact identifier of the imputed rows. The values are the WHEN common starting points, not the
    result of any search. `features` is the output of `build_what_features`; `feature_columns`
    must not contain any of `WHAT_FORBIDDEN_INPUTS` or a repeated name. Returns `(fit, score)` with
    the signatures `evaluate_folds` expects: `fit(train)` returns the fitted model,
    `score(test, fitted)` the probability of `target` == 1.
    """
    assert not set(feature_columns) & set(WHAT_FORBIDDEN_INPUTS), "a forbidden column is in the feature list"  # stop if a forbidden column is requested
    assert len(set(feature_columns)) == len(feature_columns), "the feature list has repeated names"  # stop if a column name is repeated

    def fit(train: pd.DataFrame):
        """Fit a fresh Random Forest pipeline (median imputation) on the WHAT matrix of the train rows. Input: the train dataframe. Output: the fitted pipeline."""
        model = Pipeline([  # pipeline: median fill, random forest
            ("impute", SimpleImputer(strategy="median")),  # Same: median imputation as the other non-native-NaN models.
            ("model", RandomForestClassifier(
                n_estimators=200,      # Same: number of trees.
                max_depth=12,          # Same: main lever against overfitting.
                min_samples_leaf=50,   # Same: minimum rows in a leaf.
                max_features="sqrt",   # Same: features tried per split (5 of 25 here, about 4 of 20 in WHEN).
                max_samples=0.25,      # Same: each tree trains on a random 25% of the rows.
                n_jobs=-1,             # Same: every CPU core.
                random_state=SEED,     # Same: reproducible runs.
            )),
        ])
        return model.fit(_what_matrix(train, features, feature_columns), train[target])  # fitted on the fold's train rows only

    def score(test: pd.DataFrame, fitted):
        """Score test rows on the WHAT matrix with a fitted model. Inputs: the test dataframe and the fitted model. Output: the probability of target == 1 for each row."""
        return fitted.predict_proba(_what_matrix(test, features, feature_columns))[:, 1]  # probability target == 1

    return fit, score  # return both functions for the folds


def make_what_hist_gradient_boosting(features: pd.DataFrame, feature_columns: list = WHAT_FEATURES, target: str = "new_style_year"):
    """WHAT sibling of the WHEN HistGradientBoostingClassifier: untuned boosting reference, one configuration.

    Same settings as `make_hist_gradient_boosting`, copied here because that function builds its
    input matrix from the WHEN snapshot columns. No imputer and no scaler: the model sends NaN
    (years_since_new_style, new_style_rate_so_far, n_new_styles_year and top_style_share) down a
    learned branch at every split, and trees do not need scaling. The values are the WHEN common
    starting points, not the result of any search. `features` is the output of `build_what_features`;
    `feature_columns` must not contain any of `WHAT_FORBIDDEN_INPUTS` or a repeated name. Returns
    `(fit, score)` with the signatures `evaluate_folds` expects: `fit(train)` returns the fitted
    model, `score(test, fitted)` the probability of `target` == 1.
    """
    assert not set(feature_columns) & set(WHAT_FORBIDDEN_INPUTS), "a forbidden column is in the feature list"  # stop if a forbidden column is requested
    assert len(set(feature_columns)) == len(feature_columns), "the feature list has repeated names"  # stop if a column name is repeated

    def fit(train: pd.DataFrame):
        """Fit a fresh HistGradientBoosting model on the WHAT matrix of the train rows. Input: the train dataframe. Output: the fitted model."""
        model = HistGradientBoostingClassifier(  # fresh boosting model with the stage 0 settings
            learning_rate=0.1,      # Same: step size of each new tree.
            max_iter=200,           # Same: number of boosting rounds.
            max_leaf_nodes=31,      # Same: leaves per tree.
            min_samples_leaf=50,    # Same: minimum rows in a leaf, as in the Random Forest.
            early_stopping=False,   # Same: off, all train rows are used and no validation split is carved out.
            random_state=SEED,      # Same: reproducible runs.
        )
        return model.fit(_what_matrix(train, features, feature_columns), train[target])  # fitted on the fold's train rows only

    def score(test: pd.DataFrame, fitted):
        """Score test rows on the WHAT matrix with a fitted model. Inputs: the test dataframe and the fitted model. Output: the probability of target == 1 for each row."""
        return fitted.predict_proba(_what_matrix(test, features, feature_columns))[:, 1]  # probability target == 1

    return fit, score  # return both functions for the folds
