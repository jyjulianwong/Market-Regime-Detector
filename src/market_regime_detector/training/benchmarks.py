import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from market_regime_detector.common.feature_config import FeatureDefinition
from market_regime_detector.common.scaling import DEFAULT_CLIP, apply_scaler, fit_robust_scaler

logger = logging.getLogger(__name__)

# A fixed, never-fitted slope for `fixed_rule_baseline`'s logistic mapping —
# illustrative, not tuned. The point of this baseline is a rule that was never
# shown any scores, so AUC/accuracy (scale-free) are its headline numbers, not
# log loss (which this arbitrary constant directly controls the sharpness of).
RULE_SLOPE = 2.0


def climatology_baseline(
    label: pd.Series,
    index: pd.DatetimeIndex,
    label_lag_months: int,
    initial_window_months: int,
) -> pd.Series:
    """Expanding mean of the label, shifted by `label_lag_months` positions on
    `index` so only labels knowable as of each row are averaged — e.g. at decision
    month-end `t` with `label_lag_months=1`, only labels through month `t-1`
    contribute. Falls back to 0.5 before any label is yet knowable. Trimmed to
    `index[initial_window_months:]` to match the HMM walk-forward's own test
    months (`training/evaluation.walk_forward_evaluate` never predicts the
    training-only prefix either).
    """
    known = label.reindex(index).shift(label_lag_months)
    p_up = known.expanding(min_periods=1).mean().fillna(0.5)
    return p_up.iloc[initial_window_months:]


def calibrated_persistence_baseline(
    label: pd.Series,
    index: pd.DatetimeIndex,
    label_lag_months: int,
    initial_window_months: int,
) -> pd.Series:
    """`p_t = P(label=1 | last known label)`. "Last known label" at row `t` is the
    label for month `t - label_lag_months` (the most recent one actually knowable
    by `t`); the conditional probability is re-estimated at every row from an
    *expanding* window of (prev, next) label transitions seen so far — only
    transitions between labels already knowable by `t` — with Laplace (add-1)
    smoothing so a transition never yet observed from a given state doesn't come
    out as a hard 0% or 100% (and so an infinite log loss on the first miss).

    A plain hard-persistence baseline (`p = last known label` directly) would
    have exactly this infinite-log-loss problem whenever a 0/1 call is ever wrong
    — the whole reason this is "calibrated" persistence, not literal persistence.
    """
    known = label.reindex(index).shift(label_lag_months)
    p_up = pd.Series(index=index, dtype=float)

    for i in range(len(index)):
        history = known.iloc[: i + 1].dropna()
        if len(history) < 2:
            p_up.iloc[i] = 0.5
            continue
        prev_values = history.to_numpy()[:-1]
        next_values = history.to_numpy()[1:]
        last_known = history.iloc[-1]

        matching_next = next_values[prev_values == last_known]
        n_up = int((matching_next == 1.0).sum())
        n_total = len(matching_next)
        p_up.iloc[i] = (n_up + 1) / (n_total + 2)

    return p_up.iloc[initial_window_months:]


def fixed_rule_baseline(
    axis_df: pd.DataFrame,
    feature_defs: list[FeatureDefinition],
    initial_window_months: int,
    min_observations: int = 24,
) -> pd.Series:
    """Per-feature expanding (causal) z-score — using only each feature's own
    mean/std up to and including that row, never a future one — sign-weighted and
    averaged into a composite, mapped to a probability with a *fixed*,
    never-fitted logistic `p = 1/(1+exp(-RULE_SLOPE*composite))`.

    This replaces `training/baseline.py` for scoring purposes: that module
    averages *unscaled* feature values, which `training/baseline.py`'s own module
    docstring already flags as dominated by whichever feature happens to have the
    largest native units (e.g. ICSA's raw claims count). Standardizing every
    feature before combining it is the fix. A row is `NaN` until every feature has
    at least `min_observations` expanding-window observations.
    """
    relevant = [f for f in feature_defs if f.name in axis_df.columns]
    signs = pd.Series({f.name: f.sign for f in relevant})
    sub = axis_df[[f.name for f in relevant]]

    expanding_mean = sub.expanding(min_periods=min_observations).mean()
    expanding_std = sub.expanding(min_periods=min_observations).std()
    z = (sub - expanding_mean) / expanding_std

    composite = (z * signs).mean(axis=1)
    p_up = 1.0 / (1.0 + np.exp(-RULE_SLOPE * composite))
    result: pd.Series = p_up.iloc[initial_window_months:]
    return result


def logistic_baseline(
    axis_df: pd.DataFrame,
    feature_defs: list[FeatureDefinition],
    label: pd.Series,
    label_lag_months: int,
    *,
    initial_window_months: int = 60,
    refit_every_months: int = 12,
    scaler_clip: float = DEFAULT_CLIP,
    scaler_method: str = "median_mad",
) -> pd.DataFrame:
    """Walk-forward `sklearn.linear_model.LogisticRegression` (L2, `C=1.0`) on the
    same axis features, standardized with the *same* `common.scaling` robust
    scaler + clip as the HMM (`HMM_ACCURACY_PLAN.md` Phase 1 — a plain
    `StandardScaler` here would make "agreement with the HMM" partly an artifact
    of which model's scaling handled outliers better, not a fair comparison),
    fit on each fold's own training rows — the *same* fold boundaries as
    `training/evaluation.walk_forward_evaluate`, so its predictions are directly
    comparable to the HMM's.

    For a fold starting at row `s`, training rows are `axis_df.index[: s -
    label_lag_months]` — the training window's own end is pulled back by the
    label lag, not each row's target shifted — so every training example still
    pairs a row's features with *that same row's* true label (never a
    mismatched, lagged one), while excluding the handful of rows right at the
    window's edge whose true label might not actually be revealed yet as of `s`.

    A fold is skipped (and recorded, the same shape as
    `walk_forward_evaluate.attrs["skipped_folds"]`) when its training labels
    contain a single class — `LogisticRegression` can't fit on one.
    """
    feature_names = [f.name for f in feature_defs if f.name in axis_df.columns]
    true_label = label.reindex(axis_df.index)
    n = len(axis_df)

    rows: list[dict[str, object]] = []
    skipped_folds: list[dict[str, object]] = []

    fold_id = 0
    fold_start = initial_window_months
    while fold_start < n:
        fold_end = min(fold_start + refit_every_months, n)
        train_cutoff = fold_start - label_lag_months
        train_idx = axis_df.index[:train_cutoff]
        test_idx = axis_df.index[fold_start:fold_end]

        train_labels = true_label.loc[train_idx].dropna()
        if train_labels.nunique() < 2:
            logger.warning(
                "logistic baseline fold %d (train through %s, test %s to %s) skipped: "
                "training labels contain a single class",
                fold_id,
                train_idx.max().date() if len(train_idx) else None,
                test_idx.min().date(),
                test_idx.max().date(),
            )
            skipped_folds.append(
                {
                    "fold_id": fold_id,
                    "train_end": str(train_idx.max().date()) if len(train_idx) else None,
                    "test_start": str(test_idx.min().date()),
                    "test_end": str(test_idx.max().date()),
                    "reason": "training labels contain a single class",
                }
            )
            fold_id += 1
            fold_start = fold_end
            continue

        train_X = axis_df.loc[train_labels.index, feature_names].to_numpy()
        center, scale = fit_robust_scaler(train_X, clip=scaler_clip, method=scaler_method)
        train_X_scaled = apply_scaler(train_X, center, scale, scaler_clip)

        # penalty="l2" is sklearn's own default (and the one explicit `penalty=`
        # kwarg is deprecated since 1.8, slated for removal in 1.10) — omitted
        # rather than passed, same model either way.
        model = LogisticRegression(C=1.0)
        model.fit(train_X_scaled, train_labels.to_numpy())

        test_X_scaled = apply_scaler(
            axis_df.loc[test_idx, feature_names].to_numpy(), center, scale, scaler_clip
        )
        up_column = list(model.classes_).index(1.0)
        p_up = model.predict_proba(test_X_scaled)[:, up_column]

        for decision_date, p in zip(test_idx, p_up, strict=True):
            rows.append(
                {
                    "decision_date": decision_date,
                    "p_up": float(p),
                    "label": "UP" if p >= 0.5 else "DOWN",
                    "fold_id": fold_id,
                }
            )

        fold_id += 1
        fold_start = fold_end

    result = pd.DataFrame(rows)
    result.attrs["skipped_folds"] = skipped_folds
    result.attrs["n_folds_attempted"] = fold_id
    return result
