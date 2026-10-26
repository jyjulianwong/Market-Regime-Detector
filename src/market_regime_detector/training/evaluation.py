import logging

import pandas as pd

from market_regime_detector.common.feature_config import FeatureDefinition
from market_regime_detector.common.forward_filter import filtered_posterior
from market_regime_detector.training.trainer import fit_hmm

logger = logging.getLogger(__name__)


def walk_forward_evaluate(
    df: pd.DataFrame,
    feature_defs: list[FeatureDefinition],
    *,
    initial_window_months: int = 60,
    refit_every_months: int = 12,
    n_restarts: int = 10,
    covariance_type: str = "diag",
    min_state_duration_months: float = 2.0,
    composite_separation_margin: float = 0.1,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Expanding-window walk-forward evaluation (brief section 2.6): refits
    periodically on an expanding window, produces **causal filtered posteriors**
    (never `hmmlearn.predict_proba`, per section 2.5) for each out-of-sample month,
    and aggregates them into one DataFrame (`decision_date`, `p_up`, `label`,
    `fold_id`).

    **The filter keeps its memory across refit boundaries.** After each fold's fit,
    `filtered_posterior` runs over *every* month from the start of `df` through the
    end of that fold's test window — standardized with that fold's own scaler and
    filtered with that fold's own parameters — and only the test months' rows are
    kept. Filtering just the test slice instead would restart the filter from
    `start_prob` at the first month of every fold, throwing away what the model
    already knew about the current regime: one ambiguous month right after a refit
    could then flip `p_up` across 0.5 (a spurious regime change, inflating
    flip rate and log loss) even after a long, clear run in the other regime. That
    is also not what production does — `prediction/predict.py` filters the full
    history — so this keeps the evaluation scoring the same inference path.

    Still fully causal: a fold's parameters and scaler come only from data strictly
    before its test window, and month `t`'s posterior depends only on observations
    `<= t` (the forward recursion never looks ahead). Re-filtering the earlier
    months under the new fold's parameters (rather than reusing the previous fold's
    `alpha`) is deliberate: it avoids carrying an `alpha` computed under different
    emission/transition parameters into the new model.

    **A fold whose expanding window no longer shows two separable regimes is a
    recorded gap, not a fatal error for the whole evaluation**: `fit_hmm` still
    hard-fails with no fallback for that fold's own fit (brief section 2.3's
    sanity checks are never loosened here), but that failure is caught, logged, and
    the fold's out-of-sample months are simply absent from the result rather than
    aborting every other fold. `result.attrs["skipped_folds"]` lists what was
    skipped and why; `result.attrs["n_folds_attempted"]` is the total fold count
    (evaluated + skipped), so a caller can report "N of M folds evaluable" instead
    of silently looking complete. `result.attrs["fold_fits"]` lists, per evaluated
    fold, that fold's own per-feature emission means in z-units keyed by UP/DOWN
    (not raw state index, since the index is arbitrary and can flip between
    refits) — `training/occupancy.py` turns this plus the row-level `label` column
    into the state-occupancy report.
    """
    n = len(df)
    rows: list[dict[str, object]] = []
    skipped_folds: list[dict[str, object]] = []
    fold_fits: list[dict[str, object]] = []

    fold_id = 0
    fold_start = initial_window_months
    while fold_start < n:
        fold_end = min(fold_start + refit_every_months, n)
        train_df = df.iloc[:fold_start]
        test_df = df.iloc[fold_start:fold_end]

        try:
            fit_result = fit_hmm(
                train_df,
                feature_defs,
                n_restarts=n_restarts,
                covariance_type=covariance_type,
                min_state_duration_months=min_state_duration_months,
                composite_separation_margin=composite_separation_margin,
                random_state=random_state,
            )
        except ValueError as exc:
            logger.warning(
                "walk-forward fold %d (train through %s, test %s to %s) skipped: %s",
                fold_id,
                train_df.index.max().date(),
                test_df.index.min().date(),
                test_df.index.max().date(),
                exc,
            )
            skipped_folds.append(
                {
                    "fold_id": fold_id,
                    "train_end": str(train_df.index.max().date()),
                    "test_start": str(test_df.index.min().date()),
                    "test_end": str(test_df.index.max().date()),
                    "reason": str(exc),
                }
            )
            fold_id += 1
            fold_start = fold_end
            continue

        # Filter from the very first month through the end of this test window so the
        # filter carries its memory across the refit boundary (see the docstring), then
        # keep only the out-of-sample rows.
        history_df = df.iloc[:fold_end]
        observations = (
            history_df[fit_result.feature_names].to_numpy() - fit_result.scaler_mean
        ) / fit_result.scaler_scale
        posterior = filtered_posterior(
            observations,
            fit_result.start_prob,
            fit_result.transition_matrix,
            fit_result.means,
            fit_result.variances,
        )[fold_start:]

        up_state = next(s for s, label in fit_result.state_labels.items() if label == "UP")
        p_up = posterior[:, up_state]

        fold_fits.append(
            {
                "fold_id": fold_id,
                "test_start": str(test_df.index.min().date()),
                "test_end": str(test_df.index.max().date()),
                "state_means": {
                    fit_result.state_labels[state]: {
                        name: float(fit_result.means[state, i])
                        for i, name in enumerate(fit_result.feature_names)
                    }
                    for state in fit_result.state_labels
                },
            }
        )

        for decision_date, p in zip(test_df.index, p_up, strict=True):
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
    result.attrs["fold_fits"] = fold_fits
    return result
