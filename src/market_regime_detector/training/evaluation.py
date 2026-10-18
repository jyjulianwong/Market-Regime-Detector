import pandas as pd

from market_regime_detector.common.feature_config import FeatureDefinition
from market_regime_detector.common.forward_filter import filtered_posterior
from market_regime_detector.training.trainer import fit_hmm


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

    **Simplification**: each refit segment is forward-filtered
    independently from its own start using that fit's own `start_prob`, rather than
    one continuous filter carried across refit boundaries — avoids reconciling whose
    `alpha` carries forward across a parameter change. Still fully causal: a
    segment's model is fit only on data strictly before that segment begins, and the
    segment's own posterior never looks past its own out-of-sample window.
    """
    n = len(df)
    rows: list[dict[str, object]] = []

    fold_id = 0
    fold_start = initial_window_months
    while fold_start < n:
        fold_end = min(fold_start + refit_every_months, n)
        train_df = df.iloc[:fold_start]
        test_df = df.iloc[fold_start:fold_end]

        fit_result = fit_hmm(
            train_df,
            feature_defs,
            n_restarts=n_restarts,
            covariance_type=covariance_type,
            min_state_duration_months=min_state_duration_months,
            composite_separation_margin=composite_separation_margin,
            random_state=random_state,
        )

        observations = (
            test_df[fit_result.feature_names].to_numpy() - fit_result.scaler_mean
        ) / fit_result.scaler_scale
        posterior = filtered_posterior(
            observations,
            fit_result.start_prob,
            fit_result.transition_matrix,
            fit_result.means,
            fit_result.variances,
        )

        up_state = next(s for s, label in fit_result.state_labels.items() if label == "UP")
        p_up = posterior[:, up_state]

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

    return pd.DataFrame(rows)
