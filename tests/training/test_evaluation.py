import numpy as np
import pandas as pd

from market_regime_detector.common.feature_config import Axis, FeatureDefinition, Frequency
from market_regime_detector.common.forward_filter import filtered_posterior
from market_regime_detector.training.evaluation import walk_forward_evaluate
from market_regime_detector.training.trainer import fit_hmm


def _feature(name: str, sign: int) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        axis=Axis.GROWTH,
        series_id=name.upper(),
        source="fred",
        native_frequency=Frequency.MONTHLY,
        vintage_aware=False,
        publication_lag_days=0,
        transform="identity",
        transform_params={},
        sign=sign,
    )


def _synthetic_two_regime_df(
    n_per_block: int = 12, n_blocks: int = 10, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    blocks = []
    for block in range(n_blocks):
        mean = 2.0 if block % 2 == 0 else -2.0
        blocks.append(rng.normal(loc=mean, scale=0.3, size=(n_per_block, 2)))
    data = np.concatenate(blocks, axis=0)
    dates = pd.date_range("2000-01-31", periods=len(data), freq="ME")
    return pd.DataFrame(data, columns=["f1", "f2"], index=dates)


FEATURE_DEFS = [_feature("f1", sign=1), _feature("f2", sign=1)]


def test_walk_forward_evaluate_covers_every_out_of_sample_row() -> None:
    df = _synthetic_two_regime_df(n_per_block=12, n_blocks=10)  # 120 rows

    result = walk_forward_evaluate(
        df,
        FEATURE_DEFS,
        initial_window_months=60,
        refit_every_months=12,
        random_state=0,
    )

    # rows 60..119 are out-of-sample -> 60 rows total, in 5 folds of 12
    assert len(result) == 60
    assert result["fold_id"].nunique() == 5
    assert list(result["decision_date"]) == list(df.index[60:])


def test_walk_forward_evaluate_output_columns_and_ranges() -> None:
    df = _synthetic_two_regime_df(n_per_block=12, n_blocks=10)

    result = walk_forward_evaluate(
        df,
        FEATURE_DEFS,
        initial_window_months=60,
        refit_every_months=12,
        random_state=0,
    )

    assert set(result.columns) == {"decision_date", "p_up", "label", "fold_id"}
    assert result["p_up"].between(0, 1).all()
    assert set(result["label"].unique()) <= {"UP", "DOWN"}


def test_walk_forward_evaluate_filter_carries_memory_across_fold_boundary() -> None:
    df = _synthetic_two_regime_df(n_per_block=12, n_blocks=10)

    result = walk_forward_evaluate(
        df,
        FEATURE_DEFS,
        initial_window_months=60,
        refit_every_months=12,
        random_state=0,
    )

    # Re-derive fold 0's posterior two ways from the same fit: a continuous filter
    # over history + test window must match the returned rows, not a test-only restart.
    fold0 = result[result["fold_id"] == 0]
    fit_result = fit_hmm(df.iloc[:60], FEATURE_DEFS, random_state=0)
    scaled = (df.iloc[:72][fit_result.feature_names].to_numpy() - fit_result.scaler_mean) / (
        fit_result.scaler_scale
    )
    args = (
        fit_result.start_prob,
        fit_result.transition_matrix,
        fit_result.means,
        fit_result.variances,
    )
    up_state = next(s for s, label in fit_result.state_labels.items() if label == "UP")
    continuous = filtered_posterior(scaled, *args)[60:, up_state]
    np.testing.assert_allclose(fold0["p_up"].to_numpy(), continuous)


def test_walk_forward_evaluate_empty_when_not_enough_history() -> None:
    df = _synthetic_two_regime_df(n_per_block=5, n_blocks=4)  # 20 rows

    result = walk_forward_evaluate(
        df,
        FEATURE_DEFS,
        initial_window_months=60,
        refit_every_months=12,
        random_state=0,
    )

    assert result.empty
