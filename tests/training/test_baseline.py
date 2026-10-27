import numpy as np
import pandas as pd

from market_regime_detector.common.feature_config import Axis, FeatureDefinition, Frequency
from market_regime_detector.training.baseline import baseline_signal


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


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-31", periods=n, freq="ME")


def test_baseline_signal_drops_rows_before_min_observations() -> None:
    feature_defs = [_feature("a", sign=1)]
    axis_df = pd.DataFrame({"a": np.arange(10, dtype=float)}, index=_dates(10))

    result = baseline_signal(axis_df, feature_defs, min_observations=5)

    assert len(result) == len(axis_df) - 4  # rows 0-3 have < 5 expanding observations
    assert list(result.index) == list(axis_df.index[4:])


def test_baseline_signal_ignores_features_not_in_axis_df() -> None:
    feature_defs = [_feature("a", sign=1), _feature("unused", sign=1)]
    axis_df = pd.DataFrame({"a": np.concatenate([np.zeros(5), [2.0]])}, index=_dates(6))

    result = baseline_signal(axis_df, feature_defs, min_observations=5)

    assert result.iloc[0] == "UP"  # z-score of a positive deviation from the mean


def test_baseline_signal_tie_rounds_to_up() -> None:
    feature_defs = [_feature("a", sign=1), _feature("b", sign=-1)]
    # a and b always move together -> their signed contributions always cancel to 0.
    values = np.concatenate([np.zeros(5), [1.0]])
    axis_df = pd.DataFrame({"a": values, "b": values}, index=_dates(6))

    result = baseline_signal(axis_df, feature_defs, min_observations=5)

    assert result.iloc[0] == "UP"


def test_baseline_signal_large_unit_feature_does_not_dominate_the_composite() -> None:
    """HMM_ACCURACY_PLAN.md Phase 2's problem: averaging *raw* feature values
    lets whichever feature has the largest native units (e.g. ICSA's raw claims
    count) swamp every other feature. Standardizing first should let a
    small-magnitude feature's sign still determine the composite even when a
    large-magnitude feature's raw value is far bigger.
    """
    n = 30
    rng = np.random.default_rng(0)
    # "icsa"-like: large units, no real signal (flat, tiny noise).
    icsa = 1_000_000.0 + rng.normal(scale=1.0, size=n)
    # "payems"-like: small units, a clear late upward move.
    payems = np.concatenate([np.zeros(n - 1), [5.0]])
    axis_df = pd.DataFrame({"icsa": icsa, "payems": payems}, index=_dates(n))
    feature_defs = [_feature("icsa", sign=-1), _feature("payems", sign=1)]

    result = baseline_signal(axis_df, feature_defs, min_observations=5)

    # icsa's raw scale (~1e6) would dominate an unscaled composite regardless of
    # sign; standardized, its near-zero-variance series contributes a near-zero
    # z-score, so payems' sharp upward move actually decides the label.
    assert result.iloc[-1] == "UP"


def test_baseline_signal_is_causal_appending_future_rows_does_not_change_earlier_labels() -> None:
    feature_defs = [_feature("a", sign=1)]
    rng = np.random.default_rng(0)
    values = rng.normal(size=40)
    axis_df = pd.DataFrame({"a": values}, index=_dates(40))

    short = baseline_signal(axis_df.iloc[:30], feature_defs, min_observations=5)
    full = baseline_signal(axis_df, feature_defs, min_observations=5)

    pd.testing.assert_series_equal(short, full.loc[short.index])
