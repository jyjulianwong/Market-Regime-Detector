import math

import numpy as np
import pandas as pd
import pytest

from market_regime_detector.common.feature_config import Axis, FeatureDefinition, Frequency
from market_regime_detector.training.benchmarks import (
    RULE_SLOPE,
    calibrated_persistence_baseline,
    climatology_baseline,
    fixed_rule_baseline,
    logistic_baseline,
)


def _monthly_index(start: str, n: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


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


def test_climatology_baseline_hand_computed() -> None:
    index = _monthly_index("2020-01-31", 5)
    label = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0], index=index)

    p_up = climatology_baseline(label, index, label_lag_months=1, initial_window_months=0)

    # known (label shifted by 1) = [NaN, 1, 0, 1, 0]; expanding mean (skipping
    # NaN, min_periods=1), first row falls back to 0.5 since nothing is knowable.
    expected = [0.5, 1.0, 0.5, 2.0 / 3.0, 0.5]
    np.testing.assert_allclose(p_up.to_numpy(), expected, atol=1e-10)


def test_climatology_baseline_trims_to_test_months() -> None:
    index = _monthly_index("2020-01-31", 5)
    label = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0], index=index)

    p_up = climatology_baseline(label, index, label_lag_months=1, initial_window_months=3)

    assert len(p_up) == 2
    assert p_up.index[0] == index[3]


def test_climatology_baseline_is_causal_to_a_later_mutation() -> None:
    index = _monthly_index("2020-01-31", 10)
    label = pd.Series([1.0, 0.0] * 5, index=index)
    mutated = label.copy()
    # The second-to-last month is what the 1-month lag makes knowable for the
    # *last* output row — flipping it must leave every earlier output row
    # untouched, and change exactly that last row (confirming the function is
    # sensitive to it at all, not just coincidentally ignoring everything).
    mutated.iloc[-2] = 1.0 - mutated.iloc[-2]

    original = climatology_baseline(label, index, label_lag_months=1, initial_window_months=0)
    after_mutation = climatology_baseline(
        mutated, index, label_lag_months=1, initial_window_months=0
    )

    pd.testing.assert_series_equal(original.iloc[:-1], after_mutation.iloc[:-1])
    assert original.iloc[-1] != after_mutation.iloc[-1]


def test_calibrated_persistence_baseline_hand_computed() -> None:
    index = _monthly_index("2020-01-31", 5)
    label = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0], index=index)

    p_up = calibrated_persistence_baseline(
        label, index, label_lag_months=1, initial_window_months=0
    )

    # known = [NaN, 1, 0, 1, 0]. Row-by-row (see training/benchmarks.py's
    # docstring for the transition-counting logic):
    #   i=0: history=[] -> 0.5
    #   i=1: history=[1] (len 1) -> 0.5
    #   i=2: history=[1,0]; last_known=0, no prior "prev==0" transition -> (0+1)/(0+2)=0.5
    #   i=3: history=[1,0,1]; last_known=1, one "prev==1"->0 transition -> (0+1)/(1+2)=1/3
    #   i=4: history=[1,0,1,0]; last_known=0, one "prev==0"->1 transition -> (1+1)/(1+2)=2/3
    expected = [0.5, 0.5, 0.5, 1.0 / 3.0, 2.0 / 3.0]
    np.testing.assert_allclose(p_up.to_numpy(), expected, atol=1e-10)


def test_calibrated_persistence_baseline_is_causal_to_a_later_mutation() -> None:
    index = _monthly_index("2020-01-31", 10)
    label = pd.Series([1.0, 0.0] * 5, index=index)
    mutated = label.copy()
    mutated.iloc[-2] = 1.0 - mutated.iloc[-2]

    original = calibrated_persistence_baseline(
        label, index, label_lag_months=1, initial_window_months=0
    )
    after_mutation = calibrated_persistence_baseline(
        mutated, index, label_lag_months=1, initial_window_months=0
    )

    pd.testing.assert_series_equal(original.iloc[:-1], after_mutation.iloc[:-1])
    assert original.iloc[-1] != after_mutation.iloc[-1]


def test_fixed_rule_baseline_nan_before_warmup_then_hand_computed_value() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    index = _monthly_index("2020-01-31", len(values))
    axis_df = pd.DataFrame({"a": values}, index=index)
    feature_defs = [_feature("a", sign=1)]

    p_up = fixed_rule_baseline(axis_df, feature_defs, initial_window_months=0, min_observations=5)

    assert p_up.iloc[:4].isna().all()

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    z = (values[-1] - mean) / math.sqrt(variance)
    expected_p = 1.0 / (1.0 + math.exp(-RULE_SLOPE * z))
    assert p_up.iloc[-1] == pytest.approx(expected_p, abs=1e-10)


def test_fixed_rule_baseline_sign_flips_the_direction() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]  # a sharp upward deviation at the end
    index = _monthly_index("2020-01-31", len(values))
    axis_df = pd.DataFrame({"a": values}, index=index)

    p_up_positive_sign = fixed_rule_baseline(
        axis_df, [_feature("a", sign=1)], initial_window_months=0, min_observations=5
    )
    p_up_negative_sign = fixed_rule_baseline(
        axis_df, [_feature("a", sign=-1)], initial_window_months=0, min_observations=5
    )

    assert p_up_positive_sign.iloc[-1] > 0.5
    assert p_up_negative_sign.iloc[-1] < 0.5


def test_fixed_rule_baseline_is_causal_to_a_later_mutation() -> None:
    values = [float(i) for i in range(12)]
    index = _monthly_index("2020-01-31", len(values))
    axis_df = pd.DataFrame({"a": values}, index=index)
    mutated_df = axis_df.copy()
    mutated_df.loc[index[-1], "a"] = 999.0

    original = fixed_rule_baseline(
        axis_df, [_feature("a", sign=1)], initial_window_months=0, min_observations=5
    )
    after_mutation = fixed_rule_baseline(
        mutated_df, [_feature("a", sign=1)], initial_window_months=0, min_observations=5
    )

    pd.testing.assert_series_equal(original.iloc[:-1], after_mutation.iloc[:-1])


def _logistic_fixture(n_months: int = 90) -> tuple[pd.DataFrame, pd.Series]:
    index = _monthly_index("2015-01-31", n_months)
    rng = np.random.default_rng(0)
    values = rng.normal(size=n_months)
    axis_df = pd.DataFrame({"a": values}, index=index)
    label = pd.Series((values > 0).astype(float), index=index)
    return axis_df, label


def test_logistic_baseline_matches_walk_forward_fold_shape() -> None:
    axis_df, label = _logistic_fixture(n_months=90)
    feature_defs = [_feature("a", sign=1)]

    result = logistic_baseline(
        axis_df,
        feature_defs,
        label,
        label_lag_months=1,
        initial_window_months=60,
        refit_every_months=12,
    )

    assert list(result.columns) == ["decision_date", "p_up", "label", "fold_id"]
    # Fold boundaries: [60:72), [72:84), [84:90) -> 3 folds, 30 test rows total.
    assert result.attrs["n_folds_attempted"] == 3
    assert len(result) == 30
    assert set(result["fold_id"]) == {0, 1, 2}


def test_logistic_baseline_is_causal_across_fold_boundaries() -> None:
    """Mutating data inside/after a later fold's test window must not change an
    earlier fold's training or predictions — the walk-forward property that
    matters most."""
    axis_df, label = _logistic_fixture(n_months=90)
    feature_defs = [_feature("a", sign=1)]

    mutated_df = axis_df.copy()
    mutated_label = label.copy()
    # Mutate everything from row 75 onward (well inside fold 1's test window,
    # [72:84)) — fold 0's train [0:59] and test [60:72) rows are untouched.
    mutated_df.loc[mutated_df.index[75:], "a"] = 999.0
    mutated_label.iloc[75:] = 1.0 - mutated_label.iloc[75:]

    original = logistic_baseline(axis_df, feature_defs, label, label_lag_months=1)
    after_mutation = logistic_baseline(mutated_df, feature_defs, mutated_label, label_lag_months=1)

    original_fold0 = original[original["fold_id"] == 0].reset_index(drop=True)
    mutated_fold0 = after_mutation[after_mutation["fold_id"] == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_fold0, mutated_fold0)


def test_logistic_baseline_skips_a_fold_with_single_class_training_labels() -> None:
    # Fold 0: train [0:59], test [60:72). Fold 1: train [0:71], test [72:84).
    n_months = 84
    index = _monthly_index("2015-01-31", n_months)
    axis_df = pd.DataFrame({"a": np.linspace(0, 1, n_months)}, index=index)
    # Constant 0.0 for the first 60 months, so fold 0's entire (lag-truncated)
    # training window [0:59] is single-class and must be skipped. Months 60-83
    # alternate, so fold 1's training window [0:71] — which includes months
    # 60-70 — has both classes and proceeds normally.
    label_values = [0.0] * 60 + [0.0, 1.0] * 12
    label = pd.Series(label_values, index=index)

    result = logistic_baseline(
        axis_df, [_feature("a", sign=1)], label, label_lag_months=1, initial_window_months=60
    )

    assert len(result.attrs["skipped_folds"]) == 1
    assert result.attrs["skipped_folds"][0]["fold_id"] == 0
    assert "single class" in result.attrs["skipped_folds"][0]["reason"]
    assert 0 not in set(result["fold_id"])
    assert 1 in set(result["fold_id"])
    assert len(result) == 12
