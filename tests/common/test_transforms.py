import numpy as np
import pandas as pd
import pytest

from market_regime_detector.common.transforms import (
    apply_transform,
    detrend,
    diff,
    identity,
    pct_change,
    pct_change_annualized,
    rolling_mean_diff,
    rolling_zscore,
    second_diff,
)


def _series(values: list[float], freq: str = "MS") -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=index)


def test_identity_returns_series_unchanged() -> None:
    s = _series([1.0, 2.0, 3.0])
    pd.testing.assert_series_equal(identity(s), s)


def test_pct_change() -> None:
    s = _series([100.0, 110.0, 121.0, 133.1])
    result = pct_change(s, n_periods=3)
    assert result.iloc[3] == pytest.approx(0.331)


def test_diff() -> None:
    s = _series([2.0, 2.1, 2.3, 2.4])
    result = diff(s, n_periods=1)
    assert result.iloc[1] == pytest.approx(0.1)
    assert result.iloc[3] == pytest.approx(0.1)


def test_pct_change_annualized_matches_hand_computed_formula() -> None:
    s = _series([100.0, 101.0, 102.0, 104.0])
    result = pct_change_annualized(s, n_periods=3, periods_per_year=12)

    expected = (104.0 / 100.0) ** (12 / 3) - 1
    assert result.iloc[3] == pytest.approx(expected)


def test_rolling_mean_diff() -> None:
    # 4-week rolling mean: [1,2,3,4]->2.5, [2,3,4,5]->3.5, [3,4,5,6]->4.5
    s = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], freq="W")
    result = rolling_mean_diff(s, window=4, diff_periods=1)

    assert result.iloc[4] == pytest.approx(1.0)  # 3.5 - 2.5
    assert result.iloc[5] == pytest.approx(1.0)  # 4.5 - 3.5


def test_detrend_subtracts_trailing_rolling_mean() -> None:
    s = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = detrend(s, window=3)

    # rolling mean of [3,4,5] at index 4 is 4.0
    assert result.iloc[4] == pytest.approx(5.0 - 4.0)


def test_detrend_is_causal_appending_future_rows_does_not_change_earlier_values() -> None:
    rng = np.random.default_rng(0)
    full = _series(list(rng.normal(size=30)))
    short = full.iloc[:20]

    result_short = detrend(short, window=12)
    result_full = detrend(full, window=12)

    pd.testing.assert_series_equal(result_short, result_full.loc[result_short.index])


def test_second_diff_is_diff_applied_twice() -> None:
    s = _series([1.0, 2.0, 4.0, 7.0, 11.0])  # first diffs: 1,2,3,4 -> second diffs: 1,1,1
    result = second_diff(s, n_periods=1)

    assert result.iloc[2] == pytest.approx(1.0)
    assert result.iloc[3] == pytest.approx(1.0)
    assert result.iloc[4] == pytest.approx(1.0)


def test_second_diff_is_causal_appending_future_rows_does_not_change_earlier_values() -> None:
    rng = np.random.default_rng(0)
    full = _series(list(rng.normal(size=30)))
    short = full.iloc[:20]

    result_short = second_diff(short, n_periods=1)
    result_full = second_diff(full, n_periods=1)

    pd.testing.assert_series_equal(result_short, result_full.loc[result_short.index])


def test_rolling_zscore_matches_hand_computed_formula() -> None:
    s = _series([1.0, 2.0, 3.0, 4.0, 10.0])
    result = rolling_zscore(s, window=3)

    window = s.iloc[2:5]
    expected = (window.iloc[-1] - window.mean()) / window.std()
    assert result.iloc[4] == pytest.approx(expected)


def test_rolling_zscore_is_causal_appending_future_rows_does_not_change_earlier_values() -> None:
    rng = np.random.default_rng(0)
    full = _series(list(rng.normal(size=30)))
    short = full.iloc[:20]

    result_short = rolling_zscore(short, window=12)
    result_full = rolling_zscore(full, window=12)

    pd.testing.assert_series_equal(result_short, result_full.loc[result_short.index])


def test_apply_transform_dispatches_by_name() -> None:
    s = _series([10.0, 11.0])
    result = apply_transform(s, "diff", {"n_periods": 1})
    assert result.iloc[1] == pytest.approx(1.0)


def test_apply_transform_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown transform"):
        apply_transform(_series([1.0]), "not_a_real_transform", {})
