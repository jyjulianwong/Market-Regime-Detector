import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from market_regime_detector.training.scoring import (
    CLIP_EPS,
    accuracy,
    auc,
    block_bootstrap_ci,
    brier,
    calibration_table,
    detection_lag,
    flip_rate_per_year,
    log_loss,
    mean_regime_duration_months,
    quadrant_log_loss,
    skill_score,
)


def _series(values: list[float], start: str = "2020-01-31") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="ME"))


def test_log_loss_hand_computed() -> None:
    # p=0.8,y=1 and p=0.2,y=0 both contribute -ln(0.8) by symmetry.
    p, y = _series([0.8, 0.2]), _series([1.0, 0.0])

    assert log_loss(p, y) == pytest.approx(-math.log(0.8), abs=1e-10)


def test_log_loss_clips_extreme_probabilities() -> None:
    p, y = _series([1.0]), _series([0.0])

    # Without clipping this would be -ln(0) = inf; clipped to CLIP_EPS it's -ln(CLIP_EPS).
    assert log_loss(p, y) == pytest.approx(-math.log(CLIP_EPS), abs=1e-10)
    assert math.isfinite(log_loss(p, y))


def test_log_loss_drops_unaligned_and_nan_rows() -> None:
    p = _series([0.8, 0.9, 0.2], start="2020-01-31")
    y = pd.Series([1.0, float("nan")], index=pd.date_range("2020-01-31", periods=2, freq="ME"))

    # Only 2020-01-31 survives: present in both, and non-NaN in both.
    assert log_loss(p, y) == pytest.approx(-math.log(0.8), abs=1e-10)


def test_brier_hand_computed() -> None:
    p, y = _series([0.8, 0.3]), _series([1.0, 0.0])

    assert brier(p, y) == pytest.approx(((0.8 - 1.0) ** 2 + (0.3 - 0.0) ** 2) / 2, abs=1e-10)


def test_skill_score_hand_computed() -> None:
    assert skill_score(model_loss=0.5, climatology_loss=1.0) == pytest.approx(0.5)
    assert skill_score(model_loss=1.0, climatology_loss=1.0) == pytest.approx(0.0)
    assert skill_score(model_loss=2.0, climatology_loss=1.0) == pytest.approx(-1.0)


def test_auc_matches_sklearn_on_a_perfectly_separable_sample() -> None:
    p, y = _series([0.1, 0.2, 0.8, 0.9]), _series([0.0, 0.0, 1.0, 1.0])

    result = auc(p, y)

    assert result == pytest.approx(roc_auc_score([0.0, 0.0, 1.0, 1.0], [0.1, 0.2, 0.8, 0.9]))
    assert result == pytest.approx(1.0)


def test_auc_is_none_with_a_single_class() -> None:
    p, y = _series([0.1, 0.5, 0.9]), _series([1.0, 1.0, 1.0])

    assert auc(p, y) is None


def test_accuracy_hand_computed() -> None:
    p, y = _series([0.6, 0.4, 0.49]), _series([1.0, 0.0, 1.0])

    # predicted up = [True, False, False] (0.49 < 0.5) vs y = [1, 0, 1] -> 2/3 correct.
    assert accuracy(p, y) == pytest.approx(2.0 / 3.0)


def test_calibration_table_hand_computed_bins_and_ece() -> None:
    p = _series([0.1, 0.2, 0.3, 0.7, 0.8])
    y = _series([0.0, 0.0, 1.0, 1.0, 1.0])

    table, ece = calibration_table(p, y, n_bins=2)

    assert table["n"].tolist() == [3, 2]
    np.testing.assert_allclose(table["mean_p"].to_numpy(), [0.2, 0.75], atol=1e-10)
    np.testing.assert_allclose(table["mean_y"].to_numpy(), [1 / 3, 1.0], atol=1e-10)
    expected_ece = (3 * abs(0.2 - 1 / 3) + 2 * abs(0.75 - 1.0)) / 5
    assert ece == pytest.approx(expected_ece, abs=1e-10)


def test_calibration_table_empty_input_returns_nan_ece() -> None:
    p = pd.Series(dtype=float)
    y = pd.Series(dtype=float)

    table, ece = calibration_table(p, y)

    assert table.empty
    assert math.isnan(ece)


def test_quadrant_log_loss_hand_computed() -> None:
    p_growth = _series([0.8])
    p_inflation = _series([0.3])
    y_growth = _series([1.0])
    y_inflation = _series([0.0])

    # P(true quadrant) = P(growth up) * P(inflation down) = 0.8 * 0.7 = 0.56.
    expected = -math.log(0.8 * 0.7)
    assert quadrant_log_loss(p_growth, p_inflation, y_growth, y_inflation) == pytest.approx(
        expected, abs=1e-10
    )


def test_block_bootstrap_ci_is_deterministic_given_a_seed() -> None:
    values = _series([1.0] * 30 + [-1.0] * 30)

    first = block_bootstrap_ci(values, block=6, seed=42)
    second = block_bootstrap_ci(values, block=6, seed=42)

    assert first == second


def test_block_bootstrap_ci_is_wider_than_block_one_on_an_autocorrelated_series() -> None:
    """A series with two long runs (strong autocorrelation) must give a wider CI
    under real block resampling than under block=1 (effectively i.i.d.) resampling
    — block resampling can draw disproportionately from one run, inflating the
    variance an i.i.d. bootstrap would understate.
    """
    values = _series([1.0] * 30 + [-1.0] * 30)

    _, low_block1, high_block1 = block_bootstrap_ci(values, block=1, seed=0)
    _, low_block10, high_block10 = block_bootstrap_ci(values, block=10, seed=0)

    assert (high_block10 - low_block10) > (high_block1 - low_block1)


def test_block_bootstrap_ci_paired_difference_of_two_series() -> None:
    a = _series([0.5] * 10)
    b = _series([0.3] * 10)

    point, low, high = block_bootstrap_ci(a, b, block=3, seed=0)

    assert point == pytest.approx(0.2, abs=1e-10)
    assert low <= point <= high


def test_block_bootstrap_ci_falls_back_when_fewer_rows_than_block() -> None:
    values = _series([1.0, 2.0, 3.0])

    point, low, high = block_bootstrap_ci(values, block=6, seed=0)

    assert point == low == high == pytest.approx(2.0)


def test_flip_rate_per_year_hand_computed() -> None:
    # 12 months, flips at index 1 (up->down) and index 2 (down->up): 2 flips.
    p = _series([0.9, 0.1, 0.9] + [0.9] * 9)

    assert flip_rate_per_year(p) == pytest.approx(2 * 12.0 / 12)


def test_mean_regime_duration_months_hand_computed() -> None:
    # Runs: [up]*3, [down]*1, [up]*2 -> durations [3, 1, 2], mean = 2.
    p = _series([0.9, 0.9, 0.9, 0.1, 0.9, 0.9])

    assert mean_regime_duration_months(p) == pytest.approx(2.0)


def test_detection_lag_never_detected() -> None:
    usrec = _series([0.0, 1.0, 1.0, 1.0])
    p_growth_up = _series([0.9, 0.9, 0.9, 0.9])

    assert detection_lag(p_growth_up, usrec) == [None]


def test_detection_lag_detected_at_start() -> None:
    usrec = _series([0.0, 1.0, 1.0])
    p_growth_up = _series([0.9, 0.2, 0.2])

    assert detection_lag(p_growth_up, usrec) == [0]


def test_detection_lag_detected_later() -> None:
    usrec = _series([0.0, 1.0, 1.0, 1.0])
    p_growth_up = _series([0.9, 0.9, 0.9, 0.2])

    assert detection_lag(p_growth_up, usrec) == [2]


def test_detection_lag_excludes_a_start_outside_the_scored_window() -> None:
    """A recession start before `p_growth_up`'s own index range must be skipped
    entirely, not reported as `None` ("never detected") — it was never in scope to
    detect in the first place."""
    # usrec has two 0->1 transitions: 2019-02-28 and 2019-04-30.
    usrec = _series([0.0, 1.0, 0.0, 1.0], start="2019-01-31")
    # p_growth_up only covers from 2019-04-30 onward, so the 2019-02-28 start is
    # out of its range.
    p_growth_up = _series([0.9, 0.2], start="2019-04-30")

    assert detection_lag(p_growth_up, usrec) == [1]


def test_flip_rate_per_year_with_fewer_than_two_points_is_zero() -> None:
    assert flip_rate_per_year(_series([0.9])) == 0.0
    assert flip_rate_per_year(pd.Series(dtype=float)) == 0.0


def test_mean_regime_duration_months_of_empty_series_is_nan() -> None:
    assert math.isnan(mean_regime_duration_months(pd.Series(dtype=float)))
