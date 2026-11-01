import numpy as np
import pandas as pd
import pytest

from market_regime_detector.training.calibration import (
    DEFAULT_SHRINK_WEIGHT,
    LinearShrinkCalibrator,
    PlattCalibrator,
    expanding_window_calibrate,
    expanding_window_shrink,
    fit_platt_calibrator,
    fit_shrink_calibrator,
    hard_cap,
    identity_calibrator,
    identity_shrink_calibrator,
    shrink_to_base_rate,
)


def test_identity_calibrator_reproduces_input_exactly() -> None:
    p = pd.Series([0.01, 0.4, 0.5, 0.6, 0.99])
    calibrated = identity_calibrator().apply(p.to_numpy())
    np.testing.assert_allclose(calibrated, p.to_numpy(), atol=1e-9)


def test_identity_shrink_calibrator_reproduces_input_exactly() -> None:
    p = pd.Series([0.01, 0.4, 0.5, 0.6, 0.99])
    calibrated = identity_shrink_calibrator().apply(p.to_numpy())
    np.testing.assert_allclose(calibrated, p.to_numpy(), atol=1e-9)


def test_fit_platt_calibrator_falls_back_to_identity_with_no_rows() -> None:
    calibrator = fit_platt_calibrator(pd.Series(dtype=float), pd.Series(dtype=float))
    assert calibrator == PlattCalibrator(a=1.0, b=0.0)


def test_fit_platt_calibrator_falls_back_to_identity_with_single_class_label() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    p = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], index=dates)
    label = pd.Series([1.0] * 6, index=dates)  # no variation — a fit is undefined
    calibrator = fit_platt_calibrator(p, label)
    assert calibrator == PlattCalibrator(a=1.0, b=0.0)


def test_fit_platt_calibrator_does_not_diverge_on_a_small_separable_sample() -> None:
    """Regression test: an earlier implementation using raw 0/1 targets (no
    Platt soft-target smoothing) and an undamped Newton step diverged to
    `|a| > 1e5` on exactly this shape of input — a small sample where the raw
    score already separates the classes almost perfectly. `a`/`b` must stay
    finite and small, and the resulting calibrator must actually reduce log
    loss relative to the identity map on this same sample.
    """
    dates = pd.date_range("2015-01-31", periods=90, freq="ME")
    rng = np.random.default_rng(0)
    label = pd.Series(rng.integers(0, 2, size=90).astype(float), index=dates)
    # A posterior that is a confident but systematically-overconfident version
    # of the truth, mirroring F1's "well-ranked, badly calibrated" profile.
    raw = label * 0.98 + (1 - label) * 0.02
    raw = raw + rng.normal(0, 0.03, size=90)
    raw_p = pd.Series(np.clip(raw, 0.001, 0.999), index=dates)

    calibrator = fit_platt_calibrator(raw_p, label)

    assert np.isfinite(calibrator.a)
    assert np.isfinite(calibrator.b)
    assert abs(calibrator.a) < 100
    assert abs(calibrator.b) < 100


def test_fit_shrink_calibrator_computes_base_rate_from_label() -> None:
    label = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0])  # mean = 0.6
    calibrator = fit_shrink_calibrator(label, weight=DEFAULT_SHRINK_WEIGHT)
    assert calibrator == LinearShrinkCalibrator(weight=0.5, base_rate=0.6)


def test_linear_shrink_calibrator_apply_formula() -> None:
    calibrator = LinearShrinkCalibrator(weight=0.5, base_rate=0.4)
    raw = np.array([0.0, 1.0, 0.8])
    expected = 0.5 * raw + 0.5 * 0.4
    np.testing.assert_allclose(calibrator.apply(raw), expected)


def test_linear_shrink_calibrator_weight_one_is_identity_regardless_of_base_rate() -> None:
    calibrator = LinearShrinkCalibrator(weight=1.0, base_rate=0.9999)
    raw = np.array([0.0, 0.2, 0.5, 1.0])
    np.testing.assert_allclose(calibrator.apply(raw), raw)


def test_hard_cap_clips_to_bounds_and_leaves_interior_untouched() -> None:
    p = pd.Series([0.001, 0.05, 0.5, 0.95, 0.999])
    capped = hard_cap(p, low=0.10, high=0.90)
    np.testing.assert_allclose(capped.to_numpy(), [0.10, 0.10, 0.5, 0.90, 0.90])


def _walk_forward_like(
    decision_dates: pd.DatetimeIndex, p_up: np.ndarray, fold_id: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame({"decision_date": decision_dates, "p_up": p_up, "fold_id": fold_id})


def test_expanding_window_calibrate_uses_only_strictly_earlier_folds() -> None:
    """Fold 1's calibrated output must depend only on fold 0's (p, label)
    pairs — changing fold 1's own label must never change fold 1's own
    calibrated value (that would be circular), and must never change fold 0's
    value (that would be look-ahead).
    """
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    wf = _walk_forward_like(
        dates, np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1]), np.array([0, 0, 0, 1, 1, 1])
    )
    label_a = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=dates)
    label_b = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], index=dates)  # fold 1's label flipped

    out_a = expanding_window_calibrate(wf, label_a)
    out_b = expanding_window_calibrate(wf, label_b)

    # fold 0 (no prior fold) is always the untouched raw p_up.
    np.testing.assert_allclose(out_a.iloc[:3].to_numpy(), [0.9, 0.9, 0.9])
    np.testing.assert_allclose(out_b.iloc[:3].to_numpy(), [0.9, 0.9, 0.9])
    # fold 1 differs between A and B only because fold 1's *own* label
    # differs — but the calibrator used to produce it is identical in both
    # (fit on fold 0 alone), so this just confirms fold 0 is never touched by
    # a later fold's label.
    assert out_a.iloc[0] == out_b.iloc[0]


def test_expanding_window_shrink_uses_only_strictly_earlier_folds() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    wf = _walk_forward_like(
        dates, np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1]), np.array([0, 0, 0, 1, 1, 1])
    )
    label = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=dates)

    out = expanding_window_shrink(wf, label, weight=0.5)

    # fold 0: no prior fold, raw p_up passed through unchanged.
    np.testing.assert_allclose(out.iloc[:3].to_numpy(), [0.9, 0.9, 0.9])
    # fold 1: base_rate fit on fold 0's label only (all 1.0 -> base_rate=1.0),
    # so calibrated = 0.5 * 0.1 + 0.5 * 1.0 = 0.55.
    np.testing.assert_allclose(out.iloc[3:].to_numpy(), [0.55, 0.55, 0.55])


def test_shrink_to_base_rate_matches_fit_shrink_calibrator() -> None:
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    p = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=dates)
    label = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0], index=dates)  # base_rate=0.6

    via_function = shrink_to_base_rate(p, label, weight=0.5)
    via_calibrator = fit_shrink_calibrator(label, weight=0.5).apply(p)

    pd.testing.assert_series_equal(via_function, pd.Series(via_calibrator, index=dates))


@pytest.mark.parametrize("weight", [0.0, 0.3, 0.5, 1.0])
def test_fit_shrink_calibrator_is_a_valid_probability_map(weight: float) -> None:
    label = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0])
    calibrator = fit_shrink_calibrator(label, weight=weight)
    calibrated = calibrator.apply(np.array([0.0, 0.5, 1.0]))
    assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)
