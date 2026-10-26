import numpy as np
import pytest

from market_regime_detector.common.scaling import apply_scaler, fit_robust_scaler


def _normal_with_outliers(seed: int = 0, n: int = 300, n_outliers: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=n)
    # A handful of ICSA-March-2020-style extreme prints, like real data would have.
    x[:n_outliers] = 1000.0
    return x.reshape(-1, 1)


@pytest.mark.parametrize("method", ["winsorized", "median_mad"])
def test_fit_robust_scaler_outliers_do_not_dominate_scale(method: str) -> None:
    x = _normal_with_outliers()

    center, scale = fit_robust_scaler(x, method=method)

    # A plain StandardScaler on this data would have scale in the tens (the
    # outliers alone contribute ~1000**2 to the variance) — a robust fit should
    # stay close to the non-outlier population's true std of 1.0.
    assert scale[0] < 3.0


def test_fit_robust_scaler_zero_variance_column_guards_against_div_by_zero() -> None:
    x = np.full((50, 1), 3.0)

    center, scale = fit_robust_scaler(x)

    assert scale[0] == 1.0
    z = apply_scaler(x, center, scale)
    assert np.all(np.isfinite(z))


def test_fit_robust_scaler_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        fit_robust_scaler(np.ones((10, 1)), method="bogus")


def test_apply_scaler_clips_symmetrically() -> None:
    x = _normal_with_outliers()
    center, scale = fit_robust_scaler(x)

    z = apply_scaler(x, center, scale, clip=4.0)

    assert np.all(z <= 4.0)
    assert np.all(z >= -4.0)
    # The outliers should be the ones actually hitting the clip boundary.
    assert np.all(z[:3] == 4.0)


def test_apply_scaler_matches_plain_standardization_within_clip_bounds() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=2.0, size=(100, 2))
    center, scale = fit_robust_scaler(x, method="winsorized")

    z = apply_scaler(x, center, scale, clip=100.0)  # effectively unclipped

    assert np.allclose(z, (x - center) / scale)
