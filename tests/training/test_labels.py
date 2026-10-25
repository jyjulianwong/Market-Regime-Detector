import numpy as np
import pandas as pd

from market_regime_detector.training.labels import (
    build_label_set,
    growth_label,
    inflation_label,
    recession_label,
)


def _monthly_series(start: str, values: list[float]) -> pd.Series:
    """FRED convention: one observation dated the 1st of each reference month."""
    dates = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=dates)


def test_growth_label_above_and_below_trend() -> None:
    cfnai_ma3 = _monthly_series("2020-01-01", [0.5, -0.2, 0.0, 0.3])

    label = growth_label(cfnai_ma3)

    expected_index = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    pd.testing.assert_index_equal(label.index, pd.DatetimeIndex(expected_index))
    # 0.5 > 0 -> up; -0.2 <= 0 -> down; 0.0 is not > 0 -> down; 0.3 > 0 -> up.
    np.testing.assert_array_equal(label.to_numpy(), [1.0, 0.0, 0.0, 1.0])


def test_recession_label_is_the_complement_of_usrec() -> None:
    usrec = _monthly_series("2020-01-01", [0.0, 1.0, 1.0, 0.0])

    label = recession_label(usrec)

    np.testing.assert_array_equal(label.to_numpy(), [1.0, 0.0, 0.0, 1.0])


def test_growth_label_reindexes_month_start_dates_onto_month_ends() -> None:
    """FRED series are dated the 1st; the label index must match the feature
    matrices' real calendar month-end convention (`freq="ME"`), not the raw
    month-start dates coming out of the lake."""
    cfnai_ma3 = _monthly_series("2024-02-01", [0.1])

    label = growth_label(cfnai_ma3)

    assert label.index[0] == pd.Timestamp("2024-02-29")


def test_inflation_label_detects_accelerating_inflation() -> None:
    """16 months of CPI: a flat 1%/month rate for the first 12 (Jan23-Dec23), then
    stepping up to 2%/month for the last 4 (Jan24-Apr24). By the final month
    (Apr24), 3-month-annualized inflation should clearly exceed the trailing
    year-over-year rate measured 3 months earlier — hand-computed below,
    independently of `training/labels.py`'s own formula.
    """
    base = [100.0 * (1.01**i) for i in range(12)]
    accelerated = [base[-1] * (1.02**i) for i in range(1, 5)]
    cpi = _monthly_series("2023-01-01", base + accelerated)

    label = inflation_label(cpi)

    # infl_3m_ann at Apr24 = (cpi[Apr24]/cpi[Jan24])**4 - 1 = (1.02**3)**4 - 1
    expected_3m_ann = (1.02**3) ** 4 - 1
    # infl_yoy_lagged at Apr24 = cpi.pct_change(12) measured at Jan24 = cpi[Jan24]/cpi[Jan23] - 1
    expected_yoy_lagged = (base[-1] * 1.02) / base[0] - 1
    assert expected_3m_ann > expected_yoy_lagged  # sanity check on the hand-computation itself

    assert label.loc[pd.Timestamp("2024-04-30")] == 1.0


def test_inflation_label_is_nan_before_enough_history_exists() -> None:
    """Fewer than 16 months of data means `infl_yoy_lagged` is undefined for every
    month — must come back as `NaN`, never silently `False` (pandas' `>` would
    otherwise treat the comparison as "not accelerating," which misrepresents
    "unknown" as a real answer)."""
    cpi = _monthly_series("2023-01-01", [100.0 * (1.01**i) for i in range(10)])

    label = inflation_label(cpi)

    assert label.isna().all()


def test_build_label_set_stores_lag_without_shifting_the_series() -> None:
    """`label_lag_months` is metadata the scoring harness applies later — building
    a `LabelSet` must not itself shift any series, whatever lag is passed."""
    cfnai_ma3 = _monthly_series("2020-01-01", [0.5, -0.2])
    usrec = _monthly_series("2020-01-01", [0.0, 1.0])
    cpi = _monthly_series("2020-01-01", [100.0, 101.0])

    default_lag = build_label_set(cfnai_ma3, usrec, cpi)
    custom_lag = build_label_set(cfnai_ma3, usrec, cpi, label_lag_months=3)

    assert default_lag.label_lag_months == 1
    assert custom_lag.label_lag_months == 3
    pd.testing.assert_series_equal(default_lag.growth, custom_lag.growth)
    pd.testing.assert_series_equal(default_lag.growth_recession, custom_lag.growth_recession)
    pd.testing.assert_series_equal(default_lag.inflation, custom_lag.inflation)
