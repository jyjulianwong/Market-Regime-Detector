from datetime import UTC, datetime

import pandas as pd
import pytest

from market_regime_detector.common.feature_config import Axis, FeatureDefinition, Frequency
from market_regime_detector.common.features import (
    _resample_to_monthly,
    as_of_lookup,
    build_feature_value,
    build_ratio_feature_value,
)


def _canonical_df(dates: list[str], values: list[float], source: str = "fred") -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "value": values,
            "realtime_start": pd.to_datetime(dates),
            "source": [source] * n,
            "series_id": ["TEST"] * n,
            "ingested_at": [datetime.now(UTC)] * n,
        }
    )


def _daily_feature(**overrides: object) -> FeatureDefinition:
    base: dict[str, object] = {
        "name": "test_daily",
        "axis": Axis.GROWTH,
        "series_id": "TEST",
        "source": "fred",
        "native_frequency": Frequency.DAILY,
        "vintage_aware": False,
        "publication_lag_days": 0,
        "transform": "diff",
        "transform_params": {"n_periods": 1},
        "sign": 1,
    }
    base.update(overrides)
    return FeatureDefinition(**base)  # type: ignore[arg-type]


def _monthly_feature(**overrides: object) -> FeatureDefinition:
    base: dict[str, object] = {
        "name": "test_monthly",
        "axis": Axis.GROWTH,
        "series_id": "TEST",
        "source": "alfred",
        "native_frequency": Frequency.MONTHLY,
        "vintage_aware": True,
        "publication_lag_days": 30,
        "transform": "pct_change",
        "transform_params": {"n_periods": 1},
        "sign": 1,
    }
    base.update(overrides)
    return FeatureDefinition(**base)  # type: ignore[arg-type]


def test_build_feature_value_returns_none_when_nothing_knowable() -> None:
    df = _canonical_df(["2024-06-01"], [1.0])
    feature = _daily_feature()

    result = build_feature_value(df, feature, pd.Timestamp("2024-01-01"))

    assert result is None


def test_build_feature_value_monthly_at_exact_month_end() -> None:
    df = _canonical_df(
        ["2024-01-01", "2024-02-01", "2024-03-01"],
        [100.0, 110.0, 121.0],
        source="alfred",
    )
    # vintage-aware: realtime_start == date here, so every row is immediately known.
    feature = _monthly_feature()

    result = build_feature_value(df, feature, pd.Timestamp("2024-03-01"))

    assert result == pytest.approx(0.1)  # 121/110 - 1


def test_build_feature_value_daily_mid_month_uses_partial_month() -> None:
    """The training/inference-unifying behavior: a mid-month evaluation_date collapses
    to a partial "current month" bucket automatically, with no special-casing.
    """
    df = _canonical_df(
        ["2024-01-31", "2024-02-10", "2024-02-20", "2024-03-05"],
        [1.0, 2.0, 2.5, 100.0],  # the 2024-03-05 point must not leak into Feb's bucket
    )
    feature = _daily_feature()

    result = build_feature_value(df, feature, pd.Timestamp("2024-02-20"))

    # Feb bucket (as of 2024-02-20) = last value <= that date = 2.5; Jan bucket = 1.0.
    assert result == pytest.approx(1.5)


def test_build_feature_value_exact_month_end_matches_full_month() -> None:
    df = _canonical_df(
        ["2024-01-31", "2024-02-10", "2024-02-29"],
        [1.0, 2.0, 2.5],
    )
    feature = _daily_feature()

    result = build_feature_value(df, feature, pd.Timestamp("2024-02-29"))

    assert result == pytest.approx(1.5)  # 2.5 - 1.0


def test_build_feature_value_rejects_ratio_feature() -> None:
    feature = _daily_feature(numerator_series_id="HG=F", denominator_series_id="GC=F")

    with pytest.raises(ValueError, match="ratio feature"):
        build_feature_value(
            _canonical_df(["2024-01-01"], [1.0]), feature, pd.Timestamp("2024-01-01")
        )


def test_build_ratio_feature_value() -> None:
    numerator_df = _canonical_df(["2024-01-31", "2024-02-29", "2024-03-31"], [10.0, 11.0, 12.0])
    denominator_df = _canonical_df(
        ["2024-01-31", "2024-02-29", "2024-03-31"], [100.0, 100.0, 100.0]
    )
    feature = _daily_feature(
        name="ratio_feature",
        transform="pct_change",
        transform_params={"n_periods": 1},
        numerator_series_id="TEST_NUM",
        denominator_series_id="TEST_DEN",
    )

    result = build_ratio_feature_value(
        numerator_df, denominator_df, feature, pd.Timestamp("2024-03-31")
    )

    # ratio: 0.10, 0.11, 0.12 -> pct_change(1) at the last point = 0.12/0.11 - 1
    assert result == pytest.approx(0.12 / 0.11 - 1)


def test_resample_to_monthly_mean_aggregation() -> None:
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2024-01-05", "2024-01-15", "2024-01-25"]))

    result = _resample_to_monthly(s, "mean")

    assert result.iloc[0] == pytest.approx(2.0)


def test_resample_to_monthly_rejects_unknown_aggregation() -> None:
    s = pd.Series([1.0], index=pd.to_datetime(["2024-01-05"]))

    with pytest.raises(ValueError, match="unknown aggregation"):
        _resample_to_monthly(s, "not_a_real_aggregation")


def test_as_of_lookup_returns_none_when_nothing_eligible() -> None:
    s = pd.Series([1.0], index=pd.to_datetime(["2024-06-01"]))

    assert as_of_lookup(s, pd.Timestamp("2024-01-01")) is None


def test_build_ratio_feature_value_returns_none_when_a_leg_is_empty() -> None:
    feature = _daily_feature(
        name="ratio_feature",
        numerator_series_id="TEST_NUM",
        denominator_series_id="TEST_DEN",
    )
    numerator_df = _canonical_df(["2024-01-31"], [10.0])
    empty_denominator_df = _canonical_df(["2099-01-01"], [100.0])  # not yet knowable

    result = build_ratio_feature_value(
        numerator_df, empty_denominator_df, feature, pd.Timestamp("2024-01-31")
    )

    assert result is None


def test_build_ratio_feature_value_monthly_native_uses_as_of_lookup() -> None:
    feature = _monthly_feature(
        name="ratio_feature",
        transform="pct_change",
        transform_params={"n_periods": 1},
        numerator_series_id="TEST_NUM",
        denominator_series_id="TEST_DEN",
    )
    numerator_df = _canonical_df(["2024-01-01", "2024-02-01"], [10.0, 11.0], source="alfred")
    denominator_df = _canonical_df(["2024-01-01", "2024-02-01"], [100.0, 100.0], source="alfred")

    result = build_ratio_feature_value(
        numerator_df, denominator_df, feature, pd.Timestamp("2024-02-01")
    )

    assert result == pytest.approx(0.11 / 0.10 - 1)


def test_build_ratio_feature_value_rejects_non_ratio_feature() -> None:
    feature = _daily_feature()

    with pytest.raises(ValueError, match="not a ratio feature"):
        build_ratio_feature_value(
            _canonical_df(["2024-01-01"], [1.0]),
            _canonical_df(["2024-01-01"], [1.0]),
            feature,
            pd.Timestamp("2024-01-01"),
        )
