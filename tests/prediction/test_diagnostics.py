from datetime import UTC, datetime

import pandas as pd
import pytest

from market_regime_detector.common.feature_config import (
    Axis,
    FeatureConfig,
    FeatureDefinition,
    Frequency,
)
from market_regime_detector.ingestion.series_catalog import FIRST_PASS_SERIES
from market_regime_detector.prediction.diagnostics import build_data_points, build_feature_points


def _raw_df(
    series_id: str, dates: list[str], values: list[float], source: str = "fred"
) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "value": values,
            "realtime_start": pd.to_datetime(dates),
            "source": [source] * n,
            "series_id": [series_id] * n,
            "ingested_at": [datetime.now(UTC)] * n,
        }
    )


def test_build_data_points_covers_every_first_pass_series() -> None:
    raw = {"INDPRO": _raw_df("INDPRO", ["2024-01-01"], [100.0], source="alfred")}

    points = build_data_points(raw, pd.Timestamp("2024-02-01"))

    assert len(points) == len(FIRST_PASS_SERIES)
    by_name = {p.name: p for p in points}

    indpro = by_name["INDPRO"]
    assert indpro.value == 100.0
    assert indpro.date is not None and indpro.date.isoformat() == "2024-01-01"
    assert indpro.lag_days == 31
    assert indpro.description  # every catalog entry has a human description

    # A series never ingested in this run comes back as a fully-null point, not
    # a KeyError or a dropped row — the UI still needs to list it as "no data."
    payems = by_name["PAYEMS"]
    assert payems.value is None
    assert payems.date is None
    assert payems.realtime_start is None
    assert payems.lag_days is None


def test_build_feature_points_reports_transformed_value_and_backing_lag() -> None:
    feature = FeatureDefinition(
        name="test_feature",
        axis=Axis.GROWTH,
        series_id="TESTSERIES",
        source="fred",
        native_frequency=Frequency.DAILY,
        vintage_aware=False,
        publication_lag_days=0,
        transform="identity",
        transform_params={},
        sign=1,
        description="a test feature",
    )
    config = FeatureConfig(version="test", features=(feature,))
    raw = {"TESTSERIES": _raw_df("TESTSERIES", ["2024-01-01", "2024-01-02"], [1.0, 2.0])}

    points = build_feature_points(
        config, raw, pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-10")
    )

    assert len(points) == 1
    point = points[0]
    assert point.name == "test_feature"
    assert point.description == "a test feature"
    assert point.value == pytest.approx(2.0)
    assert point.date is not None and point.date.isoformat() == "2024-01-02"
    assert point.lag_days == 8  # asof (01-10) minus the backing realtime_start (01-02)


def test_build_feature_points_ratio_feature_uses_the_more_stale_leg() -> None:
    """The bottleneck leg (B, last knowable 01-03) determines the reported date,
    not A (last knowable 01-05) — whichever leg is older governs how current the
    ratio can actually be.
    """
    feature = FeatureDefinition(
        name="ratio_feature",
        axis=Axis.GROWTH,
        series_id="A/B",
        source="yfinance",
        native_frequency=Frequency.DAILY,
        vintage_aware=False,
        publication_lag_days=0,
        transform="identity",
        transform_params={},
        sign=1,
        numerator_series_id="A",
        denominator_series_id="B",
    )
    config = FeatureConfig(version="test", features=(feature,))
    raw = {
        "A": _raw_df("A", ["2024-01-01", "2024-01-05"], [10.0, 20.0]),
        "B": _raw_df("B", ["2024-01-01", "2024-01-03"], [5.0, 5.0]),
    }

    points = build_feature_points(
        config, raw, pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-10")
    )

    assert len(points) == 1
    assert points[0].date is not None and points[0].date.isoformat() == "2024-01-03"


def test_build_feature_points_missing_series_returns_fully_null_point() -> None:
    feature = FeatureDefinition(
        name="missing_feature",
        axis=Axis.GROWTH,
        series_id="NEVER_INGESTED",
        source="fred",
        native_frequency=Frequency.DAILY,
        vintage_aware=False,
        publication_lag_days=0,
        transform="identity",
        transform_params={},
        sign=1,
    )
    config = FeatureConfig(version="test", features=(feature,))

    points = build_feature_points(
        config, {}, pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-10")
    )

    assert points[0].value is None
    assert points[0].date is None
    assert points[0].realtime_start is None
    assert points[0].lag_days is None
