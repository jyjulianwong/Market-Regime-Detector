from collections.abc import Generator
from datetime import UTC, datetime

import boto3
import pandas as pd
import pytest
from moto import mock_aws
from mypy_boto3_s3.client import S3Client

from market_regime_detector.common.feature_config import (
    Axis,
    FeatureConfig,
    FeatureDefinition,
    Frequency,
)
from market_regime_detector.ingestion.lake import write_to_lake
from market_regime_detector.training.dataset import build_dataset

BUCKET = "test-bucket"


@pytest.fixture
def s3_client() -> Generator[S3Client, None, None]:
    with mock_aws():
        client = boto3.client("s3", region_name="eu-west-2")
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-2"}
        )
        yield client


def _canonical_df(
    dates: list[str], values: list[float], series_id: str, source: str
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


def _test_feature_config() -> FeatureConfig:
    growth_feature = FeatureDefinition(
        name="growth_test",
        axis=Axis.GROWTH,
        series_id="GROWTH_SERIES",
        source="alfred",
        native_frequency=Frequency.MONTHLY,
        vintage_aware=True,
        publication_lag_days=30,
        transform="diff",
        transform_params={"n_periods": 1},
        sign=1,
    )
    inflation_feature = FeatureDefinition(
        name="inflation_test",
        axis=Axis.INFLATION,
        series_id="INFLATION_SERIES",
        source="fred",
        native_frequency=Frequency.DAILY,
        vintage_aware=False,
        publication_lag_days=0,
        transform="diff",
        transform_params={"n_periods": 1},
        sign=1,
    )
    return FeatureConfig(version="test", features=(growth_feature, inflation_feature))


def test_build_dataset_shapes_and_manifest(s3_client: S3Client) -> None:
    write_to_lake(
        s3_client,
        BUCKET,
        _canonical_df(
            ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"],
            [100.0, 101.0, 103.0, 106.0],
            "GROWTH_SERIES",
            source="alfred",
        ),
    )
    write_to_lake(
        s3_client,
        BUCKET,
        _canonical_df(
            ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"],
            [1.0, 1.2, 1.1, 1.3],
            "INFLATION_SERIES",
            source="fred",
        ),
    )

    config = _test_feature_config()
    result = build_dataset(
        s3_client, BUCKET, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-04-30"), config
    )

    assert list(result.growth_df.columns) == ["growth_test"]
    assert list(result.inflation_df.columns) == ["inflation_test"]
    assert len(result.growth_df) >= 1
    # GROWTH_SERIES is alfred-sourced (vintage_aware=True) -> manifest-level flag is
    # True even though INFLATION_SERIES (fred) isn't.
    assert result.manifest["vintage_aware"] is True
    assert result.manifest["earliest_realtime_start"] == "2020-01-01"
    assert result.manifest["decision_frequency"] == "monthly"
    assert result.manifest["feature_config_hash"] == config.content_hash()
    assets = result.manifest["assets"]
    assert isinstance(assets, list)
    assert len(assets) == 2
    dataset_id = result.manifest["dataset_id"]
    assert isinstance(dataset_id, str)
    assert dataset_id.startswith("ds-")


def test_build_dataset_drops_rows_with_any_none_feature(s3_client: S3Client) -> None:
    # Only one data point for growth -> diff(1) is always NaN -> growth_df is empty.
    write_to_lake(
        s3_client, BUCKET, _canonical_df(["2020-01-01"], [100.0], "GROWTH_SERIES", "alfred")
    )
    write_to_lake(
        s3_client, BUCKET, _canonical_df(["2020-01-31"], [1.0], "INFLATION_SERIES", "fred")
    )

    config = _test_feature_config()
    result = build_dataset(
        s3_client, BUCKET, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31"), config
    )

    assert result.growth_df.empty


def test_dataset_id_is_stable_for_identical_inputs(s3_client: S3Client) -> None:
    write_to_lake(
        s3_client,
        BUCKET,
        _canonical_df(["2020-01-01", "2020-02-01"], [100.0, 101.0], "GROWTH_SERIES", "alfred"),
    )
    write_to_lake(
        s3_client,
        BUCKET,
        _canonical_df(["2020-01-31", "2020-02-29"], [1.0, 1.2], "INFLATION_SERIES", "fred"),
    )

    config = _test_feature_config()
    result_a = build_dataset(
        s3_client, BUCKET, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29"), config
    )
    result_b = build_dataset(
        s3_client, BUCKET, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29"), config
    )

    assert result_a.manifest["dataset_id"] == result_b.manifest["dataset_id"]


def test_warns_when_a_series_has_no_lake_data(s3_client: S3Client) -> None:
    write_to_lake(
        s3_client, BUCKET, _canonical_df(["2020-01-01"], [100.0], "GROWTH_SERIES", "alfred")
    )
    # INFLATION_SERIES is never ingested.

    config = _test_feature_config()
    result = build_dataset(
        s3_client, BUCKET, pd.Timestamp("2020-01-31"), pd.Timestamp("2020-01-31"), config
    )

    warnings = result.manifest["warnings"]
    assert isinstance(warnings, list)
    assert any("INFLATION_SERIES" in w for w in warnings)
