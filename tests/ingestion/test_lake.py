import io
from collections.abc import Generator
from datetime import UTC, date, datetime

import boto3
import pandas as pd
import pytest
from moto import mock_aws
from mypy_boto3_s3.client import S3Client

from market_regime_detector.ingestion.lake import (
    get_last_ingested_date,
    read_series,
    write_to_lake,
)
from market_regime_detector.ingestion.schema import CANONICAL_COLUMNS

BUCKET = "test-bucket"


@pytest.fixture
def s3_client() -> Generator[S3Client, None, None]:
    with mock_aws():
        client = boto3.client("s3", region_name="eu-west-2")
        client.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-2"},
        )
        yield client


def _df(
    dates: list[str],
    values: list[float],
    series_id: str = "TEST",
    realtime_start: list[str] | None = None,
    ingested_at: list[datetime] | None = None,
) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "value": values,
            "realtime_start": realtime_start if realtime_start is not None else dates,
            "source": ["fred"] * n,
            "series_id": [series_id] * n,
            "ingested_at": ingested_at if ingested_at is not None else [datetime.now(UTC)] * n,
        }
    )


def _read_partition(s3_client: S3Client, series_id: str, year: int) -> pd.DataFrame:
    obj = s3_client.get_object(
        Bucket=BUCKET, Key=f"data/asset={series_id}/year={year}/data.parquet"
    )
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def test_write_to_lake_creates_partition(s3_client: S3Client) -> None:
    write_to_lake(s3_client, BUCKET, _df(["2024-01-01", "2024-02-01"], [1.0, 2.0]))

    result = _read_partition(s3_client, "TEST", 2024)
    assert len(result) == 2


def test_write_to_lake_is_idempotent(s3_client: S3Client) -> None:
    df = _df(["2024-01-01", "2024-02-01"], [1.0, 2.0])

    write_to_lake(s3_client, BUCKET, df)
    write_to_lake(s3_client, BUCKET, df)

    result = _read_partition(s3_client, "TEST", 2024)
    assert len(result) == 2


def test_write_to_lake_splits_across_year_partitions(s3_client: S3Client) -> None:
    write_to_lake(s3_client, BUCKET, _df(["2023-12-31", "2024-01-01"], [1.0, 2.0]))

    assert len(_read_partition(s3_client, "TEST", 2023)) == 1
    assert len(_read_partition(s3_client, "TEST", 2024)) == 1


def test_write_to_lake_keeps_newest_ingested_at_for_same_key(s3_client: S3Client) -> None:
    first = _df(["2024-01-01"], [1.0], ingested_at=[datetime(2024, 1, 2, tzinfo=UTC)])
    second = _df(["2024-01-01"], [1.5], ingested_at=[datetime(2024, 1, 3, tzinfo=UTC)])

    write_to_lake(s3_client, BUCKET, first)
    write_to_lake(s3_client, BUCKET, second)

    result = _read_partition(s3_client, "TEST", 2024)
    assert len(result) == 1
    assert result.loc[0, "value"] == 1.5


def test_write_to_lake_handles_ticker_with_equals_sign(s3_client: S3Client) -> None:
    write_to_lake(s3_client, BUCKET, _df(["2024-01-01"], [1.0], series_id="GC=F"))

    result = _read_partition(s3_client, "GC=F", 2024)
    assert len(result) == 1


def test_get_last_ingested_date_returns_none_when_empty(s3_client: S3Client) -> None:
    assert get_last_ingested_date(s3_client, BUCKET, "TEST") is None


def test_get_last_ingested_date_returns_max_date(s3_client: S3Client) -> None:
    # get_last_ingested_date only checks the current and prior calendar year's
    # partitions (see its docstring), so the fixture dates must be "recent" relative
    # to whenever the test actually runs — not a hardcoded past year.
    this_year = datetime.now(UTC).year
    write_to_lake(s3_client, BUCKET, _df([f"{this_year}-01-01", f"{this_year}-02-01"], [1.0, 2.0]))

    assert get_last_ingested_date(s3_client, BUCKET, "TEST") == date(this_year, 2, 1)


def test_read_series_returns_empty_frame_with_canonical_columns_when_nothing_ingested(
    s3_client: S3Client,
) -> None:
    result = read_series(s3_client, BUCKET, "NEVER_INGESTED")

    assert result.empty
    assert list(result.columns) == CANONICAL_COLUMNS


def test_read_series_concatenates_across_year_partitions(s3_client: S3Client) -> None:
    write_to_lake(
        s3_client, BUCKET, _df(["2023-12-31", "2024-01-01", "2025-06-01"], [1.0, 2.0, 3.0])
    )

    result = read_series(s3_client, BUCKET, "TEST")

    assert len(result) == 3
    assert list(result["date"]) == sorted(result["date"])


def test_read_series_only_reads_the_requested_series(s3_client: S3Client) -> None:
    write_to_lake(s3_client, BUCKET, _df(["2024-01-01"], [1.0], series_id="SERIES_A"))
    write_to_lake(s3_client, BUCKET, _df(["2024-01-01"], [2.0], series_id="SERIES_B"))

    result = read_series(s3_client, BUCKET, "SERIES_A")

    assert len(result) == 1
    assert result.loc[0, "series_id"] == "SERIES_A"
