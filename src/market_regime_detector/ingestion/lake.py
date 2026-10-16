import io
from datetime import UTC, datetime
from datetime import date as date_type
from typing import TYPE_CHECKING, cast

import pandas as pd
from botocore.exceptions import ClientError

from market_regime_detector.ingestion.schema import CANONICAL_COLUMNS, validate_schema

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_s3.client import S3Client


def _partition_key(series_id: str, year: int) -> str:
    return f"data/asset={series_id}/year={year}/data.parquet"


def _read_partition(s3: "S3Client", bucket: str, series_id: str, year: int) -> pd.DataFrame | None:
    try:
        obj = s3.get_object(Bucket=bucket, Key=_partition_key(series_id, year))
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def _write_partition(
    s3: "S3Client", bucket: str, series_id: str, year: int, df: pd.DataFrame
) -> None:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=_partition_key(series_id, year), Body=buffer.getvalue())


def write_to_lake(s3: "S3Client", bucket: str, df: pd.DataFrame) -> None:
    """Idempotently merges `df` into the S3 lake, one partition per `(series_id,
    year(date))` at `data/asset=<series_id>/year=<year>/data.parquet`.

    Each partition is compacted on write: any existing partition data is read back,
    concatenated with the new rows, de-duplicated on the natural key `(series_id,
    date, realtime_start)` keeping the newest `ingested_at`, and the whole partition
    is overwritten as a single Parquet object. Re-running the same ingestion range
    any number of times converges to the same row set — no duplicate rows, no
    unbounded growth in small files per partition.
    """
    if df.empty:
        return

    df = validate_schema(df)
    years = df["date"].dt.year

    for (series_id, year), _ in df.groupby([df["series_id"], years]):
        series_id, year = cast(str, series_id), cast(int, year)
        group = df[(df["series_id"] == series_id) & (years == year)]
        existing = _read_partition(s3, bucket, series_id, year)
        merged = pd.concat([existing, group], ignore_index=True) if existing is not None else group
        merged = validate_schema(merged)
        _write_partition(s3, bucket, series_id, year, merged)


def get_last_ingested_date(s3: "S3Client", bucket: str, series_id: str) -> date_type | None:
    """Returns the latest observation date currently in the lake for `series_id`.

    Checks the current and prior calendar year's partitions — cheap, and covers the
    realistic case of a series ingested at least once within the last ~13 months.
    Returns `None` if neither partition exists (first-ever ingestion for this series).
    """
    this_year = datetime.now(UTC).year
    latest: pd.Timestamp | None = None

    for year in (this_year, this_year - 1):
        existing = _read_partition(s3, bucket, series_id, year)
        if existing is not None and not existing.empty:
            candidate = existing["date"].max()
            if latest is None or candidate > latest:
                latest = candidate

    return latest.date() if latest is not None else None


def read_series(s3: "S3Client", bucket: str, series_id: str) -> pd.DataFrame:
    """Reads and concatenates every `(series_id, year)` partition — the training
    dataset builder's way of getting a series' full raw history in one call, versus
    `get_last_ingested_date`'s deliberately narrow current/prior-year check.
    """
    prefix = f"data/asset={series_id}/year="
    frames: list[pd.DataFrame] = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            response = s3.get_object(Bucket=bucket, Key=obj["Key"])
            frames.append(pd.read_parquet(io.BytesIO(response["Body"].read())))

    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "realtime_start"])
        .reset_index(drop=True)
    )
