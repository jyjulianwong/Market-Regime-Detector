from datetime import date, timedelta

from market_regime_detector.common.s3 import get_s3_client
from market_regime_detector.ingestion.lake import get_last_ingested_date, write_to_lake
from market_regime_detector.ingestion.providers.factory import build_provider

_FIRST_INGESTION_LOOKBACK_DAYS = 365 * 10


def ingest_series(
    series_id: str,
    source: str,
    start: date,
    end: date,
    bucket: str,
    asof: date | None = None,
) -> None:
    """Fetches `series_id` from `source` over `[start, end]` and writes it to the lake.

    A plain function with no scheduler coupling (brief section 3.1) — called by
    `scripts/ingest_data.py` today, and from Phase 6 the batch Lambda's incremental
    ingestion step too.
    """
    provider = build_provider(source)
    df = provider.fetch(series_id, start, end, asof=asof)
    s3 = get_s3_client()
    write_to_lake(s3, bucket, df)


def ingest_incremental(
    series_id: str,
    source: str,
    bucket: str,
    end: date | None = None,
    overlap_days: int = 30,
) -> None:
    """Ingests from `(last ingested date - overlap_days)` through `end` (default
    today).

    The overlap window re-fetches a trailing slice of already-ingested dates so
    recent revisions are picked up without re-pulling a series' entire history on
    every run. Falls back to a 10-year lookback for a series with no lake history yet.
    """
    effective_end = end if end is not None else date.today()

    s3 = get_s3_client()
    last_date = get_last_ingested_date(s3, bucket, series_id)
    start = (
        last_date - timedelta(days=overlap_days)
        if last_date is not None
        else effective_end - timedelta(days=_FIRST_INGESTION_LOOKBACK_DAYS)
    )

    ingest_series(series_id, source, start, effective_end, bucket)
