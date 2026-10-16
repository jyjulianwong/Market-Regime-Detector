import logging
from datetime import date
from typing import Any

from market_regime_detector.common.dynamodb import get_dynamodb_client
from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.s3 import get_s3_client
from market_regime_detector.common.settings import get_settings
from market_regime_detector.ingestion.ingest import ingest_incremental
from market_regime_detector.ingestion.series_catalog import FIRST_PASS_SERIES
from market_regime_detector.prediction.outputs import write_prediction_outputs
from market_regime_detector.prediction.predict import run_prediction

logger = logging.getLogger(__name__)


def _parse_date(event: dict[str, Any], key: str) -> date | None:
    value = event.get(key)
    return date.fromisoformat(value) if value is not None else None


def _run_incremental_ingestion(bucket: str, end: date | None) -> None:
    """Runs incremental ingestion for every first-pass series before prediction
    (brief section 3.3: "It runs incremental ingestion by calling the public
    ingestion function, then prediction"). Each series' failure is caught and
    logged, not raised — one bad provider call must not take down the whole daily
    run; prediction proceeds with whatever's already in the lake.
    """
    for spec in FIRST_PASS_SERIES:
        try:
            ingest_incremental(spec.series_id, spec.source, bucket, end=end)
        except Exception:
            logger.exception("incremental ingestion failed for %r", spec.series_id)


def handler(event: dict[str, Any], _context: object = None) -> dict[str, object]:
    """Batch Lambda entrypoint (brief section 3.3). An empty event — the scheduler's
    default once EventBridge exists (Phase 8) — means: production model, lake
    source, latest data.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    model_id = event.get("model_id")
    source = event.get("source", "lake")
    start = _parse_date(event, "start")
    end = _parse_date(event, "end")
    asof = _parse_date(event, "asof")

    _run_incremental_ingestion(settings.s3_bucket_name, end)

    s3 = get_s3_client()
    dynamodb = get_dynamodb_client()
    result = run_prediction(
        s3,
        dynamodb,
        settings.s3_bucket_name,
        settings.dynamodb_table_name,
        model_id=model_id,
        source=source,
        start=start,
        end=end,
        asof=asof,
    )
    write_prediction_outputs(s3, settings.s3_bucket_name, result)

    logger.info(
        "prediction written for model %s, observation_date=%s, is_stale=%s",
        result.model_id,
        result.observation_date,
        result.is_stale,
    )
    return {
        "model_id": result.model_id,
        "observation_date": result.observation_date.isoformat()
        if result.observation_date is not None
        else None,
        "is_stale": result.is_stale,
    }
