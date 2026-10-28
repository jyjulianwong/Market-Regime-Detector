import logging
from datetime import date
from typing import Any

from market_regime_detector.common.aws_dynamodb import get_dynamodb_client
from market_regime_detector.common.aws_s3 import get_s3_client
from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.settings import Settings, get_settings
from market_regime_detector.ingestion.ingest import ingest_incremental
from market_regime_detector.ingestion.series_catalog import FIRST_PASS_SERIES
from market_regime_detector.prediction.outputs import write_prediction_outputs
from market_regime_detector.prediction.predict import run_prediction

logger = logging.getLogger(__name__)


def _parse_date(event: dict[str, Any], key: str) -> date | None:
    value = event.get(key)
    return date.fromisoformat(value) if value is not None else None


def _handle_ingest(event: dict[str, Any], settings: Settings) -> dict[str, object]:
    """Daily-scheduled action: incremental ingestion for every first-pass series.
    Each series' failure is caught and logged, not raised — one bad provider call
    must not take down the whole run.
    """
    end = _parse_date(event, "end")
    failed: list[str] = []
    for spec in FIRST_PASS_SERIES:
        try:
            ingest_incremental(spec.series_id, spec.source, settings.s3_bucket_name, end=end)
        except Exception:
            logger.exception("incremental ingestion failed for %r", spec.series_id)
            failed.append(spec.series_id)

    result = {
        "series_ingested": len(FIRST_PASS_SERIES) - len(failed),
        "series_failed": failed,
    }
    logger.info("incremental ingestion complete: %s", result)
    return result


def _handle_predict(event: dict[str, Any], settings: Settings) -> dict[str, object]:
    """Daily-scheduled action: nowcast off whatever's already in the lake — never
    triggers ingestion itself (see the `ingest` action above for that).
    """
    model_id = event.get("model_id")
    source = event.get("source", "lake")
    start = _parse_date(event, "start")
    end = _parse_date(event, "end")
    asof = _parse_date(event, "asof")

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


_ACTIONS = {
    "ingest": _handle_ingest,
    "predict": _handle_predict,
}


def handler(event: dict[str, Any], _context: object = None) -> dict[str, object]:
    """Batch Lambda entrypoint, shared by two separately-scheduled EventBridge
    Rules (see terraform/main.tf): a daily `{"action": "ingest"}` run and a daily
    `{"action": "predict"}` run that nowcasts off whatever ingestion already
    landed in the lake. Dispatches on an explicit `action` key rather than always
    doing both, as a single combined run (brief section 3.3's original design)
    did — ingestion and prediction are now independent concerns on independent
    schedules. An empty event (e.g. a manual console test invoke) defaults to
    `"predict"`: production model, lake source, latest data — but never silently
    ingests; that must be requested explicitly.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    action = event.get("action", "predict")
    try:
        action_handler = _ACTIONS[action]
    except KeyError:
        raise ValueError(f"unknown action {action!r}, expected one of {sorted(_ACTIONS)}") from None

    return action_handler(event, settings)
