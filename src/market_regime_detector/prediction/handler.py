import logging
from datetime import date
from typing import TYPE_CHECKING, Any

from market_regime_detector.common.aws_dynamodb import get_dynamodb_client
from market_regime_detector.common.aws_s3 import get_s3_client
from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.model_registry import list_models_by_stage
from market_regime_detector.common.settings import Settings, get_settings
from market_regime_detector.ingestion.ingest import ingest_incremental
from market_regime_detector.ingestion.series_catalog import FIRST_PASS_SERIES
from market_regime_detector.prediction.outputs import write_prediction_outputs
from market_regime_detector.prediction.predict import run_prediction

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_dynamodb.client import DynamoDBClient
    from mypy_boto3_s3.client import S3Client

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


def _run_and_write_one(
    s3: "S3Client",
    dynamodb: "DynamoDBClient",
    settings: Settings,
    model_id: str | None,
    model_name: str | None,
    source: str,
    start: date | None,
    end: date | None,
    asof: date | None,
) -> dict[str, object]:
    result = run_prediction(
        s3,
        dynamodb,
        settings.s3_bucket_name,
        settings.dynamodb_table_name,
        model_id=model_id,
        model_name=model_name,
        source=source,
        start=start,
        end=end,
        asof=asof,
    )
    write_prediction_outputs(s3, settings.s3_bucket_name, result)

    logger.info(
        "prediction written for model_name=%s model_id=%s, observation_date=%s, is_stale=%s",
        result.model_name,
        result.model_id,
        result.observation_date,
        result.is_stale,
    )
    return {
        "model_id": result.model_id,
        "model_name": result.model_name,
        "observation_date": result.observation_date.isoformat()
        if result.observation_date is not None
        else None,
        "is_stale": result.is_stale,
    }


def _handle_predict(event: dict[str, Any], settings: Settings) -> dict[str, object]:
    """Daily-scheduled action: nowcast off whatever's already in the lake — never
    triggers ingestion itself (see the `ingest` action above for that).

    Multi-model support: an explicit `model_id` or `model_name` in the event (e.g.
    `scripts/promote_model.py`'s best-effort async invoke, which sends one
    `model_id`) predicts for exactly that one model. The default EventBridge
    schedule sends neither, so this fans out — one `run_prediction` +
    `write_prediction_outputs` call per `model_name` currently in production,
    found via the existing `stage-index` GSI (`list_models_by_stage`). Each
    family's failure is caught and logged individually, same resilience pattern as
    `_handle_ingest`'s per-series loop, so one broken model never blocks another's
    daily output.
    """
    model_id = event.get("model_id")
    model_name = event.get("model_name")
    source = event.get("source", "lake")
    start = _parse_date(event, "start")
    end = _parse_date(event, "end")
    asof = _parse_date(event, "asof")

    s3 = get_s3_client()
    dynamodb = get_dynamodb_client()

    if model_id is not None or model_name is not None:
        targets: list[tuple[str | None, str | None]] = [(model_id, model_name)]
    else:
        production_items = list_models_by_stage(
            dynamodb, settings.dynamodb_table_name, "production"
        )
        targets = []
        for item in production_items:
            production_model_name = item["model_name"]
            assert isinstance(production_model_name, str)
            targets.append((None, production_model_name))

    results: list[dict[str, object]] = []
    failed: list[str | None] = []
    for target_model_id, target_model_name in targets:
        try:
            results.append(
                _run_and_write_one(
                    s3,
                    dynamodb,
                    settings,
                    target_model_id,
                    target_model_name,
                    source,
                    start,
                    end,
                    asof,
                )
            )
        except Exception:
            logger.exception(
                "prediction failed for model_name=%r model_id=%r",
                target_model_name,
                target_model_id,
            )
            failed.append(target_model_name or target_model_id)

    return {"results": results, "failed": failed}


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
