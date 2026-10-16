import logging
from datetime import UTC, date, datetime

from fastapi import APIRouter, HTTPException, Query

from market_regime_detector.common.model_registry import ModelStage
from market_regime_detector.common.settings import get_settings
from market_regime_detector.server.reads import (
    cached_get_model,
    cached_list_models,
    cached_read_json,
    resolve_model,
)
from market_regime_detector.server.schemas import (
    HealthResponse,
    HistoryResponse,
    LatestPredictionResponse,
    ModelSummaryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/models", response_model=list[ModelSummaryResponse])
def list_models(stage: ModelStage | None = Query(default=None)) -> list[ModelSummaryResponse]:
    settings = get_settings()
    items = cached_list_models(settings.dynamodb_table_name, stage)
    return [ModelSummaryResponse.model_validate(item) for item in items]


@router.get("/models/{model_id}", response_model=ModelSummaryResponse)
def get_model_detail(model_id: str) -> ModelSummaryResponse:
    settings = get_settings()
    item = cached_get_model(settings.dynamodb_table_name, model_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"model {model_id!r} not found")
    return ModelSummaryResponse.model_validate(item)


def _resolve_or_404(table_name: str, model_id: str | None) -> dict[str, object]:
    item = resolve_model(table_name, model_id)
    if item is None:
        detail = f"model {model_id!r} not found" if model_id is not None else "no production model"
        raise HTTPException(status_code=404, detail=detail)
    return item


@router.get("/predictions/latest", response_model=LatestPredictionResponse)
def get_latest_prediction(
    model_id: str | None = Query(default=None),
) -> LatestPredictionResponse:
    settings = get_settings()
    registry_item = _resolve_or_404(settings.dynamodb_table_name, model_id)
    resolved_model_id = registry_item["model_id"]
    assert isinstance(resolved_model_id, str)

    data = cached_read_json(
        settings.s3_bucket_name, f"predictions/model_id={resolved_model_id}/latest.json"
    )
    if data is None:
        raise HTTPException(
            status_code=404, detail=f"no predictions exist for model {resolved_model_id!r}"
        )

    response = LatestPredictionResponse.model_validate(data)

    # Request-time freshness (brief section 3.4), OR'd with the batch job's own
    # generation-time is_stale (brief section 3.3) — two different staleness
    # signals, combined here rather than conflated into one at either end.
    age_hours = (datetime.now(UTC) - response.generated_at).total_seconds() / 3600
    if age_hours > settings.staleness_max_age_hours:
        response.is_stale = True
        response.warnings.append(
            f"result is {age_hours:.1f}h old, exceeding the "
            f"{settings.staleness_max_age_hours}h freshness threshold"
        )
    return response


@router.get("/predictions/history", response_model=HistoryResponse)
def get_prediction_history(
    model_id: str | None = Query(default=None),
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
) -> HistoryResponse:
    settings = get_settings()
    registry_item = _resolve_or_404(settings.dynamodb_table_name, model_id)
    resolved_model_id = registry_item["model_id"]
    assert isinstance(resolved_model_id, str)

    data = cached_read_json(
        settings.s3_bucket_name, f"predictions/model_id={resolved_model_id}/history.json"
    )
    if data is None:
        raise HTTPException(
            status_code=404, detail=f"no predictions exist for model {resolved_model_id!r}"
        )

    response = HistoryResponse.model_validate(data)
    if start is not None:
        response.history = [record for record in response.history if record.date >= start]
    if end is not None:
        response.history = [record for record in response.history if record.date <= end]
    return response
