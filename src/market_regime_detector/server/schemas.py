from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from market_regime_detector.common.disclaimer import DISCLAIMER


class HealthResponse(BaseModel):
    status: str = "ok"


class ModelSummaryResponse(BaseModel):
    """Mirrors the registry item's flat fields (brief section 4.3) — never the
    full asset list, which the registry item itself never contains either.
    """

    model_id: str
    model_name: str
    dataset_id: str
    stage: str
    last_modified: str
    date_start: str | None = None
    date_end: str | None = None
    decision_frequency: str | None = None
    feature_config_hash: str | None = None
    n_assets: int | None = None
    git_sha: str | None = None
    vintage_aware: bool | None = None
    model_s3_uri: str | None = None
    manifest_s3_uri: str | None = None
    # Training-defined shapes, not server-defined — kept loose deliberately.
    metrics_summary: dict[str, Any] | None = None
    hmm_config_summary: dict[str, Any] | None = None


class QuadrantProbabilities(BaseModel):
    GROWTH_UP_INFLATION_DOWN: float
    GROWTH_UP_INFLATION_UP: float
    GROWTH_DOWN_INFLATION_UP: float
    GROWTH_DOWN_INFLATION_DOWN: float


class LatestSnapshot(BaseModel):
    p_growth_up: float
    p_inflation_up: float
    quadrants: QuadrantProbabilities
    regime: str
    confidence: float


class LatestPredictionResponse(BaseModel):
    """Brief section 5's shape. `disclaimer` has a default so it's present even
    though this is "every response that contains regime information" (brief
    section 3.4), not literally every endpoint.
    """

    model_id: str
    model_name: str
    dataset_id: str
    asof: date
    generated_at: datetime
    observation_date: date | None
    is_stale: bool
    latest: LatestSnapshot | None
    warnings: list[str]
    disclaimer: str = DISCLAIMER


class HistoryRecord(BaseModel):
    date: date
    p_growth_up: float
    p_inflation_up: float
    quadrants: QuadrantProbabilities
    regime: str
    confidence: float


class HistoryResponse(BaseModel):
    """`prediction/outputs.to_history_json` doesn't itself write a `disclaimer`
    key — the default below supplies it here at serve time instead of editing
    Phase 6's stored JSON shape.
    """

    model_id: str
    model_name: str
    dataset_id: str
    generated_at: datetime
    history: list[HistoryRecord]
    disclaimer: str = DISCLAIMER
