import json
from decimal import Decimal
from typing import Any

from botocore.exceptions import ClientError

from market_regime_detector.common.aws_dynamodb import get_dynamodb_client
from market_regime_detector.common.aws_s3 import get_s3_client
from market_regime_detector.common.model_registry import (
    ModelStage,
    get_model,
    get_production_model,
    list_all_models,
    list_models_by_stage,
)
from market_regime_detector.server.cache import CACHE


def _json_safe(value: Any) -> Any:
    """DynamoDB numbers deserialize to `Decimal` (`common/model_registry.py`
    never loses precision on the way in); pydantic's default JSON encoding of a
    bare `Decimal` inside a loosely-typed `dict[str, Any]` field renders it as a
    *string*, not a number, which is surprising over an API. Converts
    recursively so registry items read back as plain JSON numbers.
    """
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {key: _json_safe(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def cached_get_model(table_name: str, model_id: str) -> dict[str, object] | None:
    def factory() -> dict[str, object] | None:
        item = get_model(get_dynamodb_client(), table_name, model_id)
        return _json_safe(item) if item is not None else None

    return CACHE.get_or_set(f"model:{model_id}", factory)


def cached_get_production_model(table_name: str) -> dict[str, object] | None:
    def factory() -> dict[str, object] | None:
        item = get_production_model(get_dynamodb_client(), table_name)
        return _json_safe(item) if item is not None else None

    return CACHE.get_or_set("model:production", factory)


def cached_list_models(table_name: str, stage: ModelStage | None) -> list[dict[str, object]]:
    def factory() -> list[dict[str, object]]:
        client = get_dynamodb_client()
        items = (
            list_models_by_stage(client, table_name, stage)
            if stage is not None
            else list_all_models(client, table_name)
        )
        return [_json_safe(item) for item in items]

    return CACHE.get_or_set(f"models:{stage or 'all'}", factory)


def resolve_model(table_name: str, model_id: str | None) -> dict[str, object] | None:
    """Explicit `model_id` or the current production model — the same resolution
    policy `prediction/predict.py` uses.
    """
    if model_id is not None:
        return cached_get_model(table_name, model_id)
    return cached_get_production_model(table_name)


def cached_read_json(bucket: str, key: str) -> dict[str, object] | None:
    """Reads and parses a JSON object from S3, or `None` if it doesn't exist — the
    caller (a route) is responsible for turning that into a 404; this module never
    raises for "not found."
    """

    def factory() -> dict[str, object] | None:
        s3 = get_s3_client()
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise
        result: dict[str, object] = json.loads(response["Body"].read())
        return result

    return CACHE.get_or_set(f"s3:{key}", factory)
