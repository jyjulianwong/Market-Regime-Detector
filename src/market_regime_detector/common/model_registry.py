from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_dynamodb.client import DynamoDBClient

ModelStage = Literal["development", "staging", "production", "archived"]

_serializer = TypeSerializer()
_deserializer = TypeDeserializer()


def _serialize(item: dict[str, object]) -> dict[str, object]:
    return {key: _serializer.serialize(value) for key, value in item.items()}


def _deserialize(item: dict[str, Any]) -> dict[str, object]:
    return {key: _deserializer.deserialize(value) for key, value in item.items()}


def put_model_item(client: "DynamoDBClient", table_name: str, item: dict[str, object]) -> None:
    """Writes a new registry item (brief section 4.3). `item["stage"]` must be
    `"development"` or `"staging"` — newly published models never start as
    `"production"`; only `promote_model` sets that.

    Idempotent: re-running with the exact same `item` content overwrites to the same
    state — this is what makes `scripts/publish_model.py` retry-safe, given the
    publisher generates a deterministic `model_id` (see `training/publisher.py`).
    """
    stage = item.get("stage")
    if stage not in ("development", "staging"):
        raise ValueError(
            f"newly published models must be 'development' or 'staging', got {stage!r}"
        )
    client.put_item(TableName=table_name, Item=_serialize(item))  # type: ignore[arg-type]


def get_model(client: "DynamoDBClient", table_name: str, model_id: str) -> dict[str, object] | None:
    response = client.get_item(
        TableName=table_name,
        Key=_serialize({"model_id": model_id}),  # type: ignore[arg-type]
    )
    item = response.get("Item")
    return _deserialize(item) if item else None


def get_production_model(client: "DynamoDBClient", table_name: str) -> dict[str, object] | None:
    """Queries the `stage-index` GSI — brief section 4.3: "latest production model"
    is a single query. At most one item should ever have `stage="production"` (the
    invariant `promote_model` maintains); returns it, or `None`.
    """
    response = client.query(
        TableName=table_name,
        IndexName="stage-index",
        KeyConditionExpression="stage = :stage",
        ExpressionAttributeValues=_serialize({":stage": "production"}),  # type: ignore[arg-type]
        Limit=1,
    )
    items = response.get("Items", [])
    return _deserialize(items[0]) if items else None


def list_models_by_stage(
    client: "DynamoDBClient", table_name: str, stage: ModelStage
) -> list[dict[str, object]]:
    response = client.query(
        TableName=table_name,
        IndexName="stage-index",
        KeyConditionExpression="stage = :stage",
        ExpressionAttributeValues=_serialize({":stage": stage}),  # type: ignore[arg-type]
    )
    return [_deserialize(item) for item in response.get("Items", [])]


def list_all_models(client: "DynamoDBClient", table_name: str) -> list[dict[str, object]]:
    """A plain `Scan` — used by `GET /v1/models` (Phase 7) when no `stage` filter is
    given. `list_models_by_stage`'s GSI only helps once you already know the stage;
    model counts are expected to stay small (a handful per environment), so an
    unindexed Scan is appropriate here — not a pattern to repeat for anything larger.
    """
    response = client.scan(TableName=table_name)
    return [_deserialize(item) for item in response.get("Items", [])]


def promote_model(client: "DynamoDBClient", table_name: str, model_id: str) -> None:
    """Atomically demotes the current `production` model (if any, and if it isn't
    already `model_id`) to `"archived"` and promotes `model_id` to `"production"` —
    brief section 4.3's invariant ("at most one item has stage=production")
    enforced structurally via a single `transact_write_items` call, not by
    convention: there's never a moment with zero or two production models visible
    to a reader.

    Raises (`ClientError`, `ConditionalCheckFailedException` inside
    `TransactionCanceledException`) if `model_id` doesn't exist.
    """
    now = datetime.now(UTC).isoformat()
    current_production = get_production_model(client, table_name)

    transact_items: list[dict[str, object]] = [
        {
            "Update": {
                "TableName": table_name,
                "Key": _serialize({"model_id": model_id}),
                "UpdateExpression": "SET stage = :production, last_modified = :now",
                "ConditionExpression": "attribute_exists(model_id)",
                "ExpressionAttributeValues": _serialize({":production": "production", ":now": now}),
            }
        }
    ]
    if current_production is not None and current_production["model_id"] != model_id:
        transact_items.append(
            {
                "Update": {
                    "TableName": table_name,
                    "Key": _serialize({"model_id": current_production["model_id"]}),
                    "UpdateExpression": "SET stage = :archived, last_modified = :now",
                    "ExpressionAttributeValues": _serialize({":archived": "archived", ":now": now}),
                }
            }
        )

    client.transact_write_items(TransactItems=transact_items)  # type: ignore[arg-type]
