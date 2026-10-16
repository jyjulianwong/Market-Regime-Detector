from collections.abc import Generator

import boto3
import pytest
from moto import mock_aws
from mypy_boto3_dynamodb.client import DynamoDBClient

from market_regime_detector.common.model_registry import (
    get_model,
    get_production_model,
    list_models_by_stage,
    promote_model,
    put_model_item,
)

TABLE_NAME = "test-registry"


@pytest.fixture
def client() -> Generator[DynamoDBClient, None, None]:
    with mock_aws():
        dynamodb = boto3.client("dynamodb", region_name="eu-west-2")
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[{"AttributeName": "model_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "model_id", "AttributeType": "S"},
                {"AttributeName": "stage", "AttributeType": "S"},
                {"AttributeName": "last_modified", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "stage-index",
                    "KeySchema": [
                        {"AttributeName": "stage", "KeyType": "HASH"},
                        {"AttributeName": "last_modified", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {
                        "ReadCapacityUnits": 5,
                        "WriteCapacityUnits": 5,
                    },
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield dynamodb


def _item(
    model_id: str, stage: str, last_modified: str = "2024-01-01T00:00:00"
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "stage": stage,
        "last_modified": last_modified,
        "dataset_id": "ds-test",
    }


def test_put_model_item_rejects_production_stage(client: DynamoDBClient) -> None:
    with pytest.raises(ValueError, match="development.*staging"):
        put_model_item(client, TABLE_NAME, _item("mrd-1", "production"))


def test_put_model_item_rejects_archived_stage(client: DynamoDBClient) -> None:
    with pytest.raises(ValueError, match="development.*staging"):
        put_model_item(client, TABLE_NAME, _item("mrd-1", "archived"))


def test_put_and_get_model(client: DynamoDBClient) -> None:
    put_model_item(client, TABLE_NAME, _item("mrd-1", "staging"))

    result = get_model(client, TABLE_NAME, "mrd-1")

    assert result is not None
    assert result["model_id"] == "mrd-1"
    assert result["stage"] == "staging"


def test_get_model_returns_none_when_missing(client: DynamoDBClient) -> None:
    assert get_model(client, TABLE_NAME, "does-not-exist") is None


def test_get_production_model_via_gsi(client: DynamoDBClient) -> None:
    put_model_item(client, TABLE_NAME, _item("mrd-1", "staging"))
    promote_model(client, TABLE_NAME, "mrd-1")

    result = get_production_model(client, TABLE_NAME)

    assert result is not None
    assert result["model_id"] == "mrd-1"
    assert result["stage"] == "production"


def test_get_production_model_returns_none_when_no_production(client: DynamoDBClient) -> None:
    put_model_item(client, TABLE_NAME, _item("mrd-1", "staging"))

    assert get_production_model(client, TABLE_NAME) is None


def test_list_models_by_stage(client: DynamoDBClient) -> None:
    put_model_item(client, TABLE_NAME, _item("mrd-1", "staging", "2024-01-01T00:00:00"))
    put_model_item(client, TABLE_NAME, _item("mrd-2", "staging", "2024-01-02T00:00:00"))
    put_model_item(client, TABLE_NAME, _item("mrd-3", "development", "2024-01-03T00:00:00"))

    staging = list_models_by_stage(client, TABLE_NAME, "staging")

    assert {item["model_id"] for item in staging} == {"mrd-1", "mrd-2"}


def test_promote_model_keeps_exactly_one_production_first_time(client: DynamoDBClient) -> None:
    put_model_item(client, TABLE_NAME, _item("mrd-1", "staging"))

    promote_model(client, TABLE_NAME, "mrd-1")

    assert get_model(client, TABLE_NAME, "mrd-1")["stage"] == "production"  # type: ignore[index]
    assert len(list_models_by_stage(client, TABLE_NAME, "production")) == 1


def test_promote_model_archives_previous_production(client: DynamoDBClient) -> None:
    put_model_item(client, TABLE_NAME, _item("mrd-1", "staging"))
    put_model_item(client, TABLE_NAME, _item("mrd-2", "staging"))
    promote_model(client, TABLE_NAME, "mrd-1")

    promote_model(client, TABLE_NAME, "mrd-2")

    assert get_model(client, TABLE_NAME, "mrd-1")["stage"] == "archived"  # type: ignore[index]
    assert get_model(client, TABLE_NAME, "mrd-2")["stage"] == "production"  # type: ignore[index]
    assert len(list_models_by_stage(client, TABLE_NAME, "production")) == 1


def test_promote_model_is_a_noop_when_already_production(client: DynamoDBClient) -> None:
    put_model_item(client, TABLE_NAME, _item("mrd-1", "staging"))
    promote_model(client, TABLE_NAME, "mrd-1")

    promote_model(client, TABLE_NAME, "mrd-1")  # promote again, idempotently

    assert len(list_models_by_stage(client, TABLE_NAME, "production")) == 1


def test_promote_model_raises_for_nonexistent_model(client: DynamoDBClient) -> None:
    with pytest.raises(Exception, match="ConditionalCheckFailed|TransactionCanceled"):
        promote_model(client, TABLE_NAME, "does-not-exist")
