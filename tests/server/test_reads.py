from collections.abc import Generator

import boto3
import pytest
from moto import mock_aws
from mypy_boto3_dynamodb.client import DynamoDBClient
from mypy_boto3_s3.client import S3Client

from market_regime_detector.common.model_registry import promote_model, put_model_item
from market_regime_detector.server.cache import CACHE
from market_regime_detector.server.reads import (
    cached_get_model,
    cached_get_production_model,
    cached_list_models,
    cached_read_json,
    resolve_model,
)

BUCKET = "test-bucket"
TABLE_NAME = "test-registry"


@pytest.fixture(autouse=True)
def _clear_cache() -> Generator[None, None, None]:
    CACHE.clear()
    yield
    CACHE.clear()


@pytest.fixture
def aws() -> Generator[tuple[S3Client, DynamoDBClient], None, None]:
    with mock_aws():
        s3 = boto3.client("s3", region_name="eu-west-2")
        s3.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-2"}
        )
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
                    "ProvisionedThroughput": {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield s3, dynamodb


def _item(model_id: str, stage: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "dataset_id": "ds-test",
        "stage": stage,
        "last_modified": "2024-01-01T00:00:00",
    }


def test_cached_get_model_returns_item(aws: tuple[S3Client, DynamoDBClient]) -> None:
    _s3, dynamodb = aws
    put_model_item(dynamodb, TABLE_NAME, _item("mrd-1", "staging"))

    result = cached_get_model(TABLE_NAME, "mrd-1")

    assert result is not None
    assert result["model_id"] == "mrd-1"


def test_cached_get_model_returns_none_when_missing(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    assert cached_get_model(TABLE_NAME, "does-not-exist") is None


def test_cached_get_model_is_cached_across_calls(aws: tuple[S3Client, DynamoDBClient]) -> None:
    _s3, dynamodb = aws
    put_model_item(dynamodb, TABLE_NAME, _item("mrd-1", "staging"))

    first = cached_get_model(TABLE_NAME, "mrd-1")
    # Delete the item directly, bypassing the registry, to prove a second call
    # within the TTL window doesn't hit DynamoDB again.
    dynamodb.delete_item(TableName=TABLE_NAME, Key={"model_id": {"S": "mrd-1"}})
    second = cached_get_model(TABLE_NAME, "mrd-1")

    assert first == second


def test_cached_get_production_model(aws: tuple[S3Client, DynamoDBClient]) -> None:
    _s3, dynamodb = aws
    put_model_item(dynamodb, TABLE_NAME, _item("mrd-1", "staging"))
    promote_model(dynamodb, TABLE_NAME, "mrd-1")

    result = cached_get_production_model(TABLE_NAME)

    assert result is not None
    assert result["model_id"] == "mrd-1"


def test_resolve_model_prefers_explicit_model_id(aws: tuple[S3Client, DynamoDBClient]) -> None:
    _s3, dynamodb = aws
    put_model_item(dynamodb, TABLE_NAME, _item("mrd-1", "staging"))
    put_model_item(dynamodb, TABLE_NAME, _item("mrd-2", "staging"))
    promote_model(dynamodb, TABLE_NAME, "mrd-1")

    result = resolve_model(TABLE_NAME, "mrd-2")

    assert result is not None
    assert result["model_id"] == "mrd-2"


def test_resolve_model_falls_back_to_production(aws: tuple[S3Client, DynamoDBClient]) -> None:
    _s3, dynamodb = aws
    put_model_item(dynamodb, TABLE_NAME, _item("mrd-1", "staging"))
    promote_model(dynamodb, TABLE_NAME, "mrd-1")

    result = resolve_model(TABLE_NAME, None)

    assert result is not None
    assert result["model_id"] == "mrd-1"


def test_cached_list_models_filters_by_stage(aws: tuple[S3Client, DynamoDBClient]) -> None:
    _s3, dynamodb = aws
    put_model_item(dynamodb, TABLE_NAME, _item("mrd-1", "staging"))
    put_model_item(dynamodb, TABLE_NAME, _item("mrd-2", "development"))

    staging = cached_list_models(TABLE_NAME, "staging")

    assert {item["model_id"] for item in staging} == {"mrd-1"}


def test_cached_list_models_returns_all_without_a_stage_filter(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    _s3, dynamodb = aws
    put_model_item(dynamodb, TABLE_NAME, _item("mrd-1", "staging"))
    put_model_item(dynamodb, TABLE_NAME, _item("mrd-2", "development"))

    all_models = cached_list_models(TABLE_NAME, None)

    assert {item["model_id"] for item in all_models} == {"mrd-1", "mrd-2"}


def test_cached_read_json_returns_parsed_object(aws: tuple[S3Client, DynamoDBClient]) -> None:
    s3, _dynamodb = aws
    s3.put_object(Bucket=BUCKET, Key="predictions/model_id=mrd-1/latest.json", Body=b'{"a": 1}')

    result = cached_read_json(BUCKET, "predictions/model_id=mrd-1/latest.json")

    assert result == {"a": 1}


def test_cached_read_json_returns_none_when_key_missing(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    assert cached_read_json(BUCKET, "predictions/model_id=mrd-1/latest.json") is None
