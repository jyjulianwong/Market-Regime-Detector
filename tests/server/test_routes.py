import json
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

from market_regime_detector.common.model_registry import promote_model, put_model_item
from market_regime_detector.common.settings import get_settings
from market_regime_detector.server.app import app

settings = get_settings()
BUCKET = settings.s3_bucket_name
TABLE_NAME = settings.dynamodb_table_name

client = TestClient(app)


@pytest.fixture(autouse=True)
def aws() -> Generator[None, None, None]:
    with mock_aws():
        s3 = boto3.client("s3", region_name=settings.aws_region)
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": settings.aws_region},  # type: ignore[typeddict-item]
        )
        dynamodb = boto3.client("dynamodb", region_name=settings.aws_region)
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
        yield


def _registry_item(model_id: str, stage: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "dataset_id": "ds-test",
        "stage": stage,
        "last_modified": "2024-01-01T00:00:00",
        "date_start": "2000-01-31",
        "date_end": "2024-01-31",
        "decision_frequency": "monthly",
        "feature_config_hash": "deadbeef",
        "n_assets": 9,
        "git_sha": None,
        "vintage_aware": True,
        "model_s3_uri": f"s3://{BUCKET}/models/{model_id}/",
        "manifest_s3_uri": f"s3://{BUCKET}/datasets/ds-test/manifest.json",
        "metrics_summary": {"growth_log_likelihood": Decimal("-100.0")},
        "hmm_config_summary": {"n_states": 2, "covariance_type": "diag", "n_restarts": 10},
    }


def _put_registry_item(model_id: str, stage: str) -> None:
    dynamodb = boto3.client("dynamodb", region_name=settings.aws_region)
    put_model_item(dynamodb, TABLE_NAME, _registry_item(model_id, stage))


def _promote(model_id: str) -> None:
    dynamodb = boto3.client("dynamodb", region_name=settings.aws_region)
    promote_model(dynamodb, TABLE_NAME, model_id)


def _latest_json(model_id: str, generated_at: str, is_stale: bool = False) -> dict[str, object]:
    return {
        "model_id": model_id,
        "dataset_id": "ds-test",
        "asof": "2024-01-20",
        "generated_at": generated_at,
        "observation_date": "2024-01-20",
        "is_stale": is_stale,
        "latest": {
            "p_growth_up": 0.7,
            "p_inflation_up": 0.3,
            "quadrants": {
                "GROWTH_UP_INFLATION_DOWN": 0.49,
                "GROWTH_UP_INFLATION_UP": 0.21,
                "GROWTH_DOWN_INFLATION_UP": 0.09,
                "GROWTH_DOWN_INFLATION_DOWN": 0.21,
            },
            "regime": "GROWTH_UP_INFLATION_DOWN",
            "confidence": 0.49,
        },
        "warnings": [],
        "disclaimer": "Descriptive regime information only; not investment advice.",
    }


def _history_json(model_id: str) -> dict[str, object]:
    record = {
        "date": "2024-01-20",
        "p_growth_up": 0.7,
        "p_inflation_up": 0.3,
        "quadrants": {
            "GROWTH_UP_INFLATION_DOWN": 0.49,
            "GROWTH_UP_INFLATION_UP": 0.21,
            "GROWTH_DOWN_INFLATION_UP": 0.09,
            "GROWTH_DOWN_INFLATION_DOWN": 0.21,
        },
        "regime": "GROWTH_UP_INFLATION_DOWN",
        "confidence": 0.49,
    }
    return {
        "model_id": model_id,
        "dataset_id": "ds-test",
        "generated_at": "2024-01-20T06:30:00+00:00",
        "history": [record, {**record, "date": "2024-01-21"}],
    }


def _put_predictions(model_id: str, generated_at: str, is_stale: bool = False) -> None:
    s3 = boto3.client("s3", region_name=settings.aws_region)
    s3.put_object(
        Bucket=BUCKET,
        Key=f"predictions/model_id={model_id}/latest.json",
        Body=json.dumps(_latest_json(model_id, generated_at, is_stale)).encode(),
    )
    s3.put_object(
        Bucket=BUCKET,
        Key=f"predictions/model_id={model_id}/history.json",
        Body=json.dumps(_history_json(model_id)).encode(),
    )


def test_health() -> None:
    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models_without_stage_filter_returns_all() -> None:
    _put_registry_item("mrd-1", "staging")
    _put_registry_item("mrd-2", "development")

    response = client.get("/v1/models")

    assert response.status_code == 200
    model_ids = {item["model_id"] for item in response.json()}
    assert model_ids == {"mrd-1", "mrd-2"}


def test_list_models_with_stage_filter() -> None:
    _put_registry_item("mrd-1", "staging")
    _put_registry_item("mrd-2", "development")

    response = client.get("/v1/models", params={"stage": "staging"})

    model_ids = {item["model_id"] for item in response.json()}
    assert model_ids == {"mrd-1"}


def test_get_model_detail_success() -> None:
    _put_registry_item("mrd-1", "staging")

    response = client.get("/v1/models/mrd-1")

    assert response.status_code == 200
    assert response.json()["model_id"] == "mrd-1"
    assert response.json()["hmm_config_summary"]["n_states"] == 2


def test_get_model_detail_404_for_unknown_model() -> None:
    response = client.get("/v1/models/does-not-exist")

    assert response.status_code == 404


def test_get_latest_prediction_defaults_to_production_model() -> None:
    _put_registry_item("mrd-1", "staging")
    _promote("mrd-1")
    _put_predictions("mrd-1", datetime.now(UTC).isoformat())

    response = client.get("/v1/predictions/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "mrd-1"
    assert body["latest"]["regime"] == "GROWTH_UP_INFLATION_DOWN"
    assert body["disclaimer"] == "Descriptive regime information only; not investment advice."
    assert body["is_stale"] is False


def test_get_latest_prediction_honors_explicit_model_id() -> None:
    _put_registry_item("mrd-1", "staging")
    _put_registry_item("mrd-2", "staging")
    _promote("mrd-1")
    _put_predictions("mrd-2", datetime.now(UTC).isoformat())

    response = client.get("/v1/predictions/latest", params={"model_id": "mrd-2"})

    assert response.status_code == 200
    assert response.json()["model_id"] == "mrd-2"


def test_get_latest_prediction_404_when_model_unknown() -> None:
    response = client.get("/v1/predictions/latest", params={"model_id": "does-not-exist"})

    assert response.status_code == 404


def test_get_latest_prediction_404_when_no_production_model() -> None:
    response = client.get("/v1/predictions/latest")

    assert response.status_code == 404


def test_get_latest_prediction_404_when_model_exists_but_has_no_predictions() -> None:
    """A separate 404 from "model not found" — brief: "return a clear 404 if no
    predictions exist for the requested model."
    """
    _put_registry_item("mrd-1", "staging")
    _promote("mrd-1")
    # No latest.json ever written.

    response = client.get("/v1/predictions/latest")

    assert response.status_code == 404
    assert "no predictions exist" in response.json()["detail"]


def test_get_latest_prediction_marks_stale_when_generated_at_is_old() -> None:
    _put_registry_item("mrd-1", "staging")
    _promote("mrd-1")
    old_timestamp = (datetime.now(UTC) - timedelta(hours=100)).isoformat()
    _put_predictions("mrd-1", old_timestamp, is_stale=False)  # batch itself said fresh

    response = client.get("/v1/predictions/latest")

    body = response.json()
    assert body["is_stale"] is True
    assert any("freshness threshold" in w for w in body["warnings"])


def test_get_prediction_history_default() -> None:
    _put_registry_item("mrd-1", "staging")
    _promote("mrd-1")
    _put_predictions("mrd-1", datetime.now(UTC).isoformat())

    response = client.get("/v1/predictions/history")

    assert response.status_code == 200
    body = response.json()
    assert len(body["history"]) == 2
    assert body["disclaimer"] == "Descriptive regime information only; not investment advice."


def test_get_prediction_history_filters_by_date_range() -> None:
    _put_registry_item("mrd-1", "staging")
    _promote("mrd-1")
    _put_predictions("mrd-1", datetime.now(UTC).isoformat())

    response = client.get(
        "/v1/predictions/history", params={"start": "2024-01-21", "end": "2024-01-21"}
    )

    body = response.json()
    assert len(body["history"]) == 1
    assert body["history"][0]["date"] == "2024-01-21"


def test_get_prediction_history_404_when_no_predictions_exist() -> None:
    _put_registry_item("mrd-1", "staging")
    _promote("mrd-1")

    response = client.get("/v1/predictions/history")

    assert response.status_code == 404
