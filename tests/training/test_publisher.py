import json
from collections.abc import Generator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws
from mypy_boto3_dynamodb.client import DynamoDBClient
from mypy_boto3_s3.client import S3Client

from market_regime_detector.common.model_registry import get_model
from market_regime_detector.training.publisher import publish_model

BUCKET = "test-bucket"
TABLE_NAME = "test-registry"
MODEL_NAME = "allweather-hmm2naive-us"


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
                {"AttributeName": "model_name", "AttributeType": "S"},
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
                },
                {
                    "IndexName": "model_name-index",
                    "KeySchema": [
                        {"AttributeName": "model_name", "KeyType": "HASH"},
                        {"AttributeName": "stage", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                    "ProvisionedThroughput": {"ReadCapacityUnits": 5, "WriteCapacityUnits": 5},
                },
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        yield s3, dynamodb


def _local_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "growth_hmm.npz").write_bytes(b"fake-npz-growth")
    (model_dir / "inflation_hmm.npz").write_bytes(b"fake-npz-inflation")
    (model_dir / "scalers.npz").write_bytes(b"fake-npz-scalers")
    (model_dir / "labels.json").write_text("{}")
    (model_dir / "config.json").write_text(
        json.dumps({"hyperparameters": {"covariance_type": "diag", "n_restarts": 10}})
    )
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "growth_log_likelihood": -123.4,
                "inflation_log_likelihood": -98.7,
                "growth_baseline_comparison": {"label_agreement_with_baseline": 0.6},
                "inflation_baseline_comparison": {"label_agreement_with_baseline": 0.55},
            }
        )
    )
    (model_dir / "model_card.md").write_text("# Model Card")
    return model_dir


def _manifest() -> dict[str, object]:
    return {
        "dataset_id": "ds-abc123",
        "created_at": "2024-01-01T00:00:00+00:00",
        "git_sha": None,
        "feature_config_version": "v1",
        "feature_config_hash": "deadbeef",
        "decision_frequency": "monthly",
        "date_start": "2000-01-31",
        "date_end": "2024-01-31",
        "vintage_aware": True,
        "earliest_realtime_start": "2000-01-01",
        "assets": [{"series_id": "TEST"}],
        "warnings": [],
    }


def test_publish_model_uploads_files_and_writes_registry_item(
    tmp_path: Path, aws: tuple[S3Client, DynamoDBClient]
) -> None:
    s3, dynamodb = aws
    model_dir = _local_model_dir(tmp_path)
    manifest = _manifest()

    model_id = publish_model(model_dir, MODEL_NAME, manifest, s3, BUCKET, dynamodb, TABLE_NAME)

    assert model_id.startswith(f"{MODEL_NAME}-")

    uploaded_keys = {obj["Key"] for obj in s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])}
    assert f"models/{MODEL_NAME}/{model_id}/growth_hmm.npz" in uploaded_keys
    assert f"models/{MODEL_NAME}/{model_id}/model_card.md" in uploaded_keys
    assert "datasets/ds-abc123/manifest.json" in uploaded_keys

    item = get_model(dynamodb, TABLE_NAME, model_id)
    assert item is not None
    assert item["model_name"] == MODEL_NAME
    assert item["stage"] == "staging"
    assert item["dataset_id"] == "ds-abc123"
    assert item["n_assets"] == 1
    assert "assets" not in item  # the full asset list must never land in DynamoDB
    assert item["model_s3_uri"] == f"s3://{BUCKET}/models/{MODEL_NAME}/{model_id}/"
    assert item["manifest_s3_uri"] == f"s3://{BUCKET}/datasets/ds-abc123/manifest.json"
    assert item["hmm_config_summary"] == {
        "n_states": 2,
        "covariance_type": "diag",
        "n_restarts": 10,
    }


def test_publish_model_respects_requested_stage(
    tmp_path: Path, aws: tuple[S3Client, DynamoDBClient]
) -> None:
    s3, dynamodb = aws
    model_dir = _local_model_dir(tmp_path)

    model_id = publish_model(
        model_dir, MODEL_NAME, _manifest(), s3, BUCKET, dynamodb, TABLE_NAME, stage="development"
    )

    item = get_model(dynamodb, TABLE_NAME, model_id)
    assert item is not None
    assert item["stage"] == "development"


def test_publish_model_rejects_invalid_model_name(
    tmp_path: Path, aws: tuple[S3Client, DynamoDBClient]
) -> None:
    s3, dynamodb = aws
    model_dir = _local_model_dir(tmp_path)

    with pytest.raises(ValueError, match="model_name"):
        publish_model(model_dir, "Not Valid!", _manifest(), s3, BUCKET, dynamodb, TABLE_NAME)


def test_publish_model_is_retry_safe(tmp_path: Path, aws: tuple[S3Client, DynamoDBClient]) -> None:
    s3, dynamodb = aws
    model_dir = _local_model_dir(tmp_path)
    manifest = _manifest()

    first_model_id = publish_model(
        model_dir, MODEL_NAME, manifest, s3, BUCKET, dynamodb, TABLE_NAME
    )
    second_model_id = publish_model(
        model_dir, MODEL_NAME, manifest, s3, BUCKET, dynamodb, TABLE_NAME
    )

    assert first_model_id == second_model_id

    first_item = get_model(dynamodb, TABLE_NAME, first_model_id)
    assert first_item is not None
    # Identity fields are unchanged by the retry — only last_modified legitimately
    # reflects each actual publish attempt's completion time.
    for key in ("model_id", "model_name", "dataset_id", "stage", "model_s3_uri", "manifest_s3_uri"):
        assert (
            first_item[key]
            == {
                "model_id": first_model_id,
                "model_name": MODEL_NAME,
                "dataset_id": "ds-abc123",
                "stage": "staging",
                "model_s3_uri": f"s3://{BUCKET}/models/{MODEL_NAME}/{first_model_id}/",
                "manifest_s3_uri": f"s3://{BUCKET}/datasets/ds-abc123/manifest.json",
            }[key]
        )

    uploaded_keys = {obj["Key"] for obj in s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])}
    # Still exactly one set of model files — no duplicate "-2" style keys.
    assert sum(1 for k in uploaded_keys if k.startswith("models/")) == 7
