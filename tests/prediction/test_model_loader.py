import io
import json
from collections.abc import Generator

import boto3
import numpy as np
import pytest
from moto import mock_aws
from mypy_boto3_s3.client import S3Client

from market_regime_detector.common.feature_config import FEATURE_CONFIG_V1
from market_regime_detector.prediction.model_loader import load_model_artifact

BUCKET = "test-bucket"
MODEL_ID = "mrd-20260101-abcdef12"


@pytest.fixture
def s3_client() -> Generator[S3Client, None, None]:
    with mock_aws():
        client = boto3.client("s3", region_name="eu-west-2")
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-2"}
        )
        yield client


def _put_npz(client: S3Client, key: str, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # type: ignore[arg-type]
    client.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())


def _put_json(client: S3Client, key: str, data: object) -> None:
    client.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(data).encode())


def _publish_fake_model(client: S3Client) -> None:
    prefix = f"models/{MODEL_ID}"
    _put_npz(
        client,
        f"{prefix}/growth_hmm.npz",
        start_prob=np.array([0.6, 0.4]),
        transition_matrix=np.array([[0.9, 0.1], [0.2, 0.8]]),
        means=np.array([[-1.0, -1.0], [1.0, 1.0]]),
        variances=np.array([[0.5, 0.5], [0.5, 0.5]]),
        feature_names=np.array(["f1", "f2"]),
    )
    _put_npz(
        client,
        f"{prefix}/inflation_hmm.npz",
        start_prob=np.array([0.5, 0.5]),
        transition_matrix=np.array([[0.85, 0.15], [0.25, 0.75]]),
        means=np.array([[-2.0], [2.0]]),
        variances=np.array([[0.3], [0.3]]),
        feature_names=np.array(["f3"]),
    )
    _put_npz(
        client,
        f"{prefix}/scalers.npz",
        growth_mean=np.array([0.0, 0.0]),
        growth_scale=np.array([1.0, 1.0]),
        inflation_mean=np.array([0.0]),
        inflation_scale=np.array([1.0]),
    )
    _put_json(
        client,
        f"{prefix}/labels.json",
        {
            "growth": {
                "state_labels": {"0": "DOWN", "1": "UP"},
                "composites": {"0": -1.0, "1": 1.0},
            },
            "inflation": {
                "state_labels": {"0": "DOWN", "1": "UP"},
                "composites": {"0": -2.0, "1": 2.0},
            },
        },
    )
    _put_json(
        client,
        f"{prefix}/config.json",
        {"feature_config": FEATURE_CONFIG_V1.to_dict(), "hyperparameters": {}},
    )


def test_load_model_artifact_reconstructs_hmm_arrays(s3_client: S3Client) -> None:
    _publish_fake_model(s3_client)

    artifact = load_model_artifact(s3_client, BUCKET, MODEL_ID)

    np.testing.assert_array_equal(artifact.growth.start_prob, [0.6, 0.4])
    np.testing.assert_array_equal(artifact.growth.means, [[-1.0, -1.0], [1.0, 1.0]])
    assert artifact.growth.feature_names == ["f1", "f2"]
    assert artifact.inflation.feature_names == ["f3"]


def test_load_model_artifact_reconstructs_scalers(s3_client: S3Client) -> None:
    _publish_fake_model(s3_client)

    artifact = load_model_artifact(s3_client, BUCKET, MODEL_ID)

    np.testing.assert_array_equal(artifact.growth_scaler_mean, [0.0, 0.0])
    np.testing.assert_array_equal(artifact.inflation_scaler_scale, [1.0])


def test_load_model_artifact_reconstructs_feature_config(s3_client: S3Client) -> None:
    _publish_fake_model(s3_client)

    artifact = load_model_artifact(s3_client, BUCKET, MODEL_ID)

    assert artifact.feature_config == FEATURE_CONFIG_V1


def test_up_state_resolves_from_string_keyed_labels(s3_client: S3Client) -> None:
    _publish_fake_model(s3_client)

    artifact = load_model_artifact(s3_client, BUCKET, MODEL_ID)

    assert artifact.growth.up_state == 1
    assert artifact.inflation.up_state == 1
