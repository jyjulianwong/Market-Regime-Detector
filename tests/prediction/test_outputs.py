import json
from collections.abc import Generator
from datetime import UTC, date, datetime

import boto3
import pandas as pd
import pytest
from moto import mock_aws
from mypy_boto3_s3.client import S3Client

from market_regime_detector.common.disclaimer import DISCLAIMER
from market_regime_detector.prediction.outputs import (
    to_history_json,
    to_latest_json,
    write_prediction_outputs,
)
from market_regime_detector.prediction.predict import PredictionResult

BUCKET = "test-bucket"


@pytest.fixture
def s3_client() -> Generator[S3Client, None, None]:
    with mock_aws():
        client = boto3.client("s3", region_name="eu-west-2")
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "eu-west-2"}
        )
        yield client


def _history_df(dates: list[str]) -> pd.DataFrame:
    n = len(dates)
    index = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "p_growth_up": [0.7] * n,
            "p_inflation_up": [0.3] * n,
            "GROWTH_UP_INFLATION_DOWN": [0.49] * n,
            "GROWTH_UP_INFLATION_UP": [0.21] * n,
            "GROWTH_DOWN_INFLATION_UP": [0.09] * n,
            "GROWTH_DOWN_INFLATION_DOWN": [0.21] * n,
            "regime": ["GROWTH_UP_INFLATION_DOWN"] * n,
            "confidence": [0.49] * n,
        },
        index=index,
    )


def _result(
    model_id: str = "mrd-20260101-abcdef12", asof: date = date(2024, 1, 20)
) -> PredictionResult:
    return PredictionResult(
        model_id=model_id,
        dataset_id="ds-test",
        asof=asof,
        generated_at=datetime(2024, 1, 20, 6, 30, tzinfo=UTC),
        observation_date=date(2024, 1, 20),
        is_stale=False,
        warnings=[],
        history=_history_df(["2024-01-19", "2024-01-20"]),
    )


def test_to_latest_json_shape() -> None:
    result = _result()

    latest_json = to_latest_json(result)

    assert latest_json["model_id"] == "mrd-20260101-abcdef12"
    assert latest_json["dataset_id"] == "ds-test"
    assert latest_json["observation_date"] == "2024-01-20"
    assert latest_json["is_stale"] is False
    assert latest_json["disclaimer"] == DISCLAIMER
    latest = latest_json["latest"]
    assert isinstance(latest, dict)
    assert latest["p_growth_up"] == pytest.approx(0.7)
    assert latest["regime"] == "GROWTH_UP_INFLATION_DOWN"
    assert set(latest["quadrants"]) == {
        "GROWTH_UP_INFLATION_DOWN",
        "GROWTH_UP_INFLATION_UP",
        "GROWTH_DOWN_INFLATION_UP",
        "GROWTH_DOWN_INFLATION_DOWN",
    }


def test_to_latest_json_handles_empty_history() -> None:
    result = _result()
    result.history = pd.DataFrame()
    result.observation_date = None
    result.is_stale = True

    latest_json = to_latest_json(result)

    assert latest_json["latest"] is None
    assert latest_json["observation_date"] is None
    assert latest_json["is_stale"] is True


def test_to_history_json_has_one_record_per_row() -> None:
    result = _result()

    history_json = to_history_json(result)

    records = history_json["history"]
    assert isinstance(records, list)
    assert len(records) == 2
    assert records[0]["date"] == "2024-01-19"
    assert records[1]["date"] == "2024-01-20"


def test_write_prediction_outputs_always_overwrites_latest_and_history(
    s3_client: S3Client,
) -> None:
    write_prediction_outputs(s3_client, BUCKET, _result())
    write_prediction_outputs(s3_client, BUCKET, _result())  # same asof, re-run

    prefix = "predictions/model_id=mrd-20260101-abcdef12"
    latest = json.loads(
        s3_client.get_object(Bucket=BUCKET, Key=f"{prefix}/latest.json")["Body"].read()
    )
    assert latest["model_id"] == "mrd-20260101-abcdef12"


def test_write_prediction_outputs_snapshot_is_write_once(s3_client: S3Client) -> None:
    prefix = "predictions/model_id=mrd-20260101-abcdef12"
    snapshot_key = f"{prefix}/snapshots/asof=2024-01-20.json"

    first_result = _result()
    write_prediction_outputs(s3_client, BUCKET, first_result)
    first_snapshot = s3_client.get_object(Bucket=BUCKET, Key=snapshot_key)["Body"].read()

    # A second run for the *same* asof with different (revised) data.
    revised_result = _result()
    revised_result.history = _history_df(["2024-01-19", "2024-01-20"])
    revised_result.history["p_growth_up"] = 0.1  # clearly different values
    write_prediction_outputs(s3_client, BUCKET, revised_result)

    second_snapshot = s3_client.get_object(Bucket=BUCKET, Key=snapshot_key)["Body"].read()
    assert first_snapshot == second_snapshot  # untouched by the revised re-run

    # But latest.json, unlike the snapshot, does reflect the revised run.
    latest = json.loads(
        s3_client.get_object(Bucket=BUCKET, Key=f"{prefix}/latest.json")["Body"].read()
    )
    assert latest["latest"]["p_growth_up"] == pytest.approx(0.1)


def test_write_prediction_outputs_different_asof_gets_its_own_snapshot(
    s3_client: S3Client,
) -> None:
    write_prediction_outputs(s3_client, BUCKET, _result(asof=date(2024, 1, 20)))
    write_prediction_outputs(s3_client, BUCKET, _result(asof=date(2024, 1, 21)))

    prefix = "predictions/model_id=mrd-20260101-abcdef12"
    keys = {
        obj["Key"]
        for obj in s3_client.list_objects_v2(Bucket=BUCKET, Prefix=f"{prefix}/snapshots/").get(
            "Contents", []
        )
    }
    assert keys == {
        f"{prefix}/snapshots/asof=2024-01-20.json",
        f"{prefix}/snapshots/asof=2024-01-21.json",
    }
