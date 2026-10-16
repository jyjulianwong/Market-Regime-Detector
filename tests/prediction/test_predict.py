import io
import json
from collections.abc import Generator
from datetime import UTC, date, datetime
from unittest.mock import call, patch

import boto3
import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM
from moto import mock_aws
from mypy_boto3_dynamodb.client import DynamoDBClient
from mypy_boto3_s3.client import S3Client

from market_regime_detector.common.feature_config import (
    Axis,
    FeatureConfig,
    FeatureDefinition,
    Frequency,
)
from market_regime_detector.common.forward_filter import filtered_posterior
from market_regime_detector.common.model_registry import promote_model, put_model_item
from market_regime_detector.ingestion.lake import write_to_lake
from market_regime_detector.prediction.predict import _fetch_raw_series, run_prediction

BUCKET = "test-bucket"
TABLE_NAME = "test-registry"
MODEL_ID = "mrd-20260101-abcdef12"

START_PROB = np.array([0.6, 0.4])
TRANSITION_MATRIX = np.array([[0.9, 0.1], [0.2, 0.8]])
GROWTH_MEANS = np.array([[-1.5], [1.5]])
INFLATION_MEANS = np.array([[-2.0], [2.0]])
VARIANCES = np.array([[0.5], [0.5]])


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


def _feature_config() -> FeatureConfig:
    growth = FeatureDefinition(
        name="growth_f1",
        axis=Axis.GROWTH,
        series_id="GROWTH_SERIES",
        source="fred",
        native_frequency=Frequency.MONTHLY,
        vintage_aware=False,
        publication_lag_days=0,
        transform="identity",
        transform_params={},
        sign=1,
    )
    inflation = FeatureDefinition(
        name="inflation_f1",
        axis=Axis.INFLATION,
        series_id="INFLATION_SERIES",
        source="fred",
        native_frequency=Frequency.MONTHLY,
        vintage_aware=False,
        publication_lag_days=0,
        transform="identity",
        transform_params={},
        sign=1,
    )
    return FeatureConfig(version="test", features=(growth, inflation))


def _sample_observations(means: np.ndarray, seed: int, n: int) -> np.ndarray:
    model = GaussianHMM(n_components=2, covariance_type="diag", random_state=seed)
    model.startprob_ = START_PROB
    model.transmat_ = TRANSITION_MATRIX
    model.means_ = means
    model.covars_ = VARIANCES
    observations, _states = model.sample(n)
    result: np.ndarray = observations[:, 0]
    return result


def _put_npz(s3: S3Client, key: str, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # type: ignore[arg-type]
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())


def _publish_fake_model(s3: S3Client) -> None:
    prefix = f"models/{MODEL_ID}"
    _put_npz(
        s3,
        f"{prefix}/growth_hmm.npz",
        start_prob=START_PROB,
        transition_matrix=TRANSITION_MATRIX,
        means=GROWTH_MEANS,
        variances=VARIANCES,
        feature_names=np.array(["growth_f1"]),
    )
    _put_npz(
        s3,
        f"{prefix}/inflation_hmm.npz",
        start_prob=START_PROB,
        transition_matrix=TRANSITION_MATRIX,
        means=INFLATION_MEANS,
        variances=VARIANCES,
        feature_names=np.array(["inflation_f1"]),
    )
    _put_npz(
        s3,
        f"{prefix}/scalers.npz",
        growth_mean=np.array([0.0]),
        growth_scale=np.array([1.0]),
        inflation_mean=np.array([0.0]),
        inflation_scale=np.array([1.0]),
    )
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{prefix}/labels.json",
        Body=json.dumps(
            {
                "growth": {"state_labels": {"0": "DOWN", "1": "UP"}, "composites": {}},
                "inflation": {"state_labels": {"0": "DOWN", "1": "UP"}, "composites": {}},
            }
        ).encode(),
    )
    s3.put_object(
        Bucket=BUCKET,
        Key=f"{prefix}/config.json",
        Body=json.dumps(
            {"feature_config": _feature_config().to_dict(), "hyperparameters": {}}
        ).encode(),
    )


def _write_lake_series(s3: S3Client, series_id: str, dates: list[str], values: list[float]) -> None:
    n = len(dates)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "value": values,
            "realtime_start": pd.to_datetime(dates),
            "source": ["fred"] * n,
            "series_id": [series_id] * n,
            "ingested_at": [datetime.now(UTC)] * n,
        }
    )
    write_to_lake(s3, BUCKET, df)


def _publish_and_promote(dynamodb: DynamoDBClient) -> None:
    put_model_item(
        dynamodb,
        TABLE_NAME,
        {
            "model_id": MODEL_ID,
            "dataset_id": "ds-test",
            "stage": "staging",
            "last_modified": "2024-01-01T00:00:00",
        },
    )
    promote_model(dynamodb, TABLE_NAME, MODEL_ID)


def test_run_prediction_matches_forward_filter_directly(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    s3, dynamodb = aws
    n_days = 20
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    growth_obs = _sample_observations(GROWTH_MEANS, seed=1, n=n_days)
    inflation_obs = _sample_observations(INFLATION_MEANS, seed=2, n=n_days)

    _write_lake_series(s3, "GROWTH_SERIES", date_strs, growth_obs.tolist())
    _write_lake_series(s3, "INFLATION_SERIES", date_strs, inflation_obs.tolist())
    _publish_fake_model(s3)
    _publish_and_promote(dynamodb)

    result = run_prediction(
        s3,
        dynamodb,
        BUCKET,
        TABLE_NAME,
        start=date(2024, 1, 1),
        end=date(2024, 1, 20),
        asof=date(2024, 1, 20),
    )

    expected_growth = filtered_posterior(
        growth_obs.reshape(-1, 1), START_PROB, TRANSITION_MATRIX, GROWTH_MEANS, VARIANCES
    )
    expected_inflation = filtered_posterior(
        inflation_obs.reshape(-1, 1), START_PROB, TRANSITION_MATRIX, INFLATION_MEANS, VARIANCES
    )

    np.testing.assert_allclose(
        result.history["p_growth_up"].to_numpy(), expected_growth[:, 1], atol=1e-8
    )
    np.testing.assert_allclose(
        result.history["p_inflation_up"].to_numpy(), expected_inflation[:, 1], atol=1e-8
    )
    assert result.model_id == MODEL_ID
    assert result.dataset_id == "ds-test"
    assert result.observation_date == date(2024, 1, 20)
    assert not result.is_stale
    assert result.warnings == []


def test_run_prediction_defaults_to_production_model(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    s3, dynamodb = aws
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    _write_lake_series(s3, "GROWTH_SERIES", dates, [1.0] * 10)
    _write_lake_series(s3, "INFLATION_SERIES", dates, [1.0] * 10)
    _publish_fake_model(s3)
    _publish_and_promote(dynamodb)

    result = run_prediction(
        s3, dynamodb, BUCKET, TABLE_NAME, start=date(2024, 1, 1), end=date(2024, 1, 10)
    )

    assert result.model_id == MODEL_ID


def test_run_prediction_honors_explicit_model_id(aws: tuple[S3Client, DynamoDBClient]) -> None:
    s3, dynamodb = aws
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    _write_lake_series(s3, "GROWTH_SERIES", dates, [1.0] * 10)
    _write_lake_series(s3, "INFLATION_SERIES", dates, [1.0] * 10)
    _publish_fake_model(s3)
    put_model_item(
        dynamodb,
        TABLE_NAME,
        {
            "model_id": MODEL_ID,
            "dataset_id": "ds-test",
            "stage": "staging",
            "last_modified": "2024-01-01T00:00:00",
        },
    )
    # Deliberately not promoted — only reachable by explicit model_id.

    result = run_prediction(
        s3,
        dynamodb,
        BUCKET,
        TABLE_NAME,
        model_id=MODEL_ID,
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
    )

    assert result.model_id == MODEL_ID


def test_run_prediction_raises_when_no_production_model_exists(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    s3, dynamodb = aws

    with pytest.raises(ValueError, match="no production model"):
        run_prediction(s3, dynamodb, BUCKET, TABLE_NAME)


def test_run_prediction_raises_for_unknown_model_id(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    s3, dynamodb = aws

    with pytest.raises(ValueError, match="not found"):
        run_prediction(s3, dynamodb, BUCKET, TABLE_NAME, model_id="does-not-exist")


def test_run_prediction_is_stale_when_raw_inputs_lag_asof(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    """`observation_date` correctly stays at `effective_end` — `as_of_lookup`
    forward-fills the last known value, so the feature itself stays "available."
    `is_stale` is what must catch the underlying series having gone quiet.
    """
    s3, dynamodb = aws
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]  # data stops Jan 10
    _write_lake_series(s3, "GROWTH_SERIES", dates, [1.0] * 10)
    _write_lake_series(s3, "INFLATION_SERIES", dates, [1.0] * 10)
    _publish_fake_model(s3)
    _publish_and_promote(dynamodb)

    result = run_prediction(
        s3,
        dynamodb,
        BUCKET,
        TABLE_NAME,
        start=date(2024, 1, 1),
        end=date(2024, 1, 20),
        asof=date(2024, 1, 20),  # 10 days after the last available observation
    )

    assert result.observation_date == date(2024, 1, 20)
    assert result.is_stale
    assert any("no data in the last" in w for w in result.warnings)


def test_fetch_raw_series_rejects_unknown_source(aws: tuple[S3Client, DynamoDBClient]) -> None:
    s3, _dynamodb = aws

    with pytest.raises(ValueError, match="unknown source"):
        _fetch_raw_series(
            s3,
            BUCKET,
            _feature_config(),
            "not_a_real_source",
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 2),
        )


def test_fetch_raw_series_live_uses_each_series_own_provider(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    s3, _dynamodb = aws
    config = _feature_config()  # growth_f1 from GROWTH_SERIES, inflation_f1 from INFLATION_SERIES
    assert config.features[0].source == "fred"

    fake_provider = type(
        "FakeProvider",
        (),
        {
            "fetch": lambda self, series_id, start, end, asof=None: pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01"]),
                    "value": [1.0],
                    "realtime_start": pd.to_datetime(["2024-01-01"]),
                    "source": ["fred"],
                    "series_id": [series_id],
                    "ingested_at": [datetime.now(UTC)],
                }
            )
        },
    )()

    with patch(
        "market_regime_detector.prediction.predict.build_provider", return_value=fake_provider
    ) as mock_build_provider:
        raw = _fetch_raw_series(
            s3, BUCKET, config, "live", date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)
        )

    assert set(raw) == {"GROWTH_SERIES", "INFLATION_SERIES"}
    assert mock_build_provider.call_args_list == [call("fred"), call("fred")]


def test_run_prediction_warns_when_a_series_has_no_data(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    s3, dynamodb = aws
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    _write_lake_series(s3, "GROWTH_SERIES", dates, [1.0] * 10)
    # INFLATION_SERIES never ingested.
    _publish_fake_model(s3)
    _publish_and_promote(dynamodb)

    result = run_prediction(
        s3, dynamodb, BUCKET, TABLE_NAME, start=date(2024, 1, 1), end=date(2024, 1, 10)
    )

    assert any("INFLATION_SERIES" in w for w in result.warnings)
    assert result.observation_date is None
    assert result.is_stale
    assert result.history.empty
