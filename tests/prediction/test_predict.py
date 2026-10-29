import io
import json
from collections.abc import Generator
from datetime import UTC, date, datetime
from unittest.mock import call, patch

import boto3
import numpy as np
import pandas as pd
import pytest
from moto import mock_aws
from mypy_boto3_dynamodb.client import DynamoDBClient
from mypy_boto3_s3.client import S3Client

from market_regime_detector.common.feature_config import (
    Axis,
    FeatureConfig,
    FeatureDefinition,
    Frequency,
)
from market_regime_detector.common.forward_filter import filtered_posterior, preview_posterior
from market_regime_detector.common.model_registry import promote_model, put_model_item
from market_regime_detector.ingestion.lake import write_to_lake
from market_regime_detector.prediction.predict import (
    PredictionResult,
    _fetch_raw_series,
    run_prediction,
)

BUCKET = "test-bucket"
TABLE_NAME = "test-registry"
MODEL_NAME = "allweather-hmm2naive-us"
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


def _feature_config() -> FeatureConfig:
    # DAILY (not MONTHLY) native frequency, matching every real first-pass market
    # feature (e.g. T10Y2Y) — this is what routes `build_feature_value` through the
    # actual resample("ME") + latest_value pipeline `_axis_posteriors`'s month-end
    # vs. daily-preview split depends on, rather than the plain as-of lookup a
    # MONTHLY-declared feature would use regardless of calendar month boundaries.
    growth = FeatureDefinition(
        name="growth_f1",
        axis=Axis.GROWTH,
        series_id="GROWTH_SERIES",
        source="fred",
        native_frequency=Frequency.DAILY,
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
        native_frequency=Frequency.DAILY,
        vintage_aware=False,
        publication_lag_days=0,
        transform="identity",
        transform_params={},
        sign=1,
    )
    return FeatureConfig(version="test", features=(growth, inflation))


def _put_npz(s3: S3Client, key: str, **arrays: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # type: ignore[arg-type]
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())


def _publish_fake_model(s3: S3Client) -> None:
    prefix = f"models/{MODEL_NAME}/{MODEL_ID}"
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
            "model_name": MODEL_NAME,
            "dataset_id": "ds-test",
            "stage": "staging",
            "last_modified": "2024-01-01T00:00:00",
        },
    )
    promote_model(dynamodb, TABLE_NAME, MODEL_ID)


# One raw observation per calendar month, dated the 1st — with DAILY native
# frequency + transform="identity", that single point becomes the whole month's
# feature value from day 1 onward (nothing else arrives that month), so every test
# below knows exactly what `_axis_posteriors` should see at any day in any month
# without needing to reverse-engineer pandas' resample behaviour by hand.
MONTHLY_TEST_DATES = ["2023-11-01", "2023-12-01", "2024-01-01", "2024-02-01"]
GROWTH_MONTHLY_VALUES = [-1.0, 1.0, -0.5, 1.5]
INFLATION_MONTHLY_VALUES = [-2.0, 2.0, 0.0, 1.0]
# Index into the arrays above / into `filtered_posterior`'s output: 0=Nov, 1=Dec,
# 2=Jan, 3=Feb.


def _setup_monthly_fixture(s3: S3Client, dynamodb: DynamoDBClient) -> None:
    _write_lake_series(s3, "GROWTH_SERIES", MONTHLY_TEST_DATES, GROWTH_MONTHLY_VALUES)
    _write_lake_series(s3, "INFLATION_SERIES", MONTHLY_TEST_DATES, INFLATION_MONTHLY_VALUES)
    _publish_fake_model(s3)
    _publish_and_promote(dynamodb)


def _run_monthly_fixture_prediction(s3: S3Client, dynamodb: DynamoDBClient) -> PredictionResult:
    return run_prediction(
        s3,
        dynamodb,
        BUCKET,
        TABLE_NAME,
        model_name=MODEL_NAME,
        start=date(2023, 11, 1),
        end=date(2024, 2, 29),
        asof=date(2024, 2, 29),
    )


def test_run_prediction_month_end_day_equals_monthly_filtered_posterior(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    """Issue A's fix, checked directly: at an exact month-end, the daily preview
    must reproduce the real recursive monthly `filtered_posterior` exactly, because
    it is the same recursion step, not an approximation of it."""
    s3, dynamodb = aws
    _setup_monthly_fixture(s3, dynamodb)

    result = _run_monthly_fixture_prediction(s3, dynamodb)

    expected_growth = filtered_posterior(
        np.array(GROWTH_MONTHLY_VALUES).reshape(-1, 1),
        START_PROB,
        TRANSITION_MATRIX,
        GROWTH_MEANS,
        VARIANCES,
    )
    expected_inflation = filtered_posterior(
        np.array(INFLATION_MONTHLY_VALUES).reshape(-1, 1),
        START_PROB,
        TRANSITION_MATRIX,
        INFLATION_MEANS,
        VARIANCES,
    )

    for month_end, month_idx in (
        (pd.Timestamp("2023-12-31"), 1),
        (pd.Timestamp("2024-01-31"), 2),
        (pd.Timestamp("2024-02-29"), 3),
    ):
        assert result.history.loc[month_end, "p_growth_up"] == pytest.approx(
            expected_growth[month_idx, 1], abs=1e-8
        )
        assert result.history.loc[month_end, "p_inflation_up"] == pytest.approx(
            expected_inflation[month_idx, 1], abs=1e-8
        )


def test_run_prediction_mid_month_day_equals_preview_posterior_from_anchor(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    """A mid-month day must equal exactly one `preview_posterior` step off the
    *previous* month's filtered posterior — never a recursive daily filter fed
    that day's (and every other day's) feature value."""
    s3, dynamodb = aws
    _setup_monthly_fixture(s3, dynamodb)

    result = _run_monthly_fixture_prediction(s3, dynamodb)

    expected_growth_monthly = filtered_posterior(
        np.array(GROWTH_MONTHLY_VALUES).reshape(-1, 1),
        START_PROB,
        TRANSITION_MATRIX,
        GROWTH_MEANS,
        VARIANCES,
    )
    december_anchor = expected_growth_monthly[1]
    expected_mid_january = preview_posterior(
        december_anchor,
        TRANSITION_MATRIX,
        np.array([[GROWTH_MONTHLY_VALUES[2]]]),
        GROWTH_MEANS,
        VARIANCES,
    )

    actual = result.history.loc[pd.Timestamp("2024-01-15"), "p_growth_up"]
    assert actual == pytest.approx(expected_mid_january[0, 1], abs=1e-8)


def test_run_prediction_constant_within_month_feature_does_not_drift(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    """Regression test for the original bug: running ~30 identical daily
    observations through a recursive `filtered_posterior` used to saturate the
    posterior toward whichever state that single observation favoured, more so
    the deeper into the month you looked. With the monthly-anchor + one-step daily
    preview design, every day in the same month previews off the same anchor with
    the same feature value, so two different days in that month must come out
    exactly equal — not just close, and never drifting apart."""
    s3, dynamodb = aws
    _setup_monthly_fixture(s3, dynamodb)

    result = _run_monthly_fixture_prediction(s3, dynamodb)

    early = result.history.loc[pd.Timestamp("2024-01-05")]
    late = result.history.loc[pd.Timestamp("2024-01-25")]

    assert early["p_growth_up"] == pytest.approx(late["p_growth_up"], abs=1e-12)
    assert early["p_inflation_up"] == pytest.approx(late["p_inflation_up"], abs=1e-12)


def test_run_prediction_drops_days_with_no_prior_month_end_anchor(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    """The lookback window's first (partial) month has no completed prior
    month-end to anchor off, so its days must be silently dropped — never chained
    to some earlier, nonexistent anchor."""
    s3, dynamodb = aws
    _setup_monthly_fixture(s3, dynamodb)

    result = _run_monthly_fixture_prediction(s3, dynamodb)

    assert result.history.index.min() == pd.Timestamp("2023-12-01")
    assert pd.Timestamp("2023-11-30") not in result.history.index
    assert pd.Timestamp("2023-11-15") not in result.history.index


def test_run_prediction_resolves_production_model_by_name(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    s3, dynamodb = aws
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    _write_lake_series(s3, "GROWTH_SERIES", dates, [1.0] * 10)
    _write_lake_series(s3, "INFLATION_SERIES", dates, [1.0] * 10)
    _publish_fake_model(s3)
    _publish_and_promote(dynamodb)

    result = run_prediction(
        s3,
        dynamodb,
        BUCKET,
        TABLE_NAME,
        model_name=MODEL_NAME,
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
    )

    assert result.model_id == MODEL_ID
    assert result.model_name == MODEL_NAME


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
            "model_name": MODEL_NAME,
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


def test_run_prediction_raises_when_neither_model_id_nor_model_name_given(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    s3, dynamodb = aws

    with pytest.raises(ValueError, match="either model_id or model_name"):
        run_prediction(s3, dynamodb, BUCKET, TABLE_NAME)


def test_run_prediction_raises_when_no_production_model_exists_for_name(
    aws: tuple[S3Client, DynamoDBClient],
) -> None:
    s3, dynamodb = aws

    with pytest.raises(ValueError, match="no production model"):
        run_prediction(s3, dynamodb, BUCKET, TABLE_NAME, model_name=MODEL_NAME)


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

    Both series are declared MONTHLY (`_feature_config`), so the gap has to clear
    that frequency's own tolerance (one full publication cycle plus slack, not a
    flat few days) to actually count as stale — see predict.py's
    stale_tolerance_days_by_frequency.
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
        model_name=MODEL_NAME,
        start=date(2024, 1, 1),
        end=date(2024, 3, 1),
        asof=date(2024, 3, 1),  # 51 days after the last available observation
    )

    assert result.observation_date == date(2024, 3, 1)
    assert result.is_stale
    assert any("no new vintage in the last" in w for w in result.warnings)


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
        s3,
        dynamodb,
        BUCKET,
        TABLE_NAME,
        model_name=MODEL_NAME,
        start=date(2024, 1, 1),
        end=date(2024, 1, 10),
    )

    assert any("INFLATION_SERIES" in w for w in result.warnings)
    assert result.observation_date is None
    assert result.is_stale
    assert result.history.empty
