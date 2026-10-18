from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from market_regime_detector.ingestion.series_catalog import FIRST_PASS_SERIES
from market_regime_detector.prediction.handler import handler


@pytest.fixture
def mocked_dependencies() -> Generator[dict[str, MagicMock], None, None]:
    with (
        patch("market_regime_detector.prediction.handler.ingest_incremental") as ingest,
        patch("market_regime_detector.prediction.handler.get_s3_client") as get_s3,
        patch("market_regime_detector.prediction.handler.get_dynamodb_client") as get_dynamodb,
        patch("market_regime_detector.prediction.handler.run_prediction") as run_pred,
        patch("market_regime_detector.prediction.handler.write_prediction_outputs") as write_out,
    ):
        result = MagicMock()
        result.model_id = "mrd-20260101-abcdef12"
        result.observation_date.isoformat.return_value = "2026-01-01"
        result.is_stale = False
        run_pred.return_value = result
        yield {
            "ingest_incremental": ingest,
            "get_s3_client": get_s3,
            "get_dynamodb_client": get_dynamodb,
            "run_prediction": run_pred,
            "write_prediction_outputs": write_out,
            "result": result,
        }


# --- action="predict" --------------------------------------------------------


def test_empty_event_defaults_to_predict_with_production_lake_latest(
    mocked_dependencies: dict[str, MagicMock],
) -> None:
    response = handler({})

    mocked_dependencies["run_prediction"].assert_called_once()
    _args, kwargs = mocked_dependencies["run_prediction"].call_args
    assert kwargs["model_id"] is None
    assert kwargs["source"] == "lake"
    assert kwargs["start"] is None
    assert kwargs["end"] is None
    assert kwargs["asof"] is None
    assert response["model_id"] == "mrd-20260101-abcdef12"


def test_predict_action_never_runs_ingestion(mocked_dependencies: dict[str, MagicMock]) -> None:
    handler({"action": "predict"})

    mocked_dependencies["ingest_incremental"].assert_not_called()


def test_explicit_model_id_in_event_is_honored(mocked_dependencies: dict[str, MagicMock]) -> None:
    handler({"action": "predict", "model_id": "mrd-20250101-deadbeef"})

    _args, kwargs = mocked_dependencies["run_prediction"].call_args
    assert kwargs["model_id"] == "mrd-20250101-deadbeef"


def test_explicit_dates_are_parsed_from_event(mocked_dependencies: dict[str, MagicMock]) -> None:
    handler(
        {
            "action": "predict",
            "start": "2024-01-01",
            "end": "2024-02-01",
            "asof": "2024-02-01",
        }
    )

    _args, kwargs = mocked_dependencies["run_prediction"].call_args
    assert kwargs["start"].isoformat() == "2024-01-01"
    assert kwargs["end"].isoformat() == "2024-02-01"
    assert kwargs["asof"].isoformat() == "2024-02-01"


def test_writes_outputs_after_prediction(mocked_dependencies: dict[str, MagicMock]) -> None:
    handler({"action": "predict"})

    write_outputs = mocked_dependencies["write_prediction_outputs"]
    write_outputs.assert_called_once()
    s3_arg, bucket_arg, result_arg = write_outputs.call_args[0]
    assert s3_arg is mocked_dependencies["get_s3_client"].return_value
    assert isinstance(bucket_arg, str)
    assert result_arg is mocked_dependencies["result"]


# --- action="ingest" ----------------------------------------------------------


def test_ingest_action_runs_incremental_ingestion_for_every_first_pass_series(
    mocked_dependencies: dict[str, MagicMock],
) -> None:
    handler({"action": "ingest"})

    assert mocked_dependencies["ingest_incremental"].call_count == len(FIRST_PASS_SERIES)


def test_ingest_action_never_runs_prediction(mocked_dependencies: dict[str, MagicMock]) -> None:
    handler({"action": "ingest"})

    mocked_dependencies["run_prediction"].assert_not_called()
    mocked_dependencies["write_prediction_outputs"].assert_not_called()


def test_a_single_series_ingestion_failure_does_not_prevent_others(
    mocked_dependencies: dict[str, MagicMock],
) -> None:
    mocked_dependencies["ingest_incremental"].side_effect = RuntimeError("provider is down")

    response = handler({"action": "ingest"})  # must not raise

    assert response["series_ingested"] == 0
    assert response["series_failed"] == [spec.series_id for spec in FIRST_PASS_SERIES]


# --- dispatch -------------------------------------------------------------


def test_unknown_action_raises(mocked_dependencies: dict[str, MagicMock]) -> None:
    with pytest.raises(ValueError, match="unknown action"):
        handler({"action": "bogus"})
