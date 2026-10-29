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
        patch("market_regime_detector.prediction.handler.list_models_by_stage") as list_production,
        patch("market_regime_detector.prediction.handler.run_prediction") as run_pred,
        patch("market_regime_detector.prediction.handler.write_prediction_outputs") as write_out,
    ):
        result = MagicMock()
        result.model_id = "allweather-hmm2naive-us-20260101-abcdef12"
        result.model_name = "allweather-hmm2naive-us"
        result.observation_date.isoformat.return_value = "2026-01-01"
        result.is_stale = False
        run_pred.return_value = result
        list_production.return_value = [{"model_name": "allweather-hmm2naive-us"}]
        yield {
            "ingest_incremental": ingest,
            "get_s3_client": get_s3,
            "get_dynamodb_client": get_dynamodb,
            "list_models_by_stage": list_production,
            "run_prediction": run_pred,
            "write_prediction_outputs": write_out,
            "result": result,
        }


# --- action="predict" --------------------------------------------------------


def test_empty_event_fans_out_over_every_production_model(
    mocked_dependencies: dict[str, MagicMock],
) -> None:
    mocked_dependencies["list_models_by_stage"].return_value = [
        {"model_name": "allweather-hmm2naive-us"},
        {"model_name": "allweather-hmm2naive-eu"},
    ]

    response = handler({})

    assert mocked_dependencies["run_prediction"].call_count == 2
    called_model_names = {
        kwargs["model_name"]
        for _args, kwargs in mocked_dependencies["run_prediction"].call_args_list
    }
    assert called_model_names == {"allweather-hmm2naive-us", "allweather-hmm2naive-eu"}
    for _args, kwargs in mocked_dependencies["run_prediction"].call_args_list:
        assert kwargs["model_id"] is None
        assert kwargs["source"] == "lake"
        assert kwargs["start"] is None
        assert kwargs["end"] is None
        assert kwargs["asof"] is None
    results = response["results"]
    assert isinstance(results, list)
    assert len(results) == 2
    assert response["failed"] == []


def test_predict_action_never_runs_ingestion(mocked_dependencies: dict[str, MagicMock]) -> None:
    handler({"action": "predict"})

    mocked_dependencies["ingest_incremental"].assert_not_called()


def test_explicit_model_id_in_event_predicts_only_that_one_model(
    mocked_dependencies: dict[str, MagicMock],
) -> None:
    handler({"action": "predict", "model_id": "allweather-hmm2naive-us-20250101-deadbeef"})

    mocked_dependencies["list_models_by_stage"].assert_not_called()
    mocked_dependencies["run_prediction"].assert_called_once()
    _args, kwargs = mocked_dependencies["run_prediction"].call_args
    assert kwargs["model_id"] == "allweather-hmm2naive-us-20250101-deadbeef"
    assert kwargs["model_name"] is None


def test_explicit_model_name_in_event_predicts_only_that_one_model(
    mocked_dependencies: dict[str, MagicMock],
) -> None:
    handler({"action": "predict", "model_name": "allweather-hmm2naive-us"})

    mocked_dependencies["list_models_by_stage"].assert_not_called()
    mocked_dependencies["run_prediction"].assert_called_once()
    _args, kwargs = mocked_dependencies["run_prediction"].call_args
    assert kwargs["model_id"] is None
    assert kwargs["model_name"] == "allweather-hmm2naive-us"


def test_explicit_dates_are_parsed_from_event(mocked_dependencies: dict[str, MagicMock]) -> None:
    handler(
        {
            "action": "predict",
            "model_name": "allweather-hmm2naive-us",
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
    handler({"action": "predict", "model_name": "allweather-hmm2naive-us"})

    write_outputs = mocked_dependencies["write_prediction_outputs"]
    write_outputs.assert_called_once()
    s3_arg, bucket_arg, result_arg = write_outputs.call_args[0]
    assert s3_arg is mocked_dependencies["get_s3_client"].return_value
    assert isinstance(bucket_arg, str)
    assert result_arg is mocked_dependencies["result"]


def test_one_model_failing_does_not_prevent_others_from_predicting(
    mocked_dependencies: dict[str, MagicMock],
) -> None:
    mocked_dependencies["list_models_by_stage"].return_value = [
        {"model_name": "allweather-hmm2naive-us"},
        {"model_name": "allweather-hmm2naive-eu"},
    ]
    good_result = mocked_dependencies["result"]

    def side_effect(*_args: object, **kwargs: object) -> object:
        if kwargs["model_name"] == "allweather-hmm2naive-eu":
            raise RuntimeError("no production model for 'allweather-hmm2naive-eu'")
        return good_result

    mocked_dependencies["run_prediction"].side_effect = side_effect

    response = handler({})  # must not raise

    results = response["results"]
    assert isinstance(results, list)
    assert len(results) == 1
    assert response["failed"] == ["allweather-hmm2naive-eu"]


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
