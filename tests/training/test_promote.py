import json
import logging
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from market_regime_detector.training.promote import invoke_batch_lambda_best_effort


def test_invokes_the_lambda_with_model_id_payload_when_function_name_is_set() -> None:
    client = MagicMock()

    invoke_batch_lambda_best_effort(client, "my-batch-function", "mrd-20260101-abcdef12")

    client.invoke.assert_called_once_with(
        FunctionName="my-batch-function",
        InvocationType="Event",
        Payload=json.dumps({"model_id": "mrd-20260101-abcdef12"}).encode(),
    )


def test_skips_invoke_when_no_function_name_configured() -> None:
    client = MagicMock()

    invoke_batch_lambda_best_effort(client, "", "mrd-20260101-abcdef12")

    client.invoke.assert_not_called()


def test_swallows_client_error_when_lambda_does_not_exist_yet(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = MagicMock()
    client.invoke.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "nope"}}, "Invoke"
    )

    with caplog.at_level(logging.WARNING):
        invoke_batch_lambda_best_effort(
            client, "not-deployed-yet", "mrd-20260101-abcdef12"
        )  # must not raise

    assert "could not invoke batch Lambda" in caplog.text
