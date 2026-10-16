import json
import logging
from collections.abc import Generator

import pytest

from market_regime_detector.common import logging as mrd_logging


@pytest.fixture(autouse=True)
def _reset_configured_flag() -> Generator[None, None, None]:
    mrd_logging._CONFIGURED = False
    yield
    mrd_logging._CONFIGURED = False


def test_configure_logging_emits_structured_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mrd_logging.configure_logging(level="DEBUG")
    logger = logging.getLogger("test.logger")

    logger.info("hello %s", "world")

    captured = capsys.readouterr()
    record = json.loads(captured.err.strip())

    assert record["level"] == "INFO"
    assert record["logger"] == "test.logger"
    assert record["message"] == "hello world"
    assert "timestamp" in record


def test_configure_logging_is_idempotent() -> None:
    mrd_logging.configure_logging(level="DEBUG")
    first_handlers = logging.getLogger().handlers

    mrd_logging.configure_logging(level="WARNING")

    assert logging.getLogger().handlers is first_handlers
    assert logging.getLogger().level == logging.DEBUG
