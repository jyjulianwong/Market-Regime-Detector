import pytest

from market_regime_detector.ingestion.retry import with_retries


def test_returns_result_on_first_success() -> None:
    assert with_retries(lambda: 42) == 42


def test_retries_then_succeeds() -> None:
    attempts = {"count": 0}

    def flaky() -> int:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient")
        return 99

    result = with_retries(flaky, max_attempts=3, backoff_base_seconds=0.0)

    assert result == 99
    assert attempts["count"] == 2


def test_raises_after_exhausting_attempts() -> None:
    def always_fails() -> int:
        raise RuntimeError("permanent")

    with pytest.raises(RuntimeError, match="permanent"):
        with_retries(always_fails, max_attempts=2, backoff_base_seconds=0.0)
