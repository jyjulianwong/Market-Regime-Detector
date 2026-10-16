import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def with_retries[T](
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    backoff_base_seconds: float = 1.0,
) -> T:
    """Calls `fn`, retrying on any exception with exponential backoff.

    Providers hit third-party rate limits (FRED, Alpha Vantage, yfinance all throttle
    free-tier usage); this is a deliberately minimal, dependency-free retry loop
    rather than pulling in a library like `tenacity` for ~15 lines of logic.
    """
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as e:  # intentionally broad — see docstring
            last_error = e
            if attempt == max_attempts - 1:
                break
            sleep_seconds = backoff_base_seconds * (2**attempt)
            logger.warning(
                "attempt %d/%d failed: %s — retrying in %.1fs",
                attempt + 1,
                max_attempts,
                e,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error
