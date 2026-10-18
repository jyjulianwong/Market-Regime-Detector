import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class TTLCache:
    """A minimal get-or-set cache, no `cachetools` dependency — brief section 3.4's
    "short-TTL caching of registry and S3 reads."

    Deliberately a plain module-level dict, not request-scoped: the whole point is
    to survive warm Lambda container reuse, so a repeat request hitting the same
    warm container within the TTL window skips DynamoDB/S3 entirely.
    """

    def __init__(self, ttl_seconds: float) -> None:
        self._ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, object]] = {}

    def get_or_set(self, key: str, factory: Callable[[], T]) -> T:
        now = time.monotonic()
        cached = self._store.get(key)
        if cached is not None and cached[0] > now:
            return cached[1]  # type: ignore[return-value]

        value = factory()
        self._store[key] = (now + self._ttl_seconds, value)
        return value

    def clear(self) -> None:
        self._store.clear()


# Brief says "short-TTL," not an exact duration — 30s is a judgment call.
# Tests clear this between cases (see tests/server/conftest.py).
CACHE = TTLCache(ttl_seconds=30.0)
