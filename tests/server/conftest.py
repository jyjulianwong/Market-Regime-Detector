from collections.abc import Generator

import pytest

from market_regime_detector.server.cache import CACHE


@pytest.fixture(autouse=True)
def _clear_server_cache() -> Generator[None, None, None]:
    CACHE.clear()
    yield
    CACHE.clear()
