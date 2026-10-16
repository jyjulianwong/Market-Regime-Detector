from market_regime_detector.server.cache import TTLCache


def test_get_or_set_calls_factory_only_once_within_ttl() -> None:
    cache = TTLCache(ttl_seconds=60.0)
    calls = {"count": 0}

    def factory() -> int:
        calls["count"] += 1
        return 42

    first = cache.get_or_set("key", factory)
    second = cache.get_or_set("key", factory)

    assert first == 42
    assert second == 42
    assert calls["count"] == 1


def test_get_or_set_recomputes_after_ttl_expires() -> None:
    cache = TTLCache(ttl_seconds=0.0)
    calls = {"count": 0}

    def factory() -> int:
        calls["count"] += 1
        return calls["count"]

    first = cache.get_or_set("key", factory)
    second = cache.get_or_set("key", factory)

    assert first == 1
    assert second == 2


def test_different_keys_are_cached_independently() -> None:
    cache = TTLCache(ttl_seconds=60.0)

    a = cache.get_or_set("a", lambda: "value-a")
    b = cache.get_or_set("b", lambda: "value-b")

    assert a == "value-a"
    assert b == "value-b"


def test_clear_forces_recomputation() -> None:
    cache = TTLCache(ttl_seconds=60.0)
    calls = {"count": 0}

    def factory() -> int:
        calls["count"] += 1
        return calls["count"]

    cache.get_or_set("key", factory)
    cache.clear()
    result = cache.get_or_set("key", factory)

    assert result == 2
    assert calls["count"] == 2
