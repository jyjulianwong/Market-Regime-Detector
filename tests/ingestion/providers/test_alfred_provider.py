from datetime import date

import pandas as pd
import pytest

from market_regime_detector.ingestion.providers import alfred_provider as alfred_provider_module
from market_regime_detector.ingestion.providers.alfred_provider import (
    AlfredProvider,
    _fetch_all_releases_paginated,
)

START = date(2024, 1, 1)
END = date(2024, 3, 1)


class _PaginatingFakeClient:
    """Simulates ALFRED's real per-request row cap: returns only the rows
    whose `realtime_start` falls inside the queried window, so a caller that
    doesn't narrow the window when the cap is hit would silently see a subset.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self._full = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01"]),
                "realtime_start": pd.to_datetime(["2024-01-15", "2024-01-20", "2024-02-10"]),
                "value": [0.9, 1.0, 2.0],
            }
        )

    def get_series_all_releases(
        self, series_id: str, realtime_start: str, realtime_end: str
    ) -> pd.DataFrame:
        self.calls.append((realtime_start, realtime_end))
        start, end = pd.Timestamp(realtime_start), pd.Timestamp(realtime_end)
        mask = (self._full["realtime_start"] >= start) & (self._full["realtime_start"] <= end)
        return self._full[mask].reset_index(drop=True)


class _AlwaysAtCapFakeClient:
    """A pathological fake where even a single-day realtime window still
    returns at-or-over the row cap — exercises the "cannot paginate further"
    hard failure rather than an infinite recursion or silent truncation.
    """

    def get_series_all_releases(
        self, series_id: str, realtime_start: str, realtime_end: str
    ) -> pd.DataFrame:
        return pd.DataFrame({"date": [pd.Timestamp("2024-01-01")] * 2, "value": [1.0, 2.0]})


def test_full_history_returns_every_vintage_per_date(alfred_provider: AlfredProvider) -> None:
    df = alfred_provider.fetch("TESTSERIES", START, END)

    # The fake client's get_series_all_releases returns two vintages for 2024-01-01.
    same_date_rows = df[df["date"] == pd.Timestamp("2024-01-01")]
    assert len(same_date_rows) == 2
    assert set(same_date_rows["realtime_start"]) == {
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-02-15"),
    }


def test_asof_queries_single_snapshot_and_tags_realtime_start(
    alfred_provider: AlfredProvider,
) -> None:
    asof = date(2024, 1, 15)

    df = alfred_provider.fetch("TESTSERIES", START, END, asof=asof)

    assert (df["realtime_start"] == pd.Timestamp(asof)).all()


def test_paginated_fetch_bisects_window_until_under_the_row_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real bug: ALFRED caps a single `series/observations`
    request at 100,000 rows and never signals truncation — a heavily-revised
    series silently loses history unless the caller detects the cap and
    narrows the window. `_ALFRED_ROW_LIMIT` is lowered here so a 3-row fake
    dataset can exercise the same bisection logic without constructing a
    real 100,000-row fixture.
    """
    monkeypatch.setattr(alfred_provider_module, "_ALFRED_ROW_LIMIT", 3)
    client = _PaginatingFakeClient()

    df = _fetch_all_releases_paginated(client, "TESTSERIES", date(2024, 1, 1), date(2024, 3, 1))

    assert len(df) == 3
    assert set(df["realtime_start"]) == {
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-01-20"),
        pd.Timestamp("2024-02-10"),
    }
    # The full window hit the (lowered) cap and had to be split at least once —
    # a single unbounded call, like the pre-fix behavior, would have returned
    # all 3 rows without ever revealing that a real 100,000-row cap had
    # silently dropped anything.
    assert len(client.calls) > 1


def test_paginated_fetch_raises_when_a_single_day_window_still_hits_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(alfred_provider_module, "_ALFRED_ROW_LIMIT", 1)
    client = _AlwaysAtCapFakeClient()

    with pytest.raises(RuntimeError, match="cannot paginate further"):
        _fetch_all_releases_paginated(client, "TESTSERIES", date(2024, 1, 1), date(2024, 1, 1))
