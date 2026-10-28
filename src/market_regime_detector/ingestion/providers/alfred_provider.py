from datetime import UTC, date, datetime, timedelta

import pandas as pd
from fredapi import Fred

from market_regime_detector.ingestion.providers.base import DataProvider
from market_regime_detector.ingestion.retry import with_retries

# ALFRED's `series/observations` endpoint caps a single request at 100,000 rows
# (its documented max `limit`) and `fredapi.get_series_all_releases` never
# paginates — it issues one unbounded request and returns whatever comes back,
# with no signal that the result was truncated. A heavily-revised weekly
# composite index (e.g. `STLFSI4`, confirmed median ~200 vintages/date) blows
# past that in a single call; the bug went undetected until Phase 5's ablation
# produced NaN metrics traced back to a lake partition silently missing most
# of its history. Fixed by bisecting the realtime-window until each sub-query
# is under the cap, not by working around the symptom.
_ALFRED_ROW_LIMIT = 100_000
# ALFRED's revision tracking realistically begins in the early 1990s (its own
# published history starts then); bounding bisection to this range instead of
# fredapi's `1776-07-04`/`9999-12-31` sentinels avoids dozens of pointless
# empty-range network calls while bisecting. If this assumption is ever wrong
# for some series, `_fetch_all_releases_paginated` raises rather than silently
# dropping data at the boundary.
_ALFRED_REALTIME_FLOOR = date(1990, 1, 1)


def _fetch_all_releases_paginated(
    client: Fred, series_id: str, window_start: date, window_end: date
) -> pd.DataFrame:
    df: pd.DataFrame = with_retries(
        lambda: client.get_series_all_releases(
            series_id,
            realtime_start=window_start.isoformat(),
            realtime_end=window_end.isoformat(),
        )
    )
    if len(df) < _ALFRED_ROW_LIMIT:
        return df
    if window_start >= window_end:
        raise RuntimeError(
            f"{series_id}: ALFRED returned {_ALFRED_ROW_LIMIT} rows for a "
            "single-day realtime window; cannot paginate further"
        )
    midpoint = window_start + timedelta(days=(window_end - window_start).days // 2)
    left = _fetch_all_releases_paginated(client, series_id, window_start, midpoint)
    right = _fetch_all_releases_paginated(
        client, series_id, midpoint + timedelta(days=1), window_end
    )
    return pd.concat([left, right], ignore_index=True)


class AlfredProvider(DataProvider):
    """Vintage-aware ALFRED series — tracks every historical revision.

    With `asof=None`, returns the *entire* revision history (`fredapi`'s
    `get_series_all_releases`: one row per `(date, realtime_start)` pair, i.e. every
    vintage ever published for every observation date) — this is what the lake needs
    to store, so training (Phase 4) can reconstruct "what was known as of any past
    decision date" without re-fetching from ALFRED.

    With `asof` set, uses ALFRED's native as-of-date snapshot query
    (`realtime_start=realtime_end=asof`) instead of filtering the full-release frame
    client-side — this returns exactly the single vintage of each observation that
    was the latest one published by `asof`.
    """

    source = "alfred"

    def __init__(self, api_key: str, client: Fred | None = None) -> None:
        self._client = client if client is not None else Fred(api_key=api_key)

    def fetch(
        self,
        series_id: str,
        start: date,
        end: date,
        asof: date | None = None,
    ) -> pd.DataFrame:
        df: pd.DataFrame
        if asof is None:
            # ALFRED rejects `realtime_end` after *its own* server-side "today",
            # which can be a day behind this machine's clock (observed: a
            # `ValueError` from a real request made while this machine's UTC
            # date had already rolled over). A one-day buffer avoids that
            # ambiguity; it only means the single most recent day's revisions
            # (if any were published literally today) are picked up by the
            # *next* ingest run instead, never a look-ahead risk.
            ceiling = date.today() - timedelta(days=1)
            df = _fetch_all_releases_paginated(
                self._client, series_id, _ALFRED_REALTIME_FLOOR, ceiling
            )
            df = df.copy()
            start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
            df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        else:
            asof_str = asof.isoformat()
            series = with_retries(
                lambda: self._client.get_series(
                    series_id,
                    observation_start=start.isoformat(),
                    observation_end=end.isoformat(),
                    realtime_start=asof_str,
                    realtime_end=asof_str,
                )
            )
            df = series.rename("value").rename_axis("date").reset_index()
            df["realtime_start"] = pd.Timestamp(asof)

        df["source"] = self.source
        df["series_id"] = series_id
        df["ingested_at"] = datetime.now(UTC)
        return df
