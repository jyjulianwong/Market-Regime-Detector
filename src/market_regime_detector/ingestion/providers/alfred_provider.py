from datetime import UTC, date, datetime

import pandas as pd
from fredapi import Fred

from market_regime_detector.ingestion.providers.base import DataProvider
from market_regime_detector.ingestion.retry import with_retries


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
            df = with_retries(lambda: self._client.get_series_all_releases(series_id))
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
