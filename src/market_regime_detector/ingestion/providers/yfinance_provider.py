from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from market_regime_detector.ingestion.providers.base import DataProvider
from market_regime_detector.ingestion.retry import with_retries
from market_regime_detector.ingestion.schema import to_naive_utc

FetchFn = Callable[[str, date, date], pd.DataFrame]


class YFinanceProvider(DataProvider):
    """Daily close prices via the unofficial `yfinance` scraper.

    Non-vintage (section 2.5: market series have no revisions), so `realtime_start`
    is set equal to the observation `date` and `asof` just drops later dates.
    `fetch_fn` is injectable so tests never hit the real yfinance/Yahoo endpoint —
    it defaults to a thin wrapper around `yfinance.Ticker(...).history(...)`.
    """

    source = "yfinance"

    def __init__(self, fetch_fn: FetchFn | None = None) -> None:
        self._fetch_fn = fetch_fn if fetch_fn is not None else self._default_fetch

    @staticmethod
    def _default_fetch(series_id: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        # yfinance's `end` is exclusive, unlike this provider's inclusive `end`.
        history: pd.DataFrame = yf.Ticker(series_id).history(
            start=start.isoformat(), end=(end + timedelta(days=1)).isoformat()
        )
        return history

    def fetch(
        self,
        series_id: str,
        start: date,
        end: date,
        asof: date | None = None,
    ) -> pd.DataFrame:
        raw = with_retries(lambda: self._fetch_fn(series_id, start, end))

        df = raw[["Close"]].rename(columns={"Close": "value"})
        df = df.rename_axis("date").reset_index()
        df["date"] = to_naive_utc(df["date"]).dt.normalize()
        df["realtime_start"] = df["date"]
        df["source"] = self.source
        df["series_id"] = series_id
        df["ingested_at"] = datetime.now(UTC)

        if asof is not None:
            df = df[df["date"] <= pd.Timestamp(asof)]

        return df
