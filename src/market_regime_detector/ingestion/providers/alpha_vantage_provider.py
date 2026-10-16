from datetime import UTC, date, datetime

import pandas as pd
import requests

from market_regime_detector.ingestion.providers.base import DataProvider
from market_regime_detector.ingestion.retry import with_retries

_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageProvider(DataProvider):
    """Daily close prices via the Alpha Vantage REST API.

    No first-pass feature (brief section 2.4) currently sources from Alpha Vantage —
    this exists because section 3.1 explicitly lists it as one of the four required
    concrete providers. Built and registered for interface completeness and as a
    future fallback/alternative data source; not dead code, just not wired to a
    series yet.

    Non-vintage, same as `YFinanceProvider`: `realtime_start` equals the observation
    `date`, and `asof` just drops later dates.
    """

    source = "alpha_vantage"

    def __init__(self, api_key: str, session: requests.Session | None = None) -> None:
        self._api_key = api_key
        self._session = session if session is not None else requests.Session()

    def fetch(
        self,
        series_id: str,
        start: date,
        end: date,
        asof: date | None = None,
    ) -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": series_id,
            "apikey": self._api_key,
            "outputsize": "full",
        }
        response = with_retries(lambda: self._session.get(_BASE_URL, params=params, timeout=30))
        response.raise_for_status()
        payload = response.json()

        time_series = payload.get("Time Series (Daily)")
        if time_series is None:
            raise ValueError(f"unexpected Alpha Vantage response for {series_id}: {payload}")

        records = [
            {"date": observation_date, "value": float(fields["4. close"])}
            for observation_date, fields in time_series.items()
        ]
        df = pd.DataFrame.from_records(records)
        df["date"] = pd.to_datetime(df["date"])

        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]

        df["realtime_start"] = df["date"]
        df["source"] = self.source
        df["series_id"] = series_id
        df["ingested_at"] = datetime.now(UTC)

        if asof is not None:
            df = df[df["date"] <= pd.Timestamp(asof)]

        return df
