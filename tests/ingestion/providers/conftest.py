from datetime import date

import pandas as pd
import pytest

from market_regime_detector.ingestion.providers.alfred_provider import AlfredProvider
from market_regime_detector.ingestion.providers.alpha_vantage_provider import (
    AlphaVantageProvider,
)
from market_regime_detector.ingestion.providers.fred_provider import FredProvider
from market_regime_detector.ingestion.providers.yfinance_provider import YFinanceProvider

SERIES_ID = "TESTSERIES"
START = date(2024, 1, 1)
END = date(2024, 3, 1)


class FakeFredClient:
    """Stands in for `fredapi.Fred` — no network, deterministic responses."""

    def get_series(
        self,
        series_id: str,
        observation_start: str,
        observation_end: str,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
    ) -> pd.Series:
        index = pd.to_datetime(["2024-01-01", "2024-02-01"])
        series = pd.Series([1.0, 2.0], index=index)
        if realtime_end is not None:
            # Real ALFRED: an observation date's data isn't published until after
            # the period it covers ends, so bounding by realtime_end naturally
            # bounds observation date too — the fake replicates that for asof tests.
            series = series[series.index <= pd.Timestamp(realtime_end)]
        return series

    def get_series_all_releases(self, series_id: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01"]),
                "realtime_start": pd.to_datetime(["2024-01-15", "2024-02-15", "2024-02-15"]),
                "value": [0.9, 1.0, 2.0],
            }
        )


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return self._payload


class FakeSession:
    def get(self, url: str, params: dict[str, str], timeout: int) -> FakeResponse:
        return FakeResponse(
            {
                "Time Series (Daily)": {
                    "2024-01-01": {"4. close": "1.0"},
                    "2024-02-01": {"4. close": "2.0"},
                }
            }
        )


def fake_yfinance_history(series_id: str, start: date, end: date) -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-01", "2024-02-01"])
    return pd.DataFrame({"Close": [1.0, 2.0]}, index=index)


@pytest.fixture
def fred_provider() -> FredProvider:
    return FredProvider(api_key="unused", client=FakeFredClient())


@pytest.fixture
def alfred_provider() -> AlfredProvider:
    return AlfredProvider(api_key="unused", client=FakeFredClient())


@pytest.fixture
def yfinance_provider() -> YFinanceProvider:
    return YFinanceProvider(fetch_fn=fake_yfinance_history)


@pytest.fixture
def alpha_vantage_provider() -> AlphaVantageProvider:
    return AlphaVantageProvider(api_key="unused", session=FakeSession())  # type: ignore[arg-type]
