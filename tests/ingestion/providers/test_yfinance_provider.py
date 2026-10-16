from datetime import date

import pandas as pd

from market_regime_detector.ingestion.providers.yfinance_provider import YFinanceProvider

START = date(2024, 1, 1)
END = date(2024, 3, 1)


def _fetch_fn(series_id: str, start: date, end: date) -> pd.DataFrame:
    index = pd.to_datetime(["2024-01-01", "2024-02-01"])
    return pd.DataFrame({"Close": [1.5, 2.5]}, index=index)


def test_close_column_renamed_to_value() -> None:
    provider = YFinanceProvider(fetch_fn=_fetch_fn)

    df = provider.fetch("GC=F", START, END)

    assert list(df["value"]) == [1.5, 2.5]
    assert "Close" not in df.columns


def test_ticker_containing_equals_sign_round_trips() -> None:
    """Futures tickers like `GC=F`/`HG=F` contain '=', the partition path's own
    separator — this just confirms the provider itself doesn't choke on it; the lake
    layer's partition-key handling is covered separately in test_lake.py.
    """
    provider = YFinanceProvider(fetch_fn=_fetch_fn)

    df = provider.fetch("GC=F", START, END)

    assert (df["series_id"] == "GC=F").all()
