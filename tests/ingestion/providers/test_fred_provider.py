from datetime import date

import pandas as pd

from market_regime_detector.ingestion.providers.fred_provider import FredProvider

START = date(2024, 1, 1)
END = date(2024, 3, 1)


def test_realtime_start_equals_observation_date(fred_provider: FredProvider) -> None:
    df = fred_provider.fetch("TESTSERIES", START, END)

    pd.testing.assert_series_equal(df["realtime_start"], df["date"], check_names=False)


def test_asof_drops_observations_after_asof(fred_provider: FredProvider) -> None:
    asof = date(2024, 1, 15)

    df = fred_provider.fetch("TESTSERIES", START, END, asof=asof)

    assert (df["date"] <= pd.Timestamp(asof)).all()
    assert len(df) == 1
