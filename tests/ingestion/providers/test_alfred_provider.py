from datetime import date

import pandas as pd

from market_regime_detector.ingestion.providers.alfred_provider import AlfredProvider

START = date(2024, 1, 1)
END = date(2024, 3, 1)


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
