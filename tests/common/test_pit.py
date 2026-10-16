from datetime import UTC, datetime

import pandas as pd

from market_regime_detector.common.pit import point_in_time_series


def _row(date: str, value: float, realtime_start: str) -> dict[str, object]:
    return {
        "date": pd.Timestamp(date),
        "value": value,
        "realtime_start": pd.Timestamp(realtime_start),
        "source": "alfred",
        "series_id": "TEST",
        "ingested_at": datetime.now(UTC),
    }


def test_vintage_aware_picks_latest_vintage_at_or_before_asof() -> None:
    df = pd.DataFrame(
        [
            _row("2024-01-01", 100.0, "2024-01-15"),
            _row("2024-01-01", 101.0, "2024-02-15"),  # a later revision
        ]
    )

    result = point_in_time_series(
        df, pd.Timestamp("2024-02-01"), vintage_aware=True, publication_lag_days=0
    )

    assert result[pd.Timestamp("2024-01-01")] == 100.0  # Feb revision not yet known


def test_vintage_aware_no_look_ahead_on_added_future_revision() -> None:
    """The defining property test: a feature at date t must be unchanged if data
    published after t is altered (brief section 3.8).
    """
    asof = pd.Timestamp("2024-02-01")
    df_before = pd.DataFrame([_row("2024-01-01", 100.0, "2024-01-15")])
    df_after = pd.concat(
        [df_before, pd.DataFrame([_row("2024-01-01", 999.0, "2024-03-01")])],
        ignore_index=True,
    )

    before = point_in_time_series(df_before, asof, vintage_aware=True, publication_lag_days=0)
    after = point_in_time_series(df_after, asof, vintage_aware=True, publication_lag_days=0)

    pd.testing.assert_series_equal(before, after)


def test_vintage_aware_revision_becomes_visible_once_asof_passes_it() -> None:
    df = pd.DataFrame(
        [
            _row("2024-01-01", 100.0, "2024-01-15"),
            _row("2024-01-01", 999.0, "2024-03-01"),
        ]
    )

    before_revision = point_in_time_series(
        df, pd.Timestamp("2024-02-15"), vintage_aware=True, publication_lag_days=0
    )
    after_revision = point_in_time_series(
        df, pd.Timestamp("2024-03-15"), vintage_aware=True, publication_lag_days=0
    )

    assert before_revision[pd.Timestamp("2024-01-01")] == 100.0
    assert after_revision[pd.Timestamp("2024-01-01")] == 999.0


def test_non_vintage_applies_publication_lag() -> None:
    df = pd.DataFrame([_row("2024-01-10", 1.0, "2024-01-10")])

    too_early = point_in_time_series(
        df, pd.Timestamp("2024-01-12"), vintage_aware=False, publication_lag_days=5
    )
    late_enough = point_in_time_series(
        df, pd.Timestamp("2024-01-16"), vintage_aware=False, publication_lag_days=5
    )

    assert too_early.empty
    assert late_enough[pd.Timestamp("2024-01-10")] == 1.0


def test_non_vintage_no_look_ahead_on_added_future_row() -> None:
    asof = pd.Timestamp("2024-01-20")
    df_before = pd.DataFrame([_row("2024-01-10", 1.0, "2024-01-10")])
    df_after = pd.concat(
        [df_before, pd.DataFrame([_row("2024-01-25", 999.0, "2024-01-25")])],
        ignore_index=True,
    )

    before = point_in_time_series(df_before, asof, vintage_aware=False, publication_lag_days=0)
    after = point_in_time_series(df_after, asof, vintage_aware=False, publication_lag_days=0)

    pd.testing.assert_series_equal(before, after)


def test_empty_when_nothing_knowable_yet() -> None:
    df = pd.DataFrame([_row("2024-06-01", 1.0, "2024-06-01")])

    result = point_in_time_series(
        df, pd.Timestamp("2024-01-01"), vintage_aware=True, publication_lag_days=0
    )

    assert result.empty
