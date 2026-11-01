from datetime import UTC, datetime

import pandas as pd

from market_regime_detector.common.pit import latest_knowable_observation, point_in_time_series


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


def test_latest_knowable_observation_vintage_aware_picks_latest_date_and_vintage() -> None:
    df = pd.DataFrame(
        [
            _row("2024-01-01", 100.0, "2024-01-15"),
            _row("2024-02-01", 200.0, "2024-02-15"),  # a later date, also knowable
        ]
    )

    obs = latest_knowable_observation(
        df, pd.Timestamp("2024-03-01"), vintage_aware=True, publication_lag_days=0
    )

    assert obs is not None
    assert obs.date == pd.Timestamp("2024-02-01")
    assert obs.value == 200.0
    assert obs.realtime_start == pd.Timestamp("2024-02-15")


def test_latest_knowable_observation_ignores_a_revision_not_yet_knowable() -> None:
    df = pd.DataFrame(
        [
            _row("2024-01-01", 100.0, "2024-01-15"),
            _row("2024-01-01", 999.0, "2024-03-01"),  # future revision of the same date
        ]
    )

    obs = latest_knowable_observation(
        df, pd.Timestamp("2024-02-01"), vintage_aware=True, publication_lag_days=0
    )

    assert obs is not None
    assert obs.value == 100.0  # March revision not yet known
    assert obs.realtime_start == pd.Timestamp("2024-01-15")


def test_latest_knowable_observation_non_vintage_applies_publication_lag() -> None:
    df = pd.DataFrame([_row("2024-01-10", 1.0, "2024-01-10")])

    too_early = latest_knowable_observation(
        df, pd.Timestamp("2024-01-12"), vintage_aware=False, publication_lag_days=5
    )
    late_enough = latest_knowable_observation(
        df, pd.Timestamp("2024-01-16"), vintage_aware=False, publication_lag_days=5
    )

    assert too_early is None
    assert late_enough is not None
    assert late_enough.value == 1.0
    assert late_enough.realtime_start == pd.Timestamp("2024-01-10")


def test_latest_knowable_observation_none_when_nothing_knowable_yet() -> None:
    df = pd.DataFrame([_row("2024-06-01", 1.0, "2024-06-01")])

    obs = latest_knowable_observation(
        df, pd.Timestamp("2024-01-01"), vintage_aware=True, publication_lag_days=0
    )

    assert obs is None


def test_latest_knowable_observation_skips_null_values() -> None:
    df = pd.DataFrame(
        [
            _row("2024-01-01", 1.0, "2024-01-01"),
            {**_row("2024-02-01", float("nan"), "2024-02-01")},
        ]
    )

    obs = latest_knowable_observation(
        df, pd.Timestamp("2024-03-01"), vintage_aware=False, publication_lag_days=0
    )

    assert obs is not None
    assert obs.date == pd.Timestamp("2024-01-01")
    assert obs.value == 1.0
