from datetime import UTC, datetime

import pandas as pd
import pytest

from market_regime_detector.ingestion.schema import CANONICAL_COLUMNS, validate_schema


def _raw_df(**overrides: object) -> pd.DataFrame:
    base = {
        "date": ["2024-01-01", "2024-02-01"],
        "value": [1.0, 2.0],
        "realtime_start": ["2024-01-01", "2024-02-01"],
        "source": ["fred", "fred"],
        "series_id": ["TEST", "TEST"],
        "ingested_at": [datetime.now(UTC), datetime.now(UTC)],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_validate_schema_returns_only_canonical_columns_in_order() -> None:
    df = _raw_df()
    df["extra_column"] = "ignored"

    result = validate_schema(df)

    assert list(result.columns) == CANONICAL_COLUMNS


def test_validate_schema_raises_on_missing_column() -> None:
    df = _raw_df().drop(columns=["value"])

    with pytest.raises(ValueError, match="value"):
        validate_schema(df)


def test_validate_schema_deduplicates_natural_key_keeping_last() -> None:
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "value": [1.0, 2.0],
            "realtime_start": ["2024-01-01", "2024-01-01"],
            "source": ["fred", "fred"],
            "series_id": ["TEST", "TEST"],
            "ingested_at": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
            ],
        }
    )

    result = validate_schema(df)

    assert len(result) == 1
    assert result.loc[0, "value"] == 2.0


def test_validate_schema_strips_timezone_from_tz_aware_timestamps() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]).tz_localize("America/New_York"),
            "value": [1.0],
            "realtime_start": pd.to_datetime(["2024-01-01"]).tz_localize("America/New_York"),
            "source": ["fred"],
            "series_id": ["TEST"],
            "ingested_at": [datetime.now(UTC)],
        }
    )

    result = validate_schema(df)

    assert result["date"].dt.tz is None
    assert result["ingested_at"].dt.tz is None


def test_validate_schema_coerces_unparseable_value_to_nan() -> None:
    # Real fredapi quirk: get_series_all_releases occasionally parses ALFRED's "."
    # (withdrawn/missing revision) sentinel as NaT instead of NaN (seen on ICSA) —
    # .astype(float) raises on that; this must coerce it to a missing value instead.
    df = _raw_df(value=[pd.NaT, 2.0])

    result = validate_schema(df)

    assert pd.isna(result.loc[0, "value"])
    assert result.loc[1, "value"] == 2.0


def test_validate_schema_is_idempotent() -> None:
    df = _raw_df()

    once = validate_schema(df)
    twice = validate_schema(once)

    pd.testing.assert_frame_equal(once, twice)
