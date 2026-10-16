from datetime import date

import pandas as pd
import pytest

from market_regime_detector.ingestion.providers.base import DataProvider
from market_regime_detector.ingestion.schema import CANONICAL_COLUMNS, validate_schema

SERIES_ID = "TESTSERIES"
START = date(2024, 1, 1)
END = date(2024, 3, 1)

PROVIDER_FIXTURES = [
    "fred_provider",
    "alfred_provider",
    "yfinance_provider",
    "alpha_vantage_provider",
]


@pytest.mark.parametrize("provider_fixture", PROVIDER_FIXTURES)
def test_provider_returns_canonical_schema(
    provider_fixture: str, request: pytest.FixtureRequest
) -> None:
    provider: DataProvider = request.getfixturevalue(provider_fixture)

    df = provider.fetch(SERIES_ID, START, END)
    validated = validate_schema(df)  # must not raise — columns/dtypes are coercible

    assert set(CANONICAL_COLUMNS).issubset(df.columns)
    assert not validated.empty
    assert (validated["series_id"] == SERIES_ID).all()
    assert (validated["source"] == provider.source).all()


@pytest.mark.parametrize("provider_fixture", PROVIDER_FIXTURES)
def test_provider_respects_asof(provider_fixture: str, request: pytest.FixtureRequest) -> None:
    provider: DataProvider = request.getfixturevalue(provider_fixture)

    df = provider.fetch(SERIES_ID, START, END, asof=START)
    validated = validate_schema(df)

    assert (validated["date"] <= pd.Timestamp(START)).all()
