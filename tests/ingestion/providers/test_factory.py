import pytest

from market_regime_detector.ingestion.providers.alfred_provider import AlfredProvider
from market_regime_detector.ingestion.providers.alpha_vantage_provider import (
    AlphaVantageProvider,
)
from market_regime_detector.ingestion.providers.factory import build_provider
from market_regime_detector.ingestion.providers.fred_provider import FredProvider
from market_regime_detector.ingestion.providers.yfinance_provider import YFinanceProvider


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("fred", FredProvider),
        ("alfred", AlfredProvider),
        ("yfinance", YFinanceProvider),
        ("alpha_vantage", AlphaVantageProvider),
    ],
)
def test_build_provider_dispatches_by_name(name: str, expected_type: type) -> None:
    assert isinstance(build_provider(name), expected_type)


def test_build_provider_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        build_provider("not_a_real_provider")
