from market_regime_detector.common.settings import get_settings
from market_regime_detector.ingestion.providers.alfred_provider import AlfredProvider
from market_regime_detector.ingestion.providers.alpha_vantage_provider import (
    AlphaVantageProvider,
)
from market_regime_detector.ingestion.providers.base import DataProvider
from market_regime_detector.ingestion.providers.fred_provider import FredProvider
from market_regime_detector.ingestion.providers.yfinance_provider import YFinanceProvider

PROVIDER_NAMES = ("fred", "alfred", "yfinance", "alpha_vantage")


def build_provider(name: str) -> DataProvider:
    """Constructs a concrete `DataProvider` by name, reading API keys from settings.

    Mirrors the `EmailProvider`/`WebSearchProvider` factory pattern used elsewhere
    (build-by-name, settings-driven construction) so callers never import a concrete
    provider class directly — only `ingest.py` and scripts should call this.
    """
    settings = get_settings()

    if name == "fred":
        return FredProvider(api_key=settings.fred_api_key)
    if name == "alfred":
        return AlfredProvider(api_key=settings.fred_api_key)
    if name == "yfinance":
        return YFinanceProvider()
    if name == "alpha_vantage":
        return AlphaVantageProvider(api_key=settings.alpha_vantage_api_key)

    raise ValueError(f"unknown provider {name!r}; expected one of {PROVIDER_NAMES}")
