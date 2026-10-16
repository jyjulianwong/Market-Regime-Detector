from datetime import date

import pytest

from market_regime_detector.ingestion.providers.alpha_vantage_provider import (
    AlphaVantageProvider,
)

START = date(2024, 1, 1)
END = date(2024, 3, 1)


def test_parses_close_field(alpha_vantage_provider: AlphaVantageProvider) -> None:
    df = alpha_vantage_provider.fetch("TESTSYMBOL", START, END)

    assert sorted(df["value"]) == [1.0, 2.0]


class _ErrorResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return {"Note": "rate limit exceeded"}


class _ErrorSession:
    def get(self, url: str, params: dict[str, str], timeout: int) -> _ErrorResponse:
        return _ErrorResponse()


def test_raises_on_unexpected_response_shape() -> None:
    provider = AlphaVantageProvider(api_key="unused", session=_ErrorSession())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unexpected Alpha Vantage response"):
        provider.fetch("TESTSYMBOL", START, END)
