from datetime import UTC, date, datetime

import pandas as pd
from fredapi import Fred

from market_regime_detector.ingestion.providers.base import DataProvider
from market_regime_detector.ingestion.retry import with_retries


class FredProvider(DataProvider):
    """Current-value (non-vintage) FRED series.

    FRED's plain `/series/observations` endpoint only ever returns the latest known
    value per date — no revision history. That's fine for the market series this
    provider is meant for (section 2.5: daily yields/spreads/oil prices aren't
    revised), but it means `asof` here only enforces "don't return observations dated
    after `asof`" — it cannot reconstruct what was actually known on a past date for
    a series that *does* get revised. Use `AlfredProvider` for those.
    """

    source = "fred"

    def __init__(self, api_key: str, client: Fred | None = None) -> None:
        self._client = client if client is not None else Fred(api_key=api_key)

    def fetch(
        self,
        series_id: str,
        start: date,
        end: date,
        asof: date | None = None,
    ) -> pd.DataFrame:
        series = with_retries(
            lambda: self._client.get_series(
                series_id,
                observation_start=start.isoformat(),
                observation_end=end.isoformat(),
            )
        )

        df: pd.DataFrame = series.rename("value").rename_axis("date").reset_index()
        df["realtime_start"] = df["date"]
        df["source"] = self.source
        df["series_id"] = series_id
        df["ingested_at"] = datetime.now(UTC)

        if asof is not None:
            df = df[df["date"] <= pd.Timestamp(asof)]

        return df
