from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    """Fetches raw observations for a single series from one external source.

    Implementations return the canonical raw schema (see
    `market_regime_detector.ingestion.schema.CANONICAL_COLUMNS`) but do not
    themselves call `validate_schema` — callers (the ingestion orchestration in
    `ingest.py`) own that step, keeping providers focused purely on "talk to the
    external API." Providers never write to S3 themselves (brief section 3.1).
    """

    source: str

    @abstractmethod
    def fetch(
        self,
        series_id: str,
        start: date,
        end: date,
        asof: date | None = None,
    ) -> pd.DataFrame:
        """Returns raw observations for `series_id` in `[start, end]`.

        If `asof` is given, the result must only contain what was knowable as of that
        date — for a vintage-aware source that means filtering to the state of
        revisions published by `asof`; for a non-vintage source it means dropping any
        observation dated after `asof`. If `asof` is `None`, return everything
        available in range (for a vintage-aware source, every historical revision).
        """
        raise NotImplementedError
