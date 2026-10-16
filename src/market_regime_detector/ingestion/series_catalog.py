from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    source: str
    axis: str  # "growth" | "inflation" — informational only in Phase 2


# First-pass series, brief section 2.4. Feature-level concerns (transform, lag,
# sign, publication_lag_days) are Phase 3's `FeatureConfig`, not this catalog —
# this is only what `ingest_data.py` needs to pull raw observations into the lake.
FIRST_PASS_SERIES = [
    SeriesSpec("INDPRO", "alfred", "growth"),
    SeriesSpec("PAYEMS", "alfred", "growth"),
    SeriesSpec("ICSA", "alfred", "growth"),
    SeriesSpec("T10Y2Y", "fred", "growth"),
    SeriesSpec("HG=F", "yfinance", "growth"),
    SeriesSpec("GC=F", "yfinance", "growth"),
    SeriesSpec("CPILFESL", "alfred", "inflation"),
    SeriesSpec("T10YIE", "fred", "inflation"),
    SeriesSpec("DFII10", "fred", "inflation"),
    SeriesSpec("DCOILWTICO", "fred", "inflation"),
]
