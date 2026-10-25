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

# Evaluation-only: ground-truth label inputs for `training/labels.py`'s scoring
# harness, never fed to a model as a feature (that would be circular — these are
# "what actually happened," not point-in-time-knowable signal). Deliberately NOT
# in FIRST_PASS_SERIES — the weekly ingest Lambda (`prediction/handler.py`)
# iterates FIRST_PASS_SERIES directly, not this catalog module as a whole, and
# must not fetch these as part of that production schedule.
EVALUATION_SERIES = [
    SeriesSpec("CFNAIMA3", "fred", "growth"),  # Chicago Fed National Activity Index, 3-mo avg
    SeriesSpec("CPIAUCSL", "fred", "inflation"),  # headline CPI, SA
    SeriesSpec("USREC", "fred", "growth"),  # NBER recession indicator (0/1), monthly
]
