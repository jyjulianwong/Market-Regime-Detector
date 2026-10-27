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
    # HMM_ACCURACY_PLAN.md Phase 4 data-availability spike: candidates confirmed
    # to have long history (no meaningful truncation vs. the current ~2000+
    # training window), no label circularity, and (where the per-date vintage
    # check below found real revisions) real ALFRED vintage history. Ingested
    # now so the lake has history to train against once Phase 5 actually wires
    # one into FeatureConfig — not yet referenced by any FeatureDefinition, same
    # "ingest now, wire later" pattern as AlphaVantageProvider (see
    # ingestion/alpha_vantage_provider.py). `source="alfred"` below is backed by
    # a real per-date revision check (ALFRED has 0 dates for NFCI; MICH/GASREGW
    # are revised on <1% of dates, so "fred" is simpler and loses nothing).
    SeriesSpec("UNRATE", "alfred", "growth"),
    SeriesSpec("PERMIT", "alfred", "growth"),
    SeriesSpec("RSAFS", "alfred", "growth"),
    SeriesSpec("RRSFS", "alfred", "growth"),
    SeriesSpec("NFCI", "fred", "growth"),
    SeriesSpec("STLFSI4", "alfred", "growth"),
    SeriesSpec("GACDISA066MSFRBNY", "alfred", "growth"),  # Empire State survey
    SeriesSpec("GACDFSA066MSFRBPHI", "alfred", "growth"),  # Philly Fed survey
    SeriesSpec("SPY", "yfinance", "growth"),
    SeriesSpec("IWM", "yfinance", "growth"),
    SeriesSpec("XLY", "yfinance", "growth"),
    SeriesSpec("XLP", "yfinance", "growth"),
    SeriesSpec("XLI", "yfinance", "growth"),
    SeriesSpec("XLU", "yfinance", "growth"),
    SeriesSpec("PCEPI", "alfred", "inflation"),
    SeriesSpec("PCEPILFE", "alfred", "inflation"),
    SeriesSpec("T5YIFR", "fred", "inflation"),
    SeriesSpec("MEDCPIM158SFRBCLE", "alfred", "inflation"),
    SeriesSpec("TRMMEANCPIM158SFRBCLE", "alfred", "inflation"),
    SeriesSpec("CES0500000003", "alfred", "inflation"),
    SeriesSpec("MICH", "fred", "inflation"),
    SeriesSpec("DTWEXBGS", "fred", "inflation"),
    SeriesSpec("GASREGW", "fred", "inflation"),
    # HMM_ACCURACY_PLAN.md Phase 5 "daily market" group candidates. These two
    # were checked for availability during the Phase 4 spike (see
    # phase4_data_availability.py's FRED_CANDIDATES) but the decision table
    # committed in that phase omitted them — a gap closed here before building
    # features on them: both have long FRED history (1990+), no vintage
    # concept (market-derived), and no circularity risk.
    SeriesSpec("VIXCLS", "fred", "growth"),
    SeriesSpec("DGS2", "fred", "growth"),
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
