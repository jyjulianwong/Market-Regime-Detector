from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesSpec:
    series_id: str
    source: str
    axis: str  # "growth" | "inflation" — informational only in Phase 2
    # Human-readable description surfaced by the API's `/predictions/latest` `data`
    # diagnostic (see prediction/diagnostics.py) — every data stream the weekly
    # ingest Lambda maintains, independent of whether any FeatureConfig uses it.
    description: str


# First-pass series, brief section 2.4. Feature-level concerns (transform, lag,
# sign, publication_lag_days) are Phase 3's `FeatureConfig`, not this catalog —
# this is only what `ingest_data.py` needs to pull raw observations into the lake.
FIRST_PASS_SERIES = [
    SeriesSpec("INDPRO", "alfred", "growth", "Industrial Production Index"),
    SeriesSpec("PAYEMS", "alfred", "growth", "All Employees, Total Nonfarm (nonfarm payrolls)"),
    SeriesSpec(
        "ICSA", "alfred", "growth", "Initial Claims (unemployment insurance, seasonally adjusted)"
    ),
    SeriesSpec("T10Y2Y", "fred", "growth", "10-Year minus 2-Year Treasury yield spread"),
    SeriesSpec("HG=F", "yfinance", "growth", "Copper futures price"),
    SeriesSpec("GC=F", "yfinance", "growth", "Gold futures price"),
    SeriesSpec("CPILFESL", "alfred", "inflation", "Core CPI (CPI less food and energy)"),
    SeriesSpec("T10YIE", "fred", "inflation", "10-Year Breakeven Inflation Rate"),
    SeriesSpec("DFII10", "fred", "inflation", "10-Year Treasury Inflation-Indexed (real) yield"),
    SeriesSpec("DCOILWTICO", "fred", "inflation", "WTI crude oil spot price"),
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
    SeriesSpec("UNRATE", "alfred", "growth", "Civilian Unemployment Rate"),
    SeriesSpec(
        "PERMIT",
        "alfred",
        "growth",
        "New Private Housing Units Authorized by Building Permits",
    ),
    SeriesSpec("RSAFS", "alfred", "growth", "Advance Retail Sales"),
    SeriesSpec("RRSFS", "alfred", "growth", "Real Retail and Food Services Sales"),
    SeriesSpec("NFCI", "fred", "growth", "Chicago Fed National Financial Conditions Index"),
    # STLFSI4 was initially catalogued as "alfred" (Phase 4 found real ALFRED
    # revisions for it), but Phase 5 found its revision density is extreme —
    # 100,000+ rows even within a single 6-year `realtime_start` window
    # (likely revised on every near-daily re-estimate across its whole
    # history), making true full-history vintage pagination impractical.
    # Reclassified to non-vintage "fred", same treatment as NFCI.
    SeriesSpec("STLFSI4", "fred", "growth", "St. Louis Fed Financial Stress Index"),
    SeriesSpec(
        "GACDISA066MSFRBNY",
        "alfred",
        "growth",
        "Empire State Manufacturing Survey (general business conditions)",
    ),  # Empire State survey
    SeriesSpec(
        "GACDFSA066MSFRBPHI",
        "alfred",
        "growth",
        "Philadelphia Fed Manufacturing Survey (general business conditions)",
    ),  # Philly Fed survey
    SeriesSpec("SPY", "yfinance", "growth", "SPDR S&P 500 ETF price"),
    SeriesSpec("IWM", "yfinance", "growth", "iShares Russell 2000 ETF price"),
    SeriesSpec("XLY", "yfinance", "growth", "Consumer Discretionary Select Sector SPDR ETF price"),
    SeriesSpec("XLP", "yfinance", "growth", "Consumer Staples Select Sector SPDR ETF price"),
    SeriesSpec("XLI", "yfinance", "growth", "Industrial Select Sector SPDR ETF price"),
    SeriesSpec("XLU", "yfinance", "growth", "Utilities Select Sector SPDR ETF price"),
    SeriesSpec("PCEPI", "alfred", "inflation", "Personal Consumption Expenditures Price Index"),
    SeriesSpec(
        "PCEPILFE",
        "alfred",
        "inflation",
        "Core PCE Price Index (PCE less food and energy)",
    ),
    SeriesSpec(
        "T5YIFR",
        "fred",
        "inflation",
        "5-Year, 5-Year Forward Inflation Expectation Rate",
    ),
    SeriesSpec("MEDCPIM158SFRBCLE", "alfred", "inflation", "Median CPI, monthly (Cleveland Fed)"),
    SeriesSpec(
        "TRMMEANCPIM158SFRBCLE",
        "alfred",
        "inflation",
        "16% Trimmed-Mean CPI, monthly (Cleveland Fed)",
    ),
    SeriesSpec("CES0500000003", "alfred", "inflation", "Average Hourly Earnings, Total Private"),
    SeriesSpec("MICH", "fred", "inflation", "University of Michigan Inflation Expectation"),
    SeriesSpec("DTWEXBGS", "fred", "inflation", "Trade Weighted U.S. Dollar Index (broad)"),
    SeriesSpec("GASREGW", "fred", "inflation", "US Regular All Formulations Gas Price"),
    # HMM_ACCURACY_PLAN.md Phase 5 "daily market" group candidates. These two
    # were checked for availability during the Phase 4 spike (see
    # phase4_data_availability.py's FRED_CANDIDATES) but the decision table
    # committed in that phase omitted them — a gap closed here before building
    # features on them: both have long FRED history (1990+), no vintage
    # concept (market-derived), and no circularity risk.
    SeriesSpec("VIXCLS", "fred", "growth", "CBOE Volatility Index (VIX)"),
    SeriesSpec("DGS2", "fred", "growth", "2-Year Treasury Constant Maturity Rate"),
]

# Evaluation-only: ground-truth label inputs for `training/labels.py`'s scoring
# harness, never fed to a model as a feature (that would be circular — these are
# "what actually happened," not point-in-time-knowable signal). Deliberately NOT
# in FIRST_PASS_SERIES — the daily ingest Lambda (`prediction/handler.py`)
# iterates FIRST_PASS_SERIES directly, not this catalog module as a whole, and
# must not fetch these as part of that production schedule.
EVALUATION_SERIES = [
    SeriesSpec(
        "CFNAIMA3",
        "fred",
        "growth",
        "Chicago Fed National Activity Index, 3-month moving average",
    ),
    SeriesSpec(
        "CPIAUCSL",
        "fred",
        "inflation",
        "Consumer Price Index for All Urban Consumers (headline CPI)",
    ),  # headline CPI, SA
    SeriesSpec("USREC", "fred", "growth", "NBER-based Recession Indicator"),  # 0/1, monthly
]
