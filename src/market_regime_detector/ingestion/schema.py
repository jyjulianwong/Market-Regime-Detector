import pandas as pd

CANONICAL_COLUMNS = ["date", "value", "realtime_start", "source", "series_id", "ingested_at"]

NATURAL_KEY = ["series_id", "date", "realtime_start"]


def to_naive_utc(series: pd.Series) -> pd.Series:
    """Parses a column to `datetime64[ns]`, converting any tz-aware values to UTC
    and then stripping the tz label — `.dt.tz_localize(None)` raises on already-naive
    input, so this only converts when there's actually a tz to convert away.

    Shared with providers that need to compare a possibly tz-aware timestamp column
    (e.g. yfinance's tz-aware index) against a tz-naive `asof` date before the row
    ever reaches `validate_schema`.
    """
    parsed = pd.to_datetime(series)
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Coerces a provider's raw DataFrame into the canonical ingestion schema.

    One row per observation: `date`, `value`, `realtime_start` (vintage start; equals
    `date` for non-vintage sources), `source`, `series_id`, `ingested_at`. `date` and
    `realtime_start` are calendar dates (midnight-normalized, tz-naive); `ingested_at`
    is a UTC instant with tzinfo stripped after conversion, so every timestamp column
    round-trips through Parquet as plain `datetime64[ns]` with no tz-aware/naive
    mixing. Rows are de-duplicated on the natural key `(series_id, date,
    realtime_start)`, keeping the most recently ingested row for each key.

    Raises `ValueError` if a required column is missing — this is the only schema
    validation ingestion does; feature-level transforms belong in `common/`, not here.
    """
    missing = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df = df[CANONICAL_COLUMNS].copy()

    df["date"] = to_naive_utc(df["date"]).dt.normalize()
    df["realtime_start"] = to_naive_utc(df["realtime_start"]).dt.normalize()
    # pd.to_numeric, not .astype(float): a real ALFRED/fredapi quirk, not a bug in
    # this codebase — fredapi's get_series_all_releases occasionally parses a
    # withdrawn/missing revision's "." sentinel as NaT rather than NaN (seen on
    # ICSA), which .astype(float) raises on but to_numeric(errors="coerce")
    # correctly treats as a missing value for that vintage, matching the semantics
    # of a genuinely unavailable observation.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["source"] = df["source"].astype(str)
    df["series_id"] = df["series_id"].astype(str)
    df["ingested_at"] = to_naive_utc(df["ingested_at"])

    df = df.drop_duplicates(subset=NATURAL_KEY, keep="last")
    df = df.sort_values(NATURAL_KEY).reset_index(drop=True)
    return df
