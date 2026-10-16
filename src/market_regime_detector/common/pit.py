import pandas as pd


def point_in_time_series(
    df: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    vintage_aware: bool,
    publication_lag_days: int,
) -> pd.Series:
    """Collapses a (possibly multi-vintage) canonical-schema DataFrame into a single
    value per observation `date`, reflecting only what was knowable as of `asof`.

    `df` must already be in the canonical ingestion schema (`date`, `value`,
    `realtime_start`, ...). Two causality mechanisms, picked per feature:

    - `vintage_aware=True` (ALFRED-sourced series, brief section 2.5): among rows
      with `realtime_start <= asof`, keep the one with the largest `realtime_start`
      for each `date` — the latest revision known as of `asof`.
    - `vintage_aware=False` (FRED/yfinance-sourced market series, no revisions):
      keep rows where `date + publication_lag_days <= asof`. For same-day market
      closes `publication_lag_days` is 0 — section 2.5's "no look-ahead alignment
      shortcuts" for market data.

    This is the single mechanism every no-look-ahead guarantee in this codebase rests
    on: mutating or adding rows dated/revised after `asof` must never change the
    result.
    """
    if vintage_aware:
        knowable = df[df["realtime_start"] <= asof]
        if knowable.empty:
            return pd.Series(dtype=float)
        latest = knowable.sort_values("realtime_start").groupby("date", as_index=False).tail(1)
        return latest.set_index("date")["value"].sort_index()

    # `Timedelta(publication_lag_days, unit="D")` rather than `Timedelta(days=...)` —
    # the latter triggers a spurious NumPy "generic unit" DeprecationWarning on the
    # pandas 2.3.x / NumPy 2.x combination this project pins.
    cutoff = asof - pd.Timedelta(publication_lag_days, unit="D")
    knowable = df[df["date"] <= cutoff]
    return knowable.set_index("date")["value"].sort_index()
