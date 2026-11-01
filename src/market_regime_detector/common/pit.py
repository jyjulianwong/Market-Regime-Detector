from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LatestObservation:
    """One row out of `latest_knowable_observation` — unlike `point_in_time_series`
    (which collapses a whole series), this keeps `realtime_start` alongside the
    value, since the diagnostics endpoint (`prediction/diagnostics.py`) needs both
    "what's the latest known value" and "how long ago did we learn it" (lag) for
    the same row.
    """

    date: pd.Timestamp
    value: float
    realtime_start: pd.Timestamp


def latest_knowable_observation(
    df: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    vintage_aware: bool,
    publication_lag_days: int,
) -> LatestObservation | None:
    """The single most recent observation knowable as of `asof` — same two
    causality mechanisms as `point_in_time_series` (vintage-aware `realtime_start`
    filtering vs. a flat publication lag), but returning one row with its
    `realtime_start` intact instead of a full date-indexed series. Returns `None`
    if nothing is knowable yet, or every knowable row's `value` is null.
    """
    if vintage_aware:
        knowable = df[df["realtime_start"] <= asof].dropna(subset=["value"])
        if knowable.empty:
            return None
        row = knowable.sort_values(["date", "realtime_start"]).iloc[-1]
    else:
        # `Timedelta(publication_lag_days, unit="D")` rather than
        # `Timedelta(days=...)` — see `point_in_time_series` below.
        cutoff = asof - pd.Timedelta(publication_lag_days, unit="D")
        knowable = df[df["date"] <= cutoff].dropna(subset=["value"])
        if knowable.empty:
            return None
        row = knowable.sort_values("date").iloc[-1]

    return LatestObservation(
        date=row["date"], value=float(row["value"]), realtime_start=row["realtime_start"]
    )


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
