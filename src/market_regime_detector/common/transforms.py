from collections.abc import Callable

import pandas as pd


def identity(series: pd.Series) -> pd.Series:
    """No-op transform — for the one feature (T10Y2Y) fed as a raw level rather than
    a rate of change (a deliberate exception — a yield spread is already a relative/
    directional quantity, not an absolute level like a price index).
    """
    return series


def pct_change(series: pd.Series, n_periods: int) -> pd.Series:
    result: pd.Series = series.pct_change(periods=n_periods)
    return result


def diff(series: pd.Series, n_periods: int) -> pd.Series:
    result: pd.Series = series.diff(periods=n_periods)
    return result


def pct_change_annualized(
    series: pd.Series, n_periods: int, periods_per_year: int = 12
) -> pd.Series:
    """Compounds an n-period change up to an annual rate, e.g. CPI's "3-month
    annualized change": `(value_t / value_{t-n}) ** (periods_per_year / n) - 1`.
    """
    ratio = series / series.shift(n_periods)
    result: pd.Series = ratio ** (periods_per_year / n_periods) - 1
    return result


def rolling_mean_diff(series: pd.Series, window: int, diff_periods: int = 1) -> pd.Series:
    """A rolling mean, then the period-over-period change of that mean — e.g. ICSA's
    "4-week average, change": smooths weekly noise before taking the change.
    """
    result: pd.Series = series.rolling(window=window).mean().diff(periods=diff_periods)
    return result


def detrend(series: pd.Series, window: int = 12) -> pd.Series:
    """Deviation from the trailing `window`-period rolling mean (`HMM_ACCURACY_PLAN.md`
    Phase 3's "surprise" feature family) — a slow-moving trend subtracted out,
    leaving how far the current value sits from its own recent history. Strictly
    backward-looking: `rolling(window)` only ever uses the current row and the
    `window - 1` rows before it, never a later one.
    """
    result: pd.Series = series - series.rolling(window=window).mean()
    return result


def second_diff(series: pd.Series, n_periods: int = 1) -> pd.Series:
    """The change of the change ("acceleration") — `diff` applied twice at the
    same `n_periods` lag, e.g. ICSA's week-over-week change in its own
    week-over-week change.
    """
    result: pd.Series = series.diff(periods=n_periods).diff(periods=n_periods)
    return result


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """A rolling (trailing-window) z-score — `(x - rolling_mean) / rolling_std`
    over `window` — never a feature's full-sample stats, which would use
    information not yet available at an early row (the full-sample mean/std
    keeps changing as more rows arrive).
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    result: pd.Series = (series - rolling_mean) / rolling_std
    return result


TRANSFORMS: dict[str, Callable[..., pd.Series]] = {
    "identity": identity,
    "pct_change": pct_change,
    "diff": diff,
    "pct_change_annualized": pct_change_annualized,
    "rolling_mean_diff": rolling_mean_diff,
    "detrend": detrend,
    "second_diff": second_diff,
    "rolling_zscore": rolling_zscore,
}


def apply_transform(
    series: pd.Series, name: str, params: dict[str, int | float | str]
) -> pd.Series:
    fn = TRANSFORMS.get(name)
    if fn is None:
        raise ValueError(f"unknown transform {name!r}; expected one of {sorted(TRANSFORMS)}")
    return fn(series, **params)
