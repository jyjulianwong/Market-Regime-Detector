from collections.abc import Callable

import pandas as pd


def identity(series: pd.Series) -> pd.Series:
    """No-op transform — for the one feature (T10Y2Y) fed as a raw level rather than
    a rate of change (see CLAUDE.md for why this is a deliberate exception).
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


TRANSFORMS: dict[str, Callable[..., pd.Series]] = {
    "identity": identity,
    "pct_change": pct_change,
    "diff": diff,
    "pct_change_annualized": pct_change_annualized,
    "rolling_mean_diff": rolling_mean_diff,
}


def apply_transform(series: pd.Series, name: str, params: dict[str, int]) -> pd.Series:
    fn = TRANSFORMS.get(name)
    if fn is None:
        raise ValueError(f"unknown transform {name!r}; expected one of {sorted(TRANSFORMS)}")
    return fn(series, **params)
