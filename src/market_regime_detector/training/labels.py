from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LabelSet:
    """Ground truth for scoring only — never fed to a model as a feature (that
    would be circular, since these use final revised data, not point-in-time
    knowledge). `label_lag_months` records how many months behind a decision
    month-end `t` these series are actually knowable by (CFNAI/CPI for month `t`
    publish in `t+1`); applying that lag is the scoring harness's job
    (`training/benchmarks.py`/`training/harness.py`), not this module's — these are
    the raw, un-shifted label series.
    """

    growth: pd.Series
    growth_recession: pd.Series
    inflation: pd.Series
    label_lag_months: int = 1


def _to_month_end(series: pd.Series) -> pd.Series:
    """Reindexes a FRED series (conventionally dated the 1st of its reference
    month) onto real calendar month-ends, matching the feature matrices' index
    convention (`freq="ME"`, brief section 4.2/`training/dataset.py`).
    `.resample("ME").last()` is exact here, not an approximation — each of these
    series has exactly one observation per month, so "last" just relabels it.
    """
    return series.sort_index().resample("ME").last().dropna()


def growth_label(cfnai_ma3: pd.Series) -> pd.Series:
    """1.0 if the Chicago Fed National Activity Index's 3-month average is positive
    ("above-trend growth"), else 0.0. The primary growth label — pre-registered
    (brief section on evaluation): fixed before looking at any score, never
    revised to improve one.
    """
    monthly = _to_month_end(cfnai_ma3)
    return (monthly > 0).astype(float)


def recession_label(usrec: pd.Series) -> pd.Series:
    """`growth_up = 1 - USREC`. A secondary growth label — NBER recessions are rare
    events over this project's sample (brief section on evaluation notes ~2 in the
    scored window), so this is a sanity check against `growth_label`, never a
    second target to tune against.
    """
    monthly = _to_month_end(usrec)
    return (1.0 - monthly).astype(float)


def inflation_label(cpi: pd.Series) -> pd.Series:
    """1.0 if 3-month-annualized CPI inflation exceeds its own trailing year-over-
    year rate as of 3 months ago ("inflation accelerating relative to its recent
    trend"), else 0.0. The primary inflation label — pre-registered, same
    discipline as `growth_label`.

    Both `infl_3m_ann` and `infl_yoy_lagged` must exist for a month to get a label;
    `pandas`' `>` comparison silently evaluates to `False` when either side is
    `NaN`, which would misrepresent "undefined" as "inflation not accelerating" —
    so NaN is restored explicitly rather than trusted to propagate on its own.
    """
    monthly = _to_month_end(cpi)
    infl_3m_ann = (monthly / monthly.shift(3)) ** 4 - 1
    infl_yoy_lagged = monthly.pct_change(12).shift(3)

    label = (infl_3m_ann > infl_yoy_lagged).astype(float)
    label[infl_3m_ann.isna() | infl_yoy_lagged.isna()] = float("nan")
    return label


def build_label_set(
    cfnai_ma3: pd.Series,
    usrec: pd.Series,
    cpi: pd.Series,
    label_lag_months: int = 1,
) -> LabelSet:
    return LabelSet(
        growth=growth_label(cfnai_ma3),
        growth_recession=recession_label(usrec),
        inflation=inflation_label(cpi),
        label_lag_months=label_lag_months,
    )
