import pandas as pd

from market_regime_detector.common.feature_config import FeatureDefinition, Frequency
from market_regime_detector.common.pit import point_in_time_series
from market_regime_detector.common.transforms import apply_transform


def _resample_to_monthly(series: pd.Series, aggregation: str) -> pd.Series:
    """Resamples an *already asof-truncated* series to monthly.

    Because the series was truncated to `evaluation_date` before this is called, the
    final bucket is the complete month when `evaluation_date` is an exact month-end
    (training) and a partial month when it's mid-month (daily inference) —
    automatically, with no special-casing. This is brief section 3.3's "trailing
    windows ending that day" falling out of resample order, not a separate code path.
    """
    resampler = series.resample("ME")
    if aggregation == "last":
        result: pd.Series = resampler.last()
    elif aggregation == "mean":
        result = resampler.mean()
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")
    return result.dropna()


def as_of_lookup(series: pd.Series, evaluation_date: pd.Timestamp) -> float | None:
    """Returns the latest value at or before `evaluation_date`, or `None` if there
    isn't one yet. Only meaningful when `series`'s index still holds true observation
    dates — see `latest_value` for why a resampled-to-monthly series can't use this.
    """
    eligible = series[series.index <= evaluation_date].dropna()
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def latest_value(series: pd.Series) -> float | None:
    """Returns the last non-null value, with no further date filtering.

    `.resample("ME")` labels every bucket with its *calendar* month-end, even when the
    underlying data was truncated mid-month — so for a mid-month `evaluation_date`
    the final bucket's label falls *after* `evaluation_date`, and `as_of_lookup`'s
    `index <= evaluation_date` check would incorrectly drop the very partial-month
    bucket the truncation was meant to produce. Truncation to `evaluation_date`
    already happened before resampling, so trusting the resampled series' last value
    directly (not its label) is what's actually correct here.
    """
    eligible = series.dropna()
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def build_feature_value(
    raw_df: pd.DataFrame,
    feature: FeatureDefinition,
    evaluation_date: pd.Timestamp,
) -> float | None:
    """Computes one feature's value as of `evaluation_date`, using only what was
    knowable by then. `raw_df` is one series' canonical-schema ingestion rows.

    Not for `feature.is_ratio` features — use `build_ratio_feature_value` instead.
    """
    if feature.is_ratio:
        raise ValueError(f"{feature.name!r} is a ratio feature; use build_ratio_feature_value")

    causal = point_in_time_series(
        raw_df,
        asof=evaluation_date,
        vintage_aware=feature.vintage_aware,
        publication_lag_days=feature.publication_lag_days,
    )
    if causal.empty:
        return None

    if feature.native_frequency == Frequency.DAILY:
        causal = _resample_to_monthly(causal, feature.aggregation)
        if causal.empty:
            return None
        transformed = apply_transform(causal, feature.transform, feature.transform_params)
        return latest_value(transformed)

    transformed = apply_transform(causal, feature.transform, feature.transform_params)
    return as_of_lookup(transformed, evaluation_date)


def build_ratio_feature_value(
    numerator_df: pd.DataFrame,
    denominator_df: pd.DataFrame,
    feature: FeatureDefinition,
    evaluation_date: pd.Timestamp,
) -> float | None:
    """The one multi-series feature (copper/gold momentum): point-in-time filters
    both legs independently, divides (pandas aligns on date, so this is equivalent to
    an inner join on the dates present in both), then reuses the same monthly-resample
    + transform + as-of-lookup pipeline as `build_feature_value`.
    """
    if not feature.is_ratio:
        raise ValueError(f"{feature.name!r} is not a ratio feature; use build_feature_value")

    numerator = point_in_time_series(
        numerator_df,
        asof=evaluation_date,
        vintage_aware=feature.vintage_aware,
        publication_lag_days=feature.publication_lag_days,
    )
    denominator = point_in_time_series(
        denominator_df,
        asof=evaluation_date,
        vintage_aware=feature.vintage_aware,
        publication_lag_days=feature.publication_lag_days,
    )
    if numerator.empty or denominator.empty:
        return None

    ratio = (numerator / denominator).dropna()
    if ratio.empty:
        return None

    if feature.native_frequency == Frequency.DAILY:
        ratio = _resample_to_monthly(ratio, feature.aggregation)
        if ratio.empty:
            return None
        transformed = apply_transform(ratio, feature.transform, feature.transform_params)
        return latest_value(transformed)

    transformed = apply_transform(ratio, feature.transform, feature.transform_params)
    return as_of_lookup(transformed, evaluation_date)
