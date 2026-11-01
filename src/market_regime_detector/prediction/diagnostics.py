from dataclasses import dataclass
from datetime import date as date_type

import pandas as pd

from market_regime_detector.common.feature_config import FeatureConfig, FeatureDefinition
from market_regime_detector.common.features import build_feature_value, build_ratio_feature_value
from market_regime_detector.common.pit import LatestObservation, latest_knowable_observation
from market_regime_detector.ingestion.schema import CANONICAL_COLUMNS
from market_regime_detector.ingestion.series_catalog import FIRST_PASS_SERIES

_EMPTY_RAW_DF = pd.DataFrame(columns=CANONICAL_COLUMNS)


@dataclass
class DataPoint:
    """One row of the `/predictions/latest` `data`/`features` diagnostics (brief
    extension, 2026-11-01) — a name+description+value+date+lag shape shared by
    both: `data` is a raw lake series (`build_data_points`), `features` is one of
    this model's actual transformed inputs (`build_feature_points`). `date`/
    `realtime_start`/`lag_days` are all `None` together when nothing is knowable
    yet for the backing series as of `asof` (e.g. never ingested).
    """

    name: str
    description: str
    value: float | None
    date: date_type | None
    realtime_start: date_type | None
    lag_days: int | None


def _to_data_point(
    name: str,
    description: str,
    value: float | None,
    obs: LatestObservation | None,
    asof: pd.Timestamp,
) -> DataPoint:
    if obs is None:
        return DataPoint(
            name=name,
            description=description,
            value=value,
            date=None,
            realtime_start=None,
            lag_days=None,
        )
    return DataPoint(
        name=name,
        description=description,
        value=value,
        date=obs.date.date(),
        realtime_start=obs.realtime_start.date(),
        lag_days=(asof - obs.realtime_start).days,
    )


def build_data_points(
    raw_lake_by_series: dict[str, pd.DataFrame], asof: pd.Timestamp
) -> list[DataPoint]:
    """One entry per series the weekly ingest Lambda maintains
    (`ingestion.series_catalog.FIRST_PASS_SERIES`) — "every data stream we
    currently have in the data lake," independent of whether any model's
    `FeatureConfig` actually uses it. `raw_lake_by_series` must be read straight
    from the lake (never `source="live"`) — this reports what the lake itself
    holds, not a live provider call.
    """
    points: list[DataPoint] = []
    for spec in FIRST_PASS_SERIES:
        raw_df = raw_lake_by_series.get(spec.series_id, _EMPTY_RAW_DF)
        obs = (
            latest_knowable_observation(
                raw_df, asof, vintage_aware=spec.source == "alfred", publication_lag_days=0
            )
            if not raw_df.empty
            else None
        )
        points.append(
            _to_data_point(spec.series_id, spec.description, obs.value if obs else None, obs, asof)
        )
    return points


def _backing_observation(
    feature: FeatureDefinition, raw_by_series: dict[str, pd.DataFrame], asof: pd.Timestamp
) -> LatestObservation | None:
    """The observation describing how stale the raw data behind `feature` is. For
    the one ratio feature (copper/gold), reports whichever leg is *more* stale
    (the smaller `realtime_start`) — the bottleneck that actually determines how
    current the ratio can be, not an arbitrary pick between the two legs.
    """
    series_ids = (
        [feature.numerator_series_id, feature.denominator_series_id]
        if feature.is_ratio
        else [feature.series_id]
    )
    observations = []
    for series_id in series_ids:
        assert series_id is not None
        raw_df = raw_by_series.get(series_id, _EMPTY_RAW_DF)
        if raw_df.empty:
            continue
        obs = latest_knowable_observation(
            raw_df,
            asof,
            vintage_aware=feature.vintage_aware,
            publication_lag_days=feature.publication_lag_days,
        )
        if obs is not None:
            observations.append(obs)
    if not observations:
        return None
    return min(observations, key=lambda obs: obs.realtime_start)


def build_feature_points(
    feature_config: FeatureConfig,
    raw_by_series: dict[str, pd.DataFrame],
    evaluation_date: pd.Timestamp,
    asof: pd.Timestamp,
) -> list[DataPoint]:
    """One entry per feature actually fed to this model's HMMs — the transformed
    value computed at `evaluation_date` (the same day `run_prediction` previewed
    for `observation_date`), paired with the staleness of the raw series backing
    it (`_backing_observation`), not `build_data_points`' raw lake value.
    """
    points: list[DataPoint] = []
    for feature in feature_config.features:
        if feature.is_ratio:
            assert feature.numerator_series_id is not None
            assert feature.denominator_series_id is not None
            value = build_ratio_feature_value(
                raw_by_series.get(feature.numerator_series_id, _EMPTY_RAW_DF),
                raw_by_series.get(feature.denominator_series_id, _EMPTY_RAW_DF),
                feature,
                evaluation_date,
            )
        else:
            value = build_feature_value(
                raw_by_series.get(feature.series_id, _EMPTY_RAW_DF), feature, evaluation_date
            )

        obs = _backing_observation(feature, raw_by_series, asof)
        points.append(_to_data_point(feature.name, feature.description, value, obs, asof))
    return points
