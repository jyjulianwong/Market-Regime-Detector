import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from market_regime_detector.common.feature_config import Axis, FeatureConfig, used_series_ids
from market_regime_detector.common.features import build_feature_value, build_ratio_feature_value
from market_regime_detector.common.forward_filter import filtered_posterior
from market_regime_detector.common.model_registry import get_model, get_production_model
from market_regime_detector.common.quadrants import quadrant_probabilities
from market_regime_detector.ingestion.lake import read_series
from market_regime_detector.ingestion.providers.factory import build_provider
from market_regime_detector.ingestion.schema import validate_schema
from market_regime_detector.prediction.model_loader import ModelArtifact, load_model_artifact

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_dynamodb.client import DynamoDBClient
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

# Enough periods for the forward filter's start_prob initialization bias to wash
# out — the same accepted simplification as Phase 4's walk-forward segments, not a
# new kind of approximation.
DEFAULT_LOOKBACK_DAYS = 730

# How far behind `asof` the latest usable observation can be before this run's own
# output is marked `is_stale` (brief section 3.3's generation-time staleness signal
# — distinct from the server's own request-time freshness check, Phase 7).
STALE_THRESHOLD_DAYS = 3


@dataclass
class PredictionResult:
    model_id: str
    dataset_id: str
    asof: date
    generated_at: datetime
    observation_date: date | None
    is_stale: bool
    warnings: list[str]
    # One row per usable day: date, p_growth_up, p_inflation_up, regime, confidence,
    # and the four quadrant probability columns.
    history: pd.DataFrame


def _fetch_raw_series(
    s3: "S3Client",
    bucket: str,
    feature_config: FeatureConfig,
    source: str,
    start: date,
    end: date,
    asof: date,
) -> dict[str, pd.DataFrame]:
    """`source="lake"` (default) reads what's already in the S3 lake. `source="live"`
    bypasses the lake entirely — each series is fetched from **its own configured
    provider** (`feature.source`), not a single caller-supplied provider name; the
    feature config spans three different providers (brief section 3.1), so one name
    can't serve every series. This resolves the brief's "a provider name" ASSUMPTION
    for `source`.
    """
    series_ids = used_series_ids(feature_config)

    if source == "lake":
        return {series_id: read_series(s3, bucket, series_id) for series_id in series_ids}

    if source != "live":
        raise ValueError(f"unknown source {source!r}; expected 'lake' or 'live'")

    provider_name_by_series: dict[str, str] = {}
    for feature in feature_config.features:
        if feature.is_ratio:
            assert feature.numerator_series_id is not None
            assert feature.denominator_series_id is not None
            provider_name_by_series[feature.numerator_series_id] = feature.source
            provider_name_by_series[feature.denominator_series_id] = feature.source
        else:
            provider_name_by_series[feature.series_id] = feature.source

    raw: dict[str, pd.DataFrame] = {}
    for series_id in series_ids:
        provider = build_provider(provider_name_by_series[series_id])
        raw[series_id] = validate_schema(provider.fetch(series_id, start, end, asof=asof))
    return raw


def _build_daily_axis_df(
    axis: Axis,
    feature_config: FeatureConfig,
    raw_by_series: dict[str, pd.DataFrame],
    evaluation_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Deliberately not a shared call into `training/dataset.py` — this project's
    one hard dependency rule (brief section 3.5) is that nothing
    under `prediction/` imports from `training/`, because `training/trainer.py`
    imports `hmmlearn` and the batch Lambda's dependency group excludes it. This
    duplicates ~15 lines of axis-matrix assembly rather than risk that boundary.
    """
    columns: dict[str, pd.Series] = {}
    for feature in feature_config.features:
        if feature.axis != axis:
            continue
        if feature.is_ratio:
            assert feature.numerator_series_id is not None
            assert feature.denominator_series_id is not None
            values = [
                build_ratio_feature_value(
                    raw_by_series[feature.numerator_series_id],
                    raw_by_series[feature.denominator_series_id],
                    feature,
                    d,
                )
                for d in evaluation_dates
            ]
        else:
            raw_df = raw_by_series[feature.series_id]
            values = [build_feature_value(raw_df, feature, d) for d in evaluation_dates]
        columns[feature.name] = pd.Series(values, index=evaluation_dates, dtype=float)
    return pd.DataFrame(columns).dropna(how="any")


def _p_up_series(
    axis_df: pd.DataFrame, feature_names: list[str], artifact_fields: ModelArtifact, axis: Axis
) -> pd.Series:
    if axis == Axis.GROWTH:
        hmm, mean, scale = (
            artifact_fields.growth,
            artifact_fields.growth_scaler_mean,
            artifact_fields.growth_scaler_scale,
        )
    else:
        hmm, mean, scale = (
            artifact_fields.inflation,
            artifact_fields.inflation_scaler_mean,
            artifact_fields.inflation_scaler_scale,
        )

    ordered = axis_df[feature_names]
    standardized = (ordered.to_numpy() - mean) / scale
    posterior = filtered_posterior(
        standardized, hmm.start_prob, hmm.transition_matrix, hmm.means, hmm.variances
    )
    return pd.Series(posterior[:, hmm.up_state], index=axis_df.index)


def run_prediction(
    s3: "S3Client",
    dynamodb_client: "DynamoDBClient",
    bucket: str,
    table_name: str,
    model_id: str | None = None,
    source: str = "lake",
    start: date | None = None,
    end: date | None = None,
    asof: date | None = None,
) -> PredictionResult:
    """Brief section 3.3's batch prediction, as a pure function over its
    dependencies — no S3 writes here, so this is testable without touching
    `prediction/outputs.py`.
    """
    effective_asof = asof if asof is not None else date.today()
    effective_end = min(end, effective_asof) if end is not None else effective_asof
    effective_start = (
        start if start is not None else effective_end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    )

    registry_item = (
        get_model(dynamodb_client, table_name, model_id)
        if model_id is not None
        else get_production_model(dynamodb_client, table_name)
    )
    if registry_item is None:
        raise ValueError(
            f"model {model_id!r} not found" if model_id is not None else "no production model"
        )
    resolved_model_id = registry_item["model_id"]
    dataset_id = registry_item["dataset_id"]
    assert isinstance(resolved_model_id, str)
    assert isinstance(dataset_id, str)

    artifact = load_model_artifact(s3, bucket, resolved_model_id)

    warnings: list[str] = []
    raw_by_series = _fetch_raw_series(
        s3, bucket, artifact.feature_config, source, effective_start, effective_end, effective_asof
    )

    # Staleness is about whether the *raw inputs* are actually current, not whether
    # the feature pipeline produced a value: build_feature_value's as-of lookup
    # forward-fills the last known value, so a feature stays "available" long after
    # a series has actually stopped updating — checking the feature output's own
    # last date would almost never catch that.
    raw_last_dates: list[pd.Timestamp] = []
    for series_id, df in raw_by_series.items():
        if df.empty:
            warnings.append(f"no data available for series {series_id!r}")
        else:
            raw_last_dates.append(df["date"].max())
    stale_inputs = (
        bool(raw_last_dates)
        and (effective_asof - min(raw_last_dates).date()).days > STALE_THRESHOLD_DAYS
    )
    if stale_inputs:
        warnings.append(
            f"at least one input series has no data in the last {STALE_THRESHOLD_DAYS} days"
        )

    evaluation_dates = pd.date_range(effective_start, effective_end, freq="D")
    growth_df = _build_daily_axis_df(
        Axis.GROWTH, artifact.feature_config, raw_by_series, evaluation_dates
    )
    inflation_df = _build_daily_axis_df(
        Axis.INFLATION, artifact.feature_config, raw_by_series, evaluation_dates
    )

    common_dates = growth_df.index.intersection(inflation_df.index)
    if len(common_dates) == 0:
        warnings.append("no usable observations in range — every day is missing a feature")
        return PredictionResult(
            model_id=resolved_model_id,
            dataset_id=dataset_id,
            asof=effective_asof,
            generated_at=datetime.now(UTC),
            observation_date=None,
            is_stale=True,
            warnings=warnings,
            history=pd.DataFrame(
                columns=[
                    "p_growth_up",
                    "p_inflation_up",
                    "GROWTH_UP_INFLATION_DOWN",
                    "GROWTH_UP_INFLATION_UP",
                    "GROWTH_DOWN_INFLATION_UP",
                    "GROWTH_DOWN_INFLATION_DOWN",
                    "regime",
                    "confidence",
                ]
            ),
        )

    p_growth_up = _p_up_series(
        growth_df.loc[common_dates], artifact.growth.feature_names, artifact, Axis.GROWTH
    )
    p_inflation_up = _p_up_series(
        inflation_df.loc[common_dates], artifact.inflation.feature_names, artifact, Axis.INFLATION
    )

    quadrants = pd.DataFrame(
        quadrant_probabilities(p_growth_up, p_inflation_up),  # type: ignore[arg-type]
        index=common_dates,
    )
    history = quadrants.copy()
    history.insert(0, "p_inflation_up", p_inflation_up)
    history.insert(0, "p_growth_up", p_growth_up)
    history["regime"] = quadrants.idxmax(axis=1)
    history["confidence"] = quadrants.max(axis=1)

    observation_date = history.index[-1].date()
    is_stale = stale_inputs or bool(warnings)

    return PredictionResult(
        model_id=resolved_model_id,
        dataset_id=dataset_id,
        asof=effective_asof,
        generated_at=datetime.now(UTC),
        observation_date=observation_date,
        is_stale=is_stale,
        warnings=warnings,
        history=history,
    )
