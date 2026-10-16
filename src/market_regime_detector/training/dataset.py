import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd

from market_regime_detector.common.feature_config import (
    FEATURE_CONFIG_V1,
    Axis,
    FeatureConfig,
    FeatureDefinition,
    used_series_ids,
)
from market_regime_detector.common.features import build_feature_value, build_ratio_feature_value
from market_regime_detector.common.git_info import get_git_sha
from market_regime_detector.ingestion.lake import read_series

if TYPE_CHECKING:
    # boto3-stubs is dev-only — see common/s3.py for why this must stay under
    # TYPE_CHECKING rather than a top-level import.
    from mypy_boto3_s3.client import S3Client


@dataclass
class DatasetResult:
    growth_df: pd.DataFrame
    inflation_df: pd.DataFrame
    manifest: dict[str, object]


def _build_feature_column(
    feature: FeatureDefinition,
    raw_by_series: dict[str, pd.DataFrame],
    decision_dates: pd.DatetimeIndex,
) -> pd.Series:
    if feature.is_ratio:
        assert feature.numerator_series_id is not None
        assert feature.denominator_series_id is not None
        numerator_df = raw_by_series[feature.numerator_series_id]
        denominator_df = raw_by_series[feature.denominator_series_id]
        values = [
            build_ratio_feature_value(numerator_df, denominator_df, feature, d)
            for d in decision_dates
        ]
    else:
        raw_df = raw_by_series[feature.series_id]
        values = [build_feature_value(raw_df, feature, d) for d in decision_dates]
    return pd.Series(values, index=decision_dates, name=feature.name, dtype=float)


def _build_axis_df(
    axis: Axis,
    feature_config: FeatureConfig,
    raw_by_series: dict[str, pd.DataFrame],
    decision_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """One row per decision date, one column per feature on this axis.

    Rows with any `None` feature value (the warm-up period before every series has
    enough history for its transform's lookback window) are dropped — the effective
    sample start is whatever the data naturally allows, not a hardcoded date.
    """
    columns = {
        feature.name: _build_feature_column(feature, raw_by_series, decision_dates)
        for feature in feature_config.features
        if feature.axis == axis
    }
    return pd.DataFrame(columns).dropna(how="any")


def _content_hash(raw_by_series: dict[str, pd.DataFrame]) -> str:
    """Stands in for brief section 4.2's "lake object version IDs" hash until Phase 8
    turns on S3 bucket versioning — hashes the actual bytes read instead, which is
    just as reproducible for a given lake state.
    """
    hasher = hashlib.sha256()
    for series_id in sorted(raw_by_series):
        hasher.update(series_id.encode("utf-8"))
        hasher.update(pd.util.hash_pandas_object(raw_by_series[series_id]).to_numpy().tobytes())
    return hasher.hexdigest()


def _asset_manifest_entries(
    raw_by_series: dict[str, pd.DataFrame], feature_config: FeatureConfig
) -> list[dict[str, object]]:
    provenance_by_series: dict[str, tuple[str, int, bool]] = {}
    for feature in feature_config.features:
        series_ids = (
            (feature.numerator_series_id, feature.denominator_series_id)
            if feature.is_ratio
            else (feature.series_id,)
        )
        for series_id in series_ids:
            assert series_id is not None
            provenance_by_series.setdefault(
                series_id, (feature.source, feature.publication_lag_days, feature.vintage_aware)
            )

    entries: list[dict[str, object]] = []
    for series_id, df in sorted(raw_by_series.items()):
        source, lag, vintage_aware = provenance_by_series[series_id]
        entries.append(
            {
                "series_id": series_id,
                "source": source,
                "publication_lag_days": lag,
                "vintage_aware": vintage_aware,
                "first_date": None if df.empty else df["date"].min().date().isoformat(),
                "last_date": None if df.empty else df["date"].max().date().isoformat(),
                "n_rows": len(df),
            }
        )
    return entries


def build_dataset(
    s3: "S3Client",
    bucket: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    feature_config: FeatureConfig = FEATURE_CONFIG_V1,
) -> DatasetResult:
    """Reads the S3 lake only (brief section 3.2 — never a provider directly),
    applies `feature_config`, and produces one feature matrix per axis plus a
    dataset manifest (brief section 4.2).
    """
    decision_dates = pd.date_range(start, end, freq="ME")
    raw_by_series = {
        series_id: read_series(s3, bucket, series_id)
        for series_id in used_series_ids(feature_config)
    }

    growth_df = _build_axis_df(Axis.GROWTH, feature_config, raw_by_series, decision_dates)
    inflation_df = _build_axis_df(Axis.INFLATION, feature_config, raw_by_series, decision_dates)

    warnings = [
        f"no lake data found for series {series_id!r}"
        for series_id, df in raw_by_series.items()
        if df.empty
    ]

    date_start, date_end = start.date().isoformat(), end.date().isoformat()
    feature_config_hash = feature_config.content_hash()
    content_hash = _content_hash(raw_by_series)

    non_empty = [df for df in raw_by_series.values() if not df.empty]
    vintage_aware = any(feature.vintage_aware for feature in feature_config.features)
    earliest_realtime_start = (
        min(df["realtime_start"].min() for df in non_empty).date().isoformat()
        if non_empty
        else None
    )

    dataset_id_payload = json.dumps(
        {
            "feature_config_hash": feature_config_hash,
            "date_start": date_start,
            "date_end": date_end,
            "content_hash": content_hash,
        },
        sort_keys=True,
    )
    dataset_id = "ds-" + hashlib.sha256(dataset_id_payload.encode("utf-8")).hexdigest()[:16]

    manifest: dict[str, object] = {
        "dataset_id": dataset_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": get_git_sha(),
        "feature_config_version": feature_config.version,
        "feature_config_hash": feature_config_hash,
        "decision_frequency": "monthly",
        "date_start": date_start,
        "date_end": date_end,
        "vintage_aware": vintage_aware,
        "earliest_realtime_start": earliest_realtime_start,
        "assets": _asset_manifest_entries(raw_by_series, feature_config),
        "warnings": warnings,
    }

    return DatasetResult(growth_df=growth_df, inflation_df=inflation_df, manifest=manifest)
