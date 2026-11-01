import json
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

from market_regime_detector.common.quadrants import QUADRANT_NAMES
from market_regime_detector.prediction.diagnostics import DataPoint
from market_regime_detector.prediction.predict import PredictionResult

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_s3.client import S3Client


def _data_point_json(point: DataPoint) -> dict[str, object]:
    return {
        "name": point.name,
        "description": point.description,
        "value": point.value,
        "date": point.date.isoformat() if point.date is not None else None,
        "realtime_start": point.realtime_start.isoformat()
        if point.realtime_start is not None
        else None,
        "lag_days": point.lag_days,
    }


def to_latest_json(result: PredictionResult) -> dict[str, object]:
    """Brief section 5's shape, plus the `data`/`features` input
    diagnostics (2026-11-01 extension). `data`/`features` only ever appear nested
    inside a non-null `latest` — `result.data`/`result.features` are empty on
    every `_empty_prediction_result` early return, so there's nothing to nest when
    `latest` is itself `None`.
    """
    if result.history.empty:
        latest = None
    else:
        last = result.history.iloc[-1]
        latest = {
            "p_growth_up": float(last["p_growth_up"]),
            "p_inflation_up": float(last["p_inflation_up"]),
            "quadrants": {name: float(last[name]) for name in QUADRANT_NAMES},
            "regime": str(last["regime"]),
            "confidence": float(last["confidence"]),
            "data": [_data_point_json(point) for point in result.data],
            "features": [_data_point_json(point) for point in result.features],
        }

    return {
        "model_id": result.model_id,
        "model_name": result.model_name,
        "dataset_id": result.dataset_id,
        "asof": result.asof.isoformat(),
        "generated_at": result.generated_at.isoformat(),
        "observation_date": result.observation_date.isoformat()
        if result.observation_date is not None
        else None,
        "is_stale": result.is_stale,
        "latest": latest,
        "warnings": result.warnings,
    }


def to_history_json(result: PredictionResult) -> dict[str, object]:
    """The full daily series computed *in this run* — not an incremental append
    across runs. Simpler, avoids merge/dedup logic; a deliberate scope reduction
    from a literal "accumulate forever" reading.
    """
    history = result.history.reset_index(names="date")
    records = [
        {
            "date": row["date"].date().isoformat(),
            "p_growth_up": float(row["p_growth_up"]),
            "p_inflation_up": float(row["p_inflation_up"]),
            "quadrants": {name: float(row[name]) for name in QUADRANT_NAMES},
            "regime": str(row["regime"]),
            "confidence": float(row["confidence"]),
        }
        for _, row in history.iterrows()
    ]
    return {
        "model_id": result.model_id,
        "model_name": result.model_name,
        "dataset_id": result.dataset_id,
        "generated_at": result.generated_at.isoformat(),
        "history": records,
    }


def _object_exists(s3: "S3Client", bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise
    return True


def write_prediction_outputs(s3: "S3Client", bucket: str, result: PredictionResult) -> None:
    """Brief section 4.1's `predictions/` layout, now nested under `model_name=`
    (multi-model support) so independently-scheduled predictions for different
    model families never collide. `latest.json`/`history.json` are always
    overwritten; the dated snapshot is **write-once per asof date** — never
    rewritten when data is later revised (brief section 3.3) — so a second call for
    the same `asof` leaves the existing snapshot untouched.
    """
    prefix = f"predictions/model_name={result.model_name}/model_id={result.model_id}"

    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/latest.json",
        Body=json.dumps(to_latest_json(result), indent=2).encode(),
        ContentType="application/json",
    )
    s3.put_object(
        Bucket=bucket,
        Key=f"{prefix}/history.json",
        Body=json.dumps(to_history_json(result), indent=2).encode(),
        ContentType="application/json",
    )

    snapshot_key = f"{prefix}/snapshots/asof={result.asof.isoformat()}.json"
    if _object_exists(s3, bucket, snapshot_key):
        return
    s3.put_object(
        Bucket=bucket,
        Key=snapshot_key,
        Body=json.dumps(to_latest_json(result), indent=2).encode(),
        ContentType="application/json",
    )
