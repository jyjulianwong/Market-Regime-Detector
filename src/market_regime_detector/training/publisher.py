import hashlib
import json
import re
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from market_regime_detector.common.git_info import get_git_sha
from market_regime_detector.common.model_registry import put_model_item

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_dynamodb.client import DynamoDBClient
    from mypy_boto3_s3.client import S3Client

PublishStage = Literal["development", "staging"]

_MODEL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)+$")


def _validate_model_name(model_name: str) -> None:
    if not _MODEL_NAME_PATTERN.match(model_name):
        raise ValueError(
            f"model_name {model_name!r} must be lowercase alphanumeric segments "
            "separated by hyphens, e.g. 'allweather-hmm2naive-us'"
        )


def _generate_model_id(model_name: str, dataset_id: str, feature_config_hash: str) -> str:
    """`<model_name>-<YYYYMMDD>-<short_git_sha>` — deterministic so retries never
    mint a new ID for the same model_name/dataset/feature-config. Falls back to a
    content hash of the dataset + feature config when there's no git SHA; still
    deterministic for the same inputs, not random. `model_name` is carried as its
    own registry attribute too (see `_registry_item`) — never re-derived by parsing
    this string back apart.
    """
    date_part = datetime.now(UTC).strftime("%Y%m%d")
    git_sha = get_git_sha()
    if git_sha is not None:
        suffix = git_sha[:8]
    else:
        payload = f"{dataset_id}:{feature_config_hash}".encode()
        suffix = hashlib.sha256(payload).hexdigest()[:8]
    return f"{model_name}-{date_part}-{suffix}"


def _upload_model_files(
    s3: "S3Client", bucket: str, model_name: str, model_id: str, local_model_dir: Path
) -> str:
    prefix = f"models/{model_name}/{model_id}"
    for path in sorted(local_model_dir.iterdir()):
        if path.is_file():
            s3.upload_file(str(path), bucket, f"{prefix}/{path.name}")
    return f"s3://{bucket}/{prefix}/"


def _upload_manifest(s3: "S3Client", bucket: str, manifest: dict[str, object]) -> str:
    dataset_id = manifest["dataset_id"]
    assert isinstance(dataset_id, str)
    key = f"datasets/{dataset_id}/manifest.json"
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(manifest, indent=2, default=str).encode())
    return f"s3://{bucket}/{key}"


def _registry_item(
    model_id: str,
    model_name: str,
    manifest: dict[str, object],
    metrics: dict[str, object],
    hyperparameters: dict[str, object],
    model_s3_uri: str,
    manifest_s3_uri: str,
    stage: PublishStage,
) -> dict[str, object]:
    """Brief section 4.3's flat summary fields only — never the full asset list
    (that stays in the manifest behind `manifest_s3_uri`; DynamoDB items cap at
    400 KB, and section 4.3 explicitly warns against copying it in).
    """
    return {
        "model_id": model_id,
        "model_name": model_name,
        "dataset_id": manifest["dataset_id"],
        "last_modified": datetime.now(UTC).isoformat(),
        "stage": stage,
        "date_start": manifest["date_start"],
        "date_end": manifest["date_end"],
        "decision_frequency": manifest["decision_frequency"],
        "feature_config_hash": manifest["feature_config_hash"],
        "n_assets": len(cast(list[object], manifest["assets"])),
        "git_sha": manifest["git_sha"],
        "vintage_aware": manifest["vintage_aware"],
        "model_s3_uri": model_s3_uri,
        "manifest_s3_uri": manifest_s3_uri,
        "metrics_summary": {
            "growth_log_likelihood": metrics.get("growth_log_likelihood"),
            "inflation_log_likelihood": metrics.get("inflation_log_likelihood"),
            "growth_baseline_comparison": metrics.get("growth_baseline_comparison"),
            "inflation_baseline_comparison": metrics.get("inflation_baseline_comparison"),
        },
        "hmm_config_summary": {
            # Brief section 2.2: two states is the fixed architecture, never an
            # actual hyperparameter — hardcoded here rather than threaded through
            # HMMHyperparameters as a knob that's never really varied.
            "n_states": 2,
            "covariance_type": hyperparameters.get("covariance_type"),
            "n_restarts": hyperparameters.get("n_restarts"),
        },
    }


def publish_model(
    local_model_dir: Path,
    model_name: str,
    manifest: dict[str, object],
    s3: "S3Client",
    bucket: str,
    dynamodb_client: "DynamoDBClient",
    table_name: str,
    stage: PublishStage = "staging",
) -> str:
    """Uploads a local model directory to S3, uploads the dataset manifest to its
    own S3 prefix, and writes the DynamoDB registry item — brief section 3.2's
    publisher, "a single command that handles all three steps and is safe to
    retry," now scoped under `model_name` (multi-model support) so independently
    published model families never collide in S3 or the registry.

    Retry-safety comes from determinism, not a separate idempotency mechanism:
    `model_id` is generated the same way every time for the same model_name +
    dataset + feature config + day (`_generate_model_id`), so re-running uploads
    the same bytes to the same S3 keys and writes a registry item with the same
    identity (`model_id`, `model_name`, `dataset_id`, S3 URIs, `stage`) —
    `last_modified` naturally reflects each actual publish attempt's completion
    time, which is correct behavior for a retry, not a broken idempotency
    guarantee.
    """
    _validate_model_name(model_name)

    # DynamoDB has no native float type (boto3 requires Decimal to avoid
    # floating-point precision ambiguity), so metrics/hyperparameters headed into
    # the registry item are parsed as Decimal straight away.
    config = json.loads((local_model_dir / "config.json").read_text(), parse_float=Decimal)
    metrics = json.loads((local_model_dir / "metrics.json").read_text(), parse_float=Decimal)

    feature_config_hash = manifest["feature_config_hash"]
    dataset_id = manifest["dataset_id"]
    assert isinstance(feature_config_hash, str)
    assert isinstance(dataset_id, str)
    model_id = _generate_model_id(model_name, dataset_id, feature_config_hash)

    model_s3_uri = _upload_model_files(s3, bucket, model_name, model_id, local_model_dir)
    manifest_s3_uri = _upload_manifest(s3, bucket, manifest)

    item = _registry_item(
        model_id,
        model_name,
        manifest,
        metrics,
        config["hyperparameters"],
        model_s3_uri,
        manifest_s3_uri,
        stage,
    )
    put_model_item(dynamodb_client, table_name, item)

    return model_id
