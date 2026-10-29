import io
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from market_regime_detector.common.feature_config import FeatureConfig

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_s3.client import S3Client


@dataclass(frozen=True)
class HMMArtifact:
    """Exactly what `common/forward_filter.py` needs plus the label mapping —
    deliberately not `training.trainer.FitResult`, which would import `hmmlearn`
    transitively (the batch Lambda's dependency group excludes it, brief section
    3.5) and carries training-only fields (log-likelihood, composites) this phase
    has no use for.
    """

    start_prob: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    feature_names: list[str]
    state_labels: dict[str, str]  # state index (as string) -> "UP" | "DOWN"

    @property
    def up_state(self) -> int:
        return next(int(state) for state, label in self.state_labels.items() if label == "UP")


@dataclass(frozen=True)
class ModelArtifact:
    growth: HMMArtifact
    inflation: HMMArtifact
    growth_scaler_mean: np.ndarray
    growth_scaler_scale: np.ndarray
    growth_scaler_clip: float
    inflation_scaler_mean: np.ndarray
    inflation_scaler_scale: np.ndarray
    inflation_scaler_clip: float
    feature_config: FeatureConfig


def _get_object_bytes(s3: "S3Client", bucket: str, key: str) -> bytes:
    response = s3.get_object(Bucket=bucket, Key=key)
    body: bytes = response["Body"].read()
    return body


def _load_hmm_npz(
    s3: "S3Client", bucket: str, key: str, state_labels: dict[str, str]
) -> HMMArtifact:
    npz = np.load(io.BytesIO(_get_object_bytes(s3, bucket, key)))
    return HMMArtifact(
        start_prob=npz["start_prob"],
        transition_matrix=npz["transition_matrix"],
        means=npz["means"],
        variances=npz["variances"],
        feature_names=[str(name) for name in npz["feature_names"]],
        state_labels=state_labels,
    )


def load_model_artifact(
    s3: "S3Client", bucket: str, model_name: str, model_id: str
) -> ModelArtifact:
    """Reads the six files `training/publisher.py` uploaded to
    `models/<model_name>/<model_id>/` and reconstructs everything inference needs —
    no pickle, per brief section 2.7.

    `growth_clip`/`inflation_clip` were added to `scalers.npz` by
    `HMM_ACCURACY_PLAN.md` Phase 1 — a model published before that change has
    neither key, so they default to `np.inf` (no clipping), reproducing exactly
    that old model's original, unclipped behavior rather than silently changing
    it under an artifact that predates clipping.
    """
    prefix = f"models/{model_name}/{model_id}"

    labels = json.loads(_get_object_bytes(s3, bucket, f"{prefix}/labels.json"))
    config = json.loads(_get_object_bytes(s3, bucket, f"{prefix}/config.json"))
    scalers = np.load(io.BytesIO(_get_object_bytes(s3, bucket, f"{prefix}/scalers.npz")))

    growth = _load_hmm_npz(s3, bucket, f"{prefix}/growth_hmm.npz", labels["growth"]["state_labels"])
    inflation = _load_hmm_npz(
        s3, bucket, f"{prefix}/inflation_hmm.npz", labels["inflation"]["state_labels"]
    )

    return ModelArtifact(
        growth=growth,
        inflation=inflation,
        growth_scaler_mean=scalers["growth_mean"],
        growth_scaler_scale=scalers["growth_scale"],
        growth_scaler_clip=float(scalers["growth_clip"]) if "growth_clip" in scalers else np.inf,
        inflation_scaler_mean=scalers["inflation_mean"],
        inflation_scaler_scale=scalers["inflation_scale"],
        inflation_scaler_clip=(
            float(scalers["inflation_clip"]) if "inflation_clip" in scalers else np.inf
        ),
        feature_config=FeatureConfig.from_dict(config["feature_config"]),
    )
