import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from market_regime_detector.common.feature_config import FEATURE_CONFIG_V1, Axis, FeatureConfig
from market_regime_detector.common.scaling import DEFAULT_CLIP
from market_regime_detector.training.baseline import baseline_signal
from market_regime_detector.training.calibration import (
    LinearShrinkCalibrator,
    fit_shrink_calibrator,
    identity_shrink_calibrator,
)
from market_regime_detector.training.evaluation import walk_forward_evaluate
from market_regime_detector.training.labels import LabelSet
from market_regime_detector.training.report import (
    combine_quadrant_probabilities,
    compare_to_baseline,
    episode_report,
    render_model_card,
)
from market_regime_detector.training.trainer import FitResult, fit_hmm


@dataclass(frozen=True)
class HMMHyperparameters:
    n_restarts: int = 10
    covariance_type: str = "diag"
    min_state_duration_months: float = 2.0
    composite_separation_margin: float = 0.1
    initial_window_months: int = 60
    refit_every_months: int = 12
    random_state: int | None = 0
    scaler_clip: float = DEFAULT_CLIP
    scaler_method: str = "median_mad"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_HYPERPARAMETERS = HMMHyperparameters()


def _save_hmm_npz(path: Path, fit: FitResult) -> None:
    """No pickle, per brief section 2.7 — plain NumPy arrays only."""
    np.savez(
        path,
        start_prob=fit.start_prob,
        transition_matrix=fit.transition_matrix,
        means=fit.means,
        variances=fit.variances,
        feature_names=np.array(fit.feature_names),
    )


def train_and_save_model(
    output_dir: Path,
    growth_df: pd.DataFrame,
    inflation_df: pd.DataFrame,
    feature_config: FeatureConfig = FEATURE_CONFIG_V1,
    hyperparameters: HMMHyperparameters = DEFAULT_HYPERPARAMETERS,
    labels: LabelSet | None = None,
) -> None:
    """Fits both HMMs on the full history (what gets saved — walk-forward fits below
    are diagnostic only, never saved), runs walk-forward evaluation + baseline +
    episode report for honest out-of-sample metrics, and writes brief section 4.1's
    six local model-directory files.

    `labels` (v4, `training/calibration.py`) fits each axis's posterior
    calibrator — a single `LinearShrinkCalibrator` per axis, `base_rate` taken
    from that axis's own ground-truth label over all history available at
    train time (never refit later, same as every other trained parameter
    here). `labels=None` (the default, and every pre-v4 call site) saves the
    identity calibrator — a model trained without labels predicts exactly as
    it did before this parameter existed, satisfying the backward-
    compatibility constraint from the other direction: an **old caller** of
    this **new** code must not get a surprise.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hp = hyperparameters

    growth_features = [f for f in feature_config.features if f.axis == Axis.GROWTH]
    inflation_features = [f for f in feature_config.features if f.axis == Axis.INFLATION]

    growth_fit = fit_hmm(
        growth_df,
        growth_features,
        n_restarts=hp.n_restarts,
        covariance_type=hp.covariance_type,
        min_state_duration_months=hp.min_state_duration_months,
        composite_separation_margin=hp.composite_separation_margin,
        random_state=hp.random_state,
        scaler_clip=hp.scaler_clip,
        scaler_method=hp.scaler_method,
    )
    inflation_fit = fit_hmm(
        inflation_df,
        inflation_features,
        n_restarts=hp.n_restarts,
        covariance_type=hp.covariance_type,
        min_state_duration_months=hp.min_state_duration_months,
        composite_separation_margin=hp.composite_separation_margin,
        random_state=hp.random_state,
        scaler_clip=hp.scaler_clip,
        scaler_method=hp.scaler_method,
    )

    growth_walk_forward = walk_forward_evaluate(
        growth_df,
        growth_features,
        initial_window_months=hp.initial_window_months,
        refit_every_months=hp.refit_every_months,
        n_restarts=hp.n_restarts,
        covariance_type=hp.covariance_type,
        min_state_duration_months=hp.min_state_duration_months,
        composite_separation_margin=hp.composite_separation_margin,
        random_state=hp.random_state,
        scaler_clip=hp.scaler_clip,
        scaler_method=hp.scaler_method,
    )
    inflation_walk_forward = walk_forward_evaluate(
        inflation_df,
        inflation_features,
        initial_window_months=hp.initial_window_months,
        refit_every_months=hp.refit_every_months,
        n_restarts=hp.n_restarts,
        covariance_type=hp.covariance_type,
        min_state_duration_months=hp.min_state_duration_months,
        composite_separation_margin=hp.composite_separation_margin,
        random_state=hp.random_state,
        scaler_clip=hp.scaler_clip,
        scaler_method=hp.scaler_method,
    )

    growth_calibrator: LinearShrinkCalibrator = identity_shrink_calibrator()
    inflation_calibrator: LinearShrinkCalibrator = identity_shrink_calibrator()
    if labels is not None:
        if labels.growth.dropna().shape[0] > 0:
            growth_calibrator = fit_shrink_calibrator(labels.growth)
        if labels.inflation.dropna().shape[0] > 0:
            inflation_calibrator = fit_shrink_calibrator(labels.inflation)

    growth_baseline = baseline_signal(growth_df, growth_features)
    inflation_baseline = baseline_signal(inflation_df, inflation_features)
    growth_comparison = compare_to_baseline(growth_walk_forward, growth_baseline)
    inflation_comparison = compare_to_baseline(inflation_walk_forward, inflation_baseline)

    quadrants = combine_quadrant_probabilities(growth_walk_forward, inflation_walk_forward)
    quadrant_episodes = episode_report(quadrants, value_column="confidence")

    _save_hmm_npz(output_dir / "growth_hmm.npz", growth_fit)
    _save_hmm_npz(output_dir / "inflation_hmm.npz", inflation_fit)

    np.savez(
        output_dir / "scalers.npz",
        growth_mean=growth_fit.scaler_mean,
        growth_scale=growth_fit.scaler_scale,
        growth_clip=np.float64(growth_fit.scaler_clip),
        inflation_mean=inflation_fit.scaler_mean,
        inflation_scale=inflation_fit.scaler_scale,
        inflation_clip=np.float64(inflation_fit.scaler_clip),
        growth_calibrator_weight=np.float64(growth_calibrator.weight),
        growth_calibrator_base_rate=np.float64(growth_calibrator.base_rate),
        inflation_calibrator_weight=np.float64(inflation_calibrator.weight),
        inflation_calibrator_base_rate=np.float64(inflation_calibrator.base_rate),
    )

    (output_dir / "labels.json").write_text(
        json.dumps(
            {
                "growth": {
                    "state_labels": growth_fit.state_labels,
                    "composites": growth_fit.composites,
                },
                "inflation": {
                    "state_labels": inflation_fit.state_labels,
                    "composites": inflation_fit.composites,
                },
            },
            indent=2,
        )
    )

    (output_dir / "config.json").write_text(
        json.dumps(
            {"feature_config": feature_config.to_dict(), "hyperparameters": hp.to_dict()},
            indent=2,
        )
    )

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "growth_calibrator": {
                    "weight": growth_calibrator.weight,
                    "base_rate": growth_calibrator.base_rate,
                },
                "inflation_calibrator": {
                    "weight": inflation_calibrator.weight,
                    "base_rate": inflation_calibrator.base_rate,
                },
                "growth_log_likelihood": growth_fit.log_likelihood,
                "inflation_log_likelihood": inflation_fit.log_likelihood,
                "growth_baseline_comparison": growth_comparison,
                "inflation_baseline_comparison": inflation_comparison,
                "n_growth_walk_forward_folds_evaluated": (
                    int(growth_walk_forward["fold_id"].nunique())
                    if not growth_walk_forward.empty
                    else 0
                ),
                "n_growth_walk_forward_folds_attempted": growth_walk_forward.attrs.get(
                    "n_folds_attempted", 0
                ),
                "growth_walk_forward_skipped_folds": growth_walk_forward.attrs.get(
                    "skipped_folds", []
                ),
                "n_inflation_walk_forward_folds_evaluated": (
                    int(inflation_walk_forward["fold_id"].nunique())
                    if not inflation_walk_forward.empty
                    else 0
                ),
                "n_inflation_walk_forward_folds_attempted": inflation_walk_forward.attrs.get(
                    "n_folds_attempted", 0
                ),
                "inflation_walk_forward_skipped_folds": inflation_walk_forward.attrs.get(
                    "skipped_folds", []
                ),
                "quadrant_episode_report": quadrant_episodes.to_dict(orient="records"),
            },
            indent=2,
            default=str,
        )
    )

    model_card = render_model_card(
        feature_config=feature_config,
        growth_fit=growth_fit,
        inflation_fit=inflation_fit,
        growth_baseline_comparison=growth_comparison,
        inflation_baseline_comparison=inflation_comparison,
        quadrant_episode_report=quadrant_episodes,
        hyperparameters=hp.to_dict(),
        growth_skipped_folds=growth_walk_forward.attrs.get("skipped_folds", []),
        inflation_skipped_folds=inflation_walk_forward.attrs.get("skipped_folds", []),
    )
    (output_dir / "model_card.md").write_text(model_card)
