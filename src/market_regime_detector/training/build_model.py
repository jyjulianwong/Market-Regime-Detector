import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from market_regime_detector.common.feature_config import FEATURE_CONFIG_V1, Axis, FeatureConfig
from market_regime_detector.training.baseline import baseline_signal
from market_regime_detector.training.evaluation import walk_forward_evaluate
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
) -> None:
    """Fits both HMMs on the full history (what gets saved — walk-forward fits below
    are diagnostic only, never saved), runs walk-forward evaluation + baseline +
    episode report for honest out-of-sample metrics, and writes brief section 4.1's
    six local model-directory files.
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
    )
    inflation_fit = fit_hmm(
        inflation_df,
        inflation_features,
        n_restarts=hp.n_restarts,
        covariance_type=hp.covariance_type,
        min_state_duration_months=hp.min_state_duration_months,
        composite_separation_margin=hp.composite_separation_margin,
        random_state=hp.random_state,
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
    )

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
        inflation_mean=inflation_fit.scaler_mean,
        inflation_scale=inflation_fit.scaler_scale,
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
