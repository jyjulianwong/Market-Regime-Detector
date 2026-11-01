#!/usr/bin/env python3
"""v4 experiment E1: does Platt-calibrating the raw HMM posterior beat it?

Usage:
    uv run python scripts/evaluate_calibration.py --start 2000-01-01 \
        --output-dir evaluation/allweather-hmm2naive-us/v4-calibration --feature-config v3

Builds the same walk-forward OOS posteriors `scripts/evaluate_model.py` scores
(same dataset, same `DEFAULT_HYPERPARAMETERS`), then compares four treatments
of each axis's raw `p_up` against the pre-registered ground-truth label:

- `raw`: unchanged (today's production behaviour).
- `platt`: `training/calibration.expanding_window_calibrate` — a Platt
  calibrator refit on every walk-forward fold boundary using only strictly
  earlier OOS posteriors, so this is itself a fully causal OOS comparison, not
  circular.
- `shrink`: `training/calibration.shrink_to_base_rate` (fixed 50/50 shrink
  toward the label's own historical base rate) — the report's simpler
  comparison point.
- `hard_cap`: `training/calibration.hard_cap` at [0.10, 0.90] — the report's
  stopgap fallback if `platt` doesn't clear the bar.

Metrics: log loss, Brier, AUC (should be unchanged for `platt`/`hard_cap`,
both monotone maps — `shrink` is also monotone here since it's an affine
function of p), skill vs. climatology with a block-bootstrap CI, and ECE.
Never changes the model or writes a published artifact — read-only against
the lake, writes only to `--output-dir`.
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from market_regime_detector.common.aws_s3 import get_s3_client
from market_regime_detector.common.feature_config import Axis, FeatureConfig
from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.settings import get_settings
from market_regime_detector.training.benchmarks import climatology_baseline
from market_regime_detector.training.build_model import DEFAULT_HYPERPARAMETERS, HMMHyperparameters
from market_regime_detector.training.calibration import (
    expanding_window_calibrate,
    expanding_window_shrink,
    hard_cap,
    shrink_to_base_rate,
)
from market_regime_detector.training.dataset import build_dataset
from market_regime_detector.training.evaluation import walk_forward_evaluate
from market_regime_detector.training.labels import build_label_set
from market_regime_detector.training.scoring import (
    accuracy,
    auc,
    block_bootstrap_ci,
    brier,
    calibration_table,
    log_loss,
)

logger = logging.getLogger(__name__)

TREATMENTS = ("raw", "platt", "shrink_causal", "shrink_noncausal_upper_bound", "hard_cap")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-config", default="v3")
    return parser.parse_args()


def _per_month_log_loss(p: pd.Series, y: pd.Series) -> pd.Series:
    aligned = pd.concat([p.rename("p"), y.rename("y")], axis=1, join="inner").dropna()
    p_clipped = aligned["p"].clip(1e-4, 1 - 1e-4)
    result: pd.Series = -(
        aligned["y"] * np.log(p_clipped) + (1 - aligned["y"]) * np.log(1 - p_clipped)
    )
    return result


def _skill_with_ci(
    model_p: pd.Series, climatology_p: pd.Series, label: pd.Series
) -> tuple[float, float, float]:
    model_losses = _per_month_log_loss(model_p, label)
    climatology_losses = _per_month_log_loss(climatology_p, label)
    common = model_losses.index.intersection(climatology_losses.index)
    ratio_ci = block_bootstrap_ci(model_losses.loc[common], climatology_losses.loc[common])
    point = 1.0 - model_losses.loc[common].mean() / climatology_losses.loc[common].mean()
    # block_bootstrap_ci on (a, b) returns a CI for a generic statistic of the
    # pair; here we want the skill ratio's own CI, so bootstrap it directly.
    del ratio_ci
    model_arr = model_losses.loc[common].to_numpy()
    climatology_arr = climatology_losses.loc[common].to_numpy()
    n = len(common)
    block = 6
    if n < block:
        return float(point), float(point), float(point)
    rng = np.random.default_rng(0)
    n_boot = 2000
    n_blocks_needed = -(-n // block)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks_needed)
        m = np.concatenate([model_arr[s : s + block] for s in starts])[:n]
        c = np.concatenate([climatology_arr[s : s + block] for s in starts])[:n]
        boot[i] = 1.0 - m.mean() / c.mean()
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(point), float(low), float(high)


def _axis_report(
    axis_df: pd.DataFrame,
    feature_config: FeatureConfig,
    axis: Axis,
    label: pd.Series,
    hp: HMMHyperparameters,
) -> dict[str, object]:
    feature_defs = [f for f in feature_config.features if f.axis == axis]
    wf = walk_forward_evaluate(
        axis_df,
        feature_defs,
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
    raw_p = pd.Series(wf["p_up"].to_numpy(), index=pd.DatetimeIndex(wf["decision_date"]))
    climatology_p = climatology_baseline(
        label, pd.DatetimeIndex(axis_df.index), 1, hp.initial_window_months
    )

    treatments = {
        "raw": raw_p,
        "platt": pd.Series(expanding_window_calibrate(wf, label).to_numpy(), index=raw_p.index),
        "shrink_causal": pd.Series(
            expanding_window_shrink(wf, label).to_numpy(), index=raw_p.index
        ),
        "shrink_noncausal_upper_bound": shrink_to_base_rate(raw_p, label.reindex(raw_p.index)),
        "hard_cap": hard_cap(raw_p),
    }

    metrics: dict[str, object] = {"skipped_folds": len(wf.attrs.get("skipped_folds", []))}
    for name, p in treatments.items():
        skill, skill_low, skill_high = _skill_with_ci(p, climatology_p, label)
        _, ece = calibration_table(p, label)
        metrics[name] = {
            "n": int(pd.concat([p, label], axis=1, join="inner").dropna().shape[0]),
            "log_loss": log_loss(p, label),
            "brier": brier(p, label),
            "auc": auc(p, label),
            "accuracy": accuracy(p, label),
            "ece": ece,
            "skill_vs_climatology": skill,
            "skill_ci_low": skill_low,
            "skill_ci_high": skill_high,
        }
    return metrics


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args()
    end = args.end if args.end is not None else date.today()

    from market_regime_detector.common.feature_config import FEATURE_CONFIGS

    feature_config = FEATURE_CONFIGS[args.feature_config]
    s3 = get_s3_client()
    dataset = build_dataset(
        s3, settings.s3_bucket_name, pd.Timestamp(args.start), pd.Timestamp(end), feature_config
    )

    from market_regime_detector.ingestion.lake import read_series

    def _label_series(series_id: str) -> pd.Series:
        df = read_series(s3, settings.s3_bucket_name, series_id)
        result: pd.Series = df.sort_values("date").groupby("date")["value"].last()
        return result

    labels = build_label_set(
        _label_series("CFNAIMA3"), _label_series("USREC"), _label_series("CPIAUCSL")
    )

    report = {
        "growth": _axis_report(
            dataset.growth_df, feature_config, Axis.GROWTH, labels.growth, DEFAULT_HYPERPARAMETERS
        ),
        "inflation": _axis_report(
            dataset.inflation_df,
            feature_config,
            Axis.INFLATION,
            labels.inflation,
            DEFAULT_HYPERPARAMETERS,
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration_evaluation.json").write_text(json.dumps(report, indent=2))
    logger.info("calibration evaluation written to %s", args.output_dir)

    for axis_name, axis_report in report.items():
        print(f"\n=== {axis_name} ===")
        print(f"skipped_folds={axis_report['skipped_folds']}")
        for treatment in TREATMENTS:
            m = cast("dict[str, object]", axis_report[treatment])
            print(
                f"{treatment:<10} n={m['n']:>4} log_loss={cast(float, m['log_loss']):.4f} "
                f"auc={m['auc']} ece={cast(float, m['ece']):.4f} "
                f"skill={cast(float, m['skill_vs_climatology']):.3f} "
                f"[{cast(float, m['skill_ci_low']):.3f}, {cast(float, m['skill_ci_high']):.3f}]"
            )


if __name__ == "__main__":
    main()
