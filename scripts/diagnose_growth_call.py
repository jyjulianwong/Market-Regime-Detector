#!/usr/bin/env python3
"""v4 diagnostic D1: explain a single day's axis call feature-by-feature.

Usage:
    uv run python scripts/diagnose_growth_call.py --model-id allweather-hmm2naive-us-20261028-b4887a5e \
        --asof 2026-10-28 --axis growth

For the requested `asof`, reproduces `prediction/predict.run_prediction`'s
posterior (read-only — makes no S3 writes, same guarantee as `run_prediction`
itself), then decomposes the final day's standardized observation into each
feature's own log-density under the UP state and the DOWN state, and the
resulting per-feature log-likelihood-ratio (LLR). Diagonal covariance means the
state log-density is just a sum of independent per-feature Gaussian log-
densities, so this decomposition is exact, not an approximation — it explains
*why* the model leans toward whichever state has the higher summed LLR, even
though the actual posterior also depends on the prior (previous month's
filtered alpha) and the transition matrix, which this script reports separately
rather than folding into the per-feature numbers.

This is a diagnostic only: no model change, no new metric fed back into
training. See `docs/model_cards/v4/MODEL_CARD.md` for how this was used.
"""

import argparse
import logging
from datetime import date, timedelta
from typing import cast

import numpy as np
import pandas as pd

from market_regime_detector.common.aws_dynamodb import get_dynamodb_client
from market_regime_detector.common.aws_s3 import get_s3_client
from market_regime_detector.common.feature_config import Axis
from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.model_registry import get_model, get_production_model_by_name
from market_regime_detector.common.scaling import apply_scaler
from market_regime_detector.common.settings import get_settings
from market_regime_detector.prediction.model_loader import load_model_artifact
from market_regime_detector.prediction.predict import (
    DEFAULT_LOOKBACK_DAYS,
    _build_daily_axis_df,
    _fetch_raw_series,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--asof", type=date.fromisoformat, required=True)
    parser.add_argument("--axis", choices=["growth", "inflation"], required=True)
    parser.add_argument(
        "--source", choices=["lake", "live"], default="lake", help="Same meaning as run_prediction."
    )
    args = parser.parse_args()
    if args.model_id is None and args.model_name is None:
        parser.error("either --model-id or --model-name is required")
    return args


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args()

    s3 = get_s3_client()
    dynamodb = get_dynamodb_client()

    if args.model_id is not None:
        item = get_model(dynamodb, settings.dynamodb_table_name, args.model_id)
    else:
        item = get_production_model_by_name(dynamodb, settings.dynamodb_table_name, args.model_name)
    if item is None:
        raise ValueError("model not found")
    model_id = str(item["model_id"])
    model_name = str(item["model_name"])

    artifact = load_model_artifact(s3, settings.s3_bucket_name, model_name, model_id)
    axis = Axis.GROWTH if args.axis == "growth" else Axis.INFLATION
    hmm = artifact.growth if axis == Axis.GROWTH else artifact.inflation
    mean, scale, clip = (
        (artifact.growth_scaler_mean, artifact.growth_scaler_scale, artifact.growth_scaler_clip)
        if axis == Axis.GROWTH
        else (
            artifact.inflation_scaler_mean,
            artifact.inflation_scaler_scale,
            artifact.inflation_scaler_clip,
        )
    )

    start = args.asof - timedelta(days=DEFAULT_LOOKBACK_DAYS)
    raw_by_series = _fetch_raw_series(
        s3,
        settings.s3_bucket_name,
        artifact.feature_config,
        args.source,
        start,
        args.asof,
        args.asof,
    )
    evaluation_dates = pd.date_range(start, args.asof, freq="D")
    daily_df = _build_daily_axis_df(axis, artifact.feature_config, raw_by_series, evaluation_dates)
    if daily_df.empty or daily_df.index[-1].date() != args.asof:
        logger.warning(
            "last available feature row is %s, not the requested --asof %s",
            daily_df.index[-1].date() if not daily_df.empty else None,
            args.asof,
        )
    row = daily_df.iloc[-1]
    observation_date = cast(pd.Timestamp, row.name)
    feature_names = hmm.feature_names
    x = row[feature_names].to_numpy(dtype=float)
    x_std = apply_scaler(x.reshape(1, -1), mean, scale, clip)[0]

    up_state = hmm.up_state
    down_state = 1 - up_state

    def log_density(x_std_i: float, mean_i: float, var_i: float) -> float:
        return float(-0.5 * np.log(2 * np.pi * var_i) - (x_std_i - mean_i) ** 2 / (2 * var_i))

    print(f"model_id={model_id} model_name={model_name} axis={args.axis} asof={args.asof}")
    print(f"observation_date={observation_date.date()}")
    print(f"{'feature':<32}{'raw':>12}{'z':>10}{'LL(UP)':>12}{'LL(DOWN)':>12}{'LLR(UP-DOWN)':>14}")
    total_llr = 0.0
    for i, name in enumerate(feature_names):
        ll_up = log_density(x_std[i], hmm.means[up_state, i], hmm.variances[up_state, i])
        ll_down = log_density(x_std[i], hmm.means[down_state, i], hmm.variances[down_state, i])
        llr = ll_up - ll_down
        total_llr += llr
        print(f"{name:<32}{x[i]:>12.4f}{x_std[i]:>10.3f}{ll_up:>12.3f}{ll_down:>12.3f}{llr:>14.3f}")
    print(f"\nsum of per-feature LLR (UP - DOWN), emission-only: {total_llr:.3f}")
    print(
        "(the actual filtered posterior also folds in the transition matrix and "
        "the previous month's alpha — this is the emission evidence only, see "
        "docstring)"
    )


if __name__ == "__main__":
    main()
