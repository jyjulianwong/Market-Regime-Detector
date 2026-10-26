#!/usr/bin/env python3
"""Score a named `FeatureConfig` (default `v1`, see `common.feature_config.FEATURE_CONFIGS`)
with `training.build_model.DEFAULT_HYPERPARAMETERS` against four causal baselines,
writing `evaluation.json` and `evaluation.md` to `--output-dir`.

Usage:
    uv run python scripts/evaluate_model.py --start 2000-01-01 --output-dir ./eval/baseline-v1
    uv run python scripts/evaluate_model.py --start 2000-01-01 --output-dir ./eval/dev \
        --dev-end 2018-12-31 --dev-only
    uv run python scripts/evaluate_model.py --start 2000-01-01 --output-dir ./eval/v2 \
        --feature-config v2

A thin wrapper around `market_regime_detector.training` — no scoring logic of its
own. This never changes features, hyperparameters, or the model itself; it only
measures what's already there.
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd

from market_regime_detector.common.aws_s3 import get_s3_client
from market_regime_detector.common.feature_config import FEATURE_CONFIGS
from market_regime_detector.common.git_info import get_git_sha
from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.settings import get_settings
from market_regime_detector.ingestion.lake import read_series
from market_regime_detector.training.build_model import DEFAULT_HYPERPARAMETERS
from market_regime_detector.training.dataset import build_dataset
from market_regime_detector.training.harness import render_evaluation_markdown, run_evaluation
from market_regime_detector.training.labels import build_label_set

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start", type=date.fromisoformat, required=True, help="Dataset start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, default=None, help="Dataset end date. Defaults to today."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Local directory to write results into."
    )
    parser.add_argument(
        "--dev-end",
        type=date.fromisoformat,
        default=date(2016, 12, 31),
        help="Last date in the dev split; everything after is holdout.",
    )
    parser.add_argument(
        "--dev-only",
        action="store_true",
        help="Omit the holdout split entirely (never compute it) — use while "
        "iterating, look at holdout only once at the end.",
    )
    parser.add_argument(
        "--feature-config",
        default="v1",
        choices=sorted(FEATURE_CONFIGS),
        help="Named FeatureConfig to evaluate (see common.feature_config.FEATURE_CONFIGS).",
    )
    return parser.parse_args()


def _label_series(s3: "S3Client", bucket: str, series_id: str) -> pd.Series:
    """Reads one evaluation-only series from the lake and collapses it to a plain
    date-indexed `pd.Series`. These are non-vintage (brief's evaluation series),
    so "take the last value per date" is a defensive dedupe, not a real vintage
    resolution — `realtime_start` already equals `date` for every row.
    """
    df = read_series(s3, bucket, series_id)
    result: pd.Series = df.sort_values("date").groupby("date")["value"].last()
    return result


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args()
    end = args.end if args.end is not None else date.today()

    feature_config = FEATURE_CONFIGS[args.feature_config]

    s3 = get_s3_client()
    logger.info(
        "building dataset from %s to %s with feature config %s",
        args.start,
        end,
        feature_config.version,
    )
    dataset = build_dataset(
        s3, settings.s3_bucket_name, pd.Timestamp(args.start), pd.Timestamp(end), feature_config
    )
    for warning in cast(list[str], dataset.manifest["warnings"]):
        logger.warning(warning)

    cfnai = _label_series(s3, settings.s3_bucket_name, "CFNAIMA3")
    cpi = _label_series(s3, settings.s3_bucket_name, "CPIAUCSL")
    usrec = _label_series(s3, settings.s3_bucket_name, "USREC")
    labels = build_label_set(cfnai, usrec, cpi)

    report = run_evaluation(
        dataset.growth_df,
        dataset.inflation_df,
        labels,
        feature_config,
        DEFAULT_HYPERPARAMETERS,
        dev_end=pd.Timestamp(args.dev_end),
        include_holdout=not args.dev_only,
    )

    metadata = {
        "dataset_id": dataset.manifest["dataset_id"],
        "feature_config_version": dataset.manifest["feature_config_version"],
        "feature_config_hash": dataset.manifest["feature_config_hash"],
        "git_sha": get_git_sha(),
        "date_start": dataset.manifest["date_start"],
        "date_end": dataset.manifest["date_end"],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=str))
    (args.output_dir / "evaluation.md").write_text(render_evaluation_markdown(report, metadata))
    logger.info("evaluation written to %s", args.output_dir)


if __name__ == "__main__":
    main()
