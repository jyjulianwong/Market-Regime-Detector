#!/usr/bin/env python3
"""Build a training dataset from the S3 lake, fit both HMMs, and write a local model
directory with metrics and a model card.

Usage:
    uv run python scripts/train_model.py --start 2000-01-01 --end 2026-10-16 --output-dir ./model

A thin wrapper around `market_regime_detector.training` — no logic of its own. The
dataset manifest is written alongside the model files for now; Phase 5's publisher
is responsible for uploading each to its own S3 prefix (`datasets/<id>/` vs.
`models/<version>/`, brief section 4.1).
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd

from market_regime_detector.common.aws_s3 import get_s3_client
from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.settings import get_settings
from market_regime_detector.training.build_model import train_and_save_model
from market_regime_detector.training.dataset import build_dataset

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
        "--output-dir", type=Path, required=True, help="Local directory to write the model into."
    )
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args()
    end = args.end if args.end is not None else date.today()

    s3 = get_s3_client()
    logger.info("building dataset from %s to %s", args.start, end)
    dataset = build_dataset(
        s3, settings.s3_bucket_name, pd.Timestamp(args.start), pd.Timestamp(end)
    )

    logger.info(
        "dataset %s: %d growth rows, %d inflation rows",
        dataset.manifest["dataset_id"],
        len(dataset.growth_df),
        len(dataset.inflation_df),
    )
    for warning in cast(list[str], dataset.manifest["warnings"]):
        logger.warning(warning)

    train_and_save_model(args.output_dir, dataset.growth_df, dataset.inflation_df)

    (args.output_dir / "manifest.json").write_text(
        json.dumps(dataset.manifest, indent=2, default=str)
    )
    logger.info("model written to %s", args.output_dir)


if __name__ == "__main__":
    main()
