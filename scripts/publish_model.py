#!/usr/bin/env python3
"""Publish a local model directory (from scripts/train_model.py) to S3 and the
DynamoDB model registry.

Usage:
    uv run python scripts/publish_model.py --model-dir ./model --manifest-path ./model/manifest.json --model-name allweather-hmm2naive-us
    uv run python scripts/publish_model.py --model-dir ./model --manifest-path ./model/manifest.json --model-name allweather-hmm2naive-us --stage development

`--model-name` is the stable family identity (multi-model support) — e.g.
"allweather-hmm2naive-us" — distinct from the `model_id` this generates, which
also encodes a date and content hash. A thin wrapper around
`market_regime_detector.training.publisher` — no logic of its own.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Literal, get_args

from market_regime_detector.common.aws_dynamodb import get_dynamodb_client
from market_regime_detector.common.aws_s3 import get_s3_client
from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.settings import get_settings
from market_regime_detector.training.publisher import PublishStage, publish_model

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="Local model directory to publish."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to the dataset manifest.json written alongside the model directory.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Stable model family identity, e.g. 'allweather-hmm2naive-us'. Lowercase "
        "alphanumeric segments separated by hyphens.",
    )
    parser.add_argument(
        "--stage",
        choices=get_args(PublishStage),
        default="staging",
        help="Stage to publish under. Never 'production' — only promote_model.py sets that.",
    )
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args()
    stage: Literal["development", "staging"] = args.stage

    manifest = json.loads(args.manifest_path.read_text())

    model_id = publish_model(
        args.model_dir,
        args.model_name,
        manifest,
        get_s3_client(),
        settings.s3_bucket_name,
        get_dynamodb_client(),
        settings.dynamodb_table_name,
        stage=stage,
    )

    logger.info("published %s as %s", model_id, stage)


if __name__ == "__main__":
    main()
