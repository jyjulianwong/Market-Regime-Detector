#!/usr/bin/env python3
"""Run the prediction batch job locally, with any of its optional parameters.

Usage:
    uv run python scripts/run_prediction.py
    uv run python scripts/run_prediction.py --model-id mrd-20261016-ab12cd3
    uv run python scripts/run_prediction.py --source live --start 2024-01-01 --end 2024-06-01 --asof 2024-06-01

An empty invocation means: production model, lake source, latest data — mirrors
`prediction/handler.py`'s default event. A thin wrapper around
`market_regime_detector.prediction` — no logic of its own. Unlike the Lambda
handler, this does **not** run incremental ingestion first; run
`scripts/ingest_data.py --incremental` separately if you want fresh lake data.
"""

import argparse
import logging
from datetime import date

from market_regime_detector.common.dynamodb import get_dynamodb_client
from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.s3 import get_s3_client
from market_regime_detector.common.settings import get_settings
from market_regime_detector.prediction.outputs import write_prediction_outputs
from market_regime_detector.prediction.predict import run_prediction

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id", default=None, help="Use a specific model instead of production."
    )
    parser.add_argument(
        "--source",
        choices=["lake", "live"],
        default="lake",
        help="'lake' reads the S3 data lake (default); 'live' fetches each series "
        "from its own configured provider, bypassing the lake.",
    )
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument(
        "--asof",
        type=date.fromisoformat,
        default=None,
        help="Point-in-time date. Defaults to today.",
    )
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args()

    s3 = get_s3_client()
    result = run_prediction(
        s3,
        get_dynamodb_client(),
        settings.s3_bucket_name,
        settings.dynamodb_table_name,
        model_id=args.model_id,
        source=args.source,
        start=args.start,
        end=args.end,
        asof=args.asof,
    )
    write_prediction_outputs(s3, settings.s3_bucket_name, result)

    logger.info(
        "prediction written for model %s, observation_date=%s, is_stale=%s",
        result.model_id,
        result.observation_date,
        result.is_stale,
    )


if __name__ == "__main__":
    main()
