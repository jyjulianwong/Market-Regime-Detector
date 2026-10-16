#!/usr/bin/env python3
"""Promote a registered model to "production", demoting the current production
model (if any) to "archived" in the same atomic DynamoDB transaction.

Usage:
    uv run python scripts/promote_model.py --model-id mrd-20261016-ab12cd3

Also best-effort async-invokes the batch Lambda so the API isn't left serving stale
results — see `training.promote.invoke_batch_lambda_best_effort` for why that step
tolerates the Lambda not existing yet. A thin wrapper around
`market_regime_detector.common.model_registry` and `training.promote` — no logic of
its own.
"""

import argparse
import logging

from market_regime_detector.common.dynamodb import get_dynamodb_client
from market_regime_detector.common.lambda_client import get_lambda_client
from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.model_registry import promote_model
from market_regime_detector.common.settings import get_settings
from market_regime_detector.training.promote import invoke_batch_lambda_best_effort

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Registry model_id to promote.")
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args()

    promote_model(get_dynamodb_client(), settings.dynamodb_table_name, args.model_id)
    logger.info("promoted %s to production", args.model_id)

    invoke_batch_lambda_best_effort(
        get_lambda_client(), settings.batch_lambda_function_name, args.model_id
    )


if __name__ == "__main__":
    main()
