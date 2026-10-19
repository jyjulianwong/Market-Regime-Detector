from typing import TYPE_CHECKING

import boto3

from market_regime_detector.common.settings import get_settings

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_lambda.client import LambdaClient


def get_lambda_client() -> "LambdaClient":
    """Returns a boto3 Lambda client configured for this project's region.

    Used by `scripts/promote_model.py` to best-effort async-invoke the batch Lambda
    after a promotion (brief section 3.7) — see `training/promote.py`.
    """
    settings = get_settings()
    return boto3.client("lambda", region_name=settings.aws_region)
