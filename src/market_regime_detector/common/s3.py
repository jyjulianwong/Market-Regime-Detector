from typing import TYPE_CHECKING

import boto3

from market_regime_detector.common.settings import get_settings

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_s3.client import S3Client


def get_s3_client() -> "S3Client":
    """Returns a boto3 S3 client configured for this project's region.

    A thin wrapper rather than calling `boto3.client("s3")` directly at every call
    site, so region configuration and (later) retry/timeout tuning live in one place.
    """
    settings = get_settings()
    return boto3.client("s3", region_name=settings.aws_region)
