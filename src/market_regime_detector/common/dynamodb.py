from typing import TYPE_CHECKING

import boto3

from market_regime_detector.common.settings import get_settings

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_dynamodb.client import DynamoDBClient
    from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource


def get_dynamodb_resource() -> "DynamoDBServiceResource":
    """Returns a boto3 DynamoDB resource configured for this project's region.

    A thin wrapper rather than calling `boto3.resource("dynamodb")` directly at every
    call site, so region configuration and (later) retry/timeout tuning live in one
    place.
    """
    settings = get_settings()
    return boto3.resource("dynamodb", region_name=settings.aws_region)


def get_dynamodb_client() -> "DynamoDBClient":
    """Returns a boto3 DynamoDB low-level client.

    The registry (`common/model_registry.py`) needs this rather than the resource
    above for `transact_write_items` — promotion's atomicity (brief section 4.3)
    isn't exposed through the resource API.
    """
    settings = get_settings()
    return boto3.client("dynamodb", region_name=settings.aws_region)
