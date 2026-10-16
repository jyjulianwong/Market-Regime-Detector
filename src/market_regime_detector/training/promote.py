import json
import logging
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

if TYPE_CHECKING:
    # boto3-stubs is dev-only (see pyproject.toml) — never installed in a deployed
    # Lambda, so this import must not run at runtime, only during type-checking.
    from mypy_boto3_lambda.client import LambdaClient

logger = logging.getLogger(__name__)


def invoke_batch_lambda_best_effort(
    client: "LambdaClient", function_name: str, model_id: str
) -> None:
    """Async-invokes the batch Lambda **with `model_id`** for the newly promoted
    model (brief section 3.7: "async-invoke the batch Lambda with that model_id"),
    so the API isn't left serving stale results until the next scheduled run.
    `model_id` is passed as the Lambda event payload — `prediction/handler.py`'s
    `event.get("model_id")` reads it back.

    Best-effort and deliberately swallows failures: the batch Lambda doesn't exist
    until Phase 8's Terraform deploys it (and `function_name` is `""` until then,
    brief section 6's settings pattern), and this repo has no AWS infra at all yet.
    The DynamoDB promotion this accompanies is the actual invariant that must
    succeed — by the time this runs, it already has — so a missing function,
    permissions issue, or transient AWS error here is logged and never raised.
    """
    if not function_name:
        logger.warning("no batch Lambda function name configured — skipping invoke")
        return
    try:
        client.invoke(
            FunctionName=function_name,
            InvocationType="Event",
            Payload=json.dumps({"model_id": model_id}).encode(),
        )
    except ClientError as e:
        logger.warning("could not invoke batch Lambda %r: %s", function_name, e)
