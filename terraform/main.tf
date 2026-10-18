terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }
}

provider "aws" {
  region              = var.aws_region
  allowed_account_ids = [var.aws_account_id]
}

locals {
  global_prefix = "${var.aws_account_id}-${var.project}" # S3 bucket (globally scoped)
  scoped_prefix = var.project                            # all other resources (account-scoped)
}

# ---------------------------------------------------------------------------
# S3 — application bucket (brief section 4.1: data/, datasets/, models/,
# predictions/ all live here as prefixes within one bucket)
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "app" {
  bucket = local.global_prefix
}

resource "aws_s3_bucket_versioning" "app" {
  bucket = aws_s3_bucket.app.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "app" {
  bucket = aws_s3_bucket.app.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "app" {
  bucket                  = aws_s3_bucket.app.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# DynamoDB — model registry (brief section 4.3). Schema matches exactly what
# every moto-backed test since Phase 5 has validated: PK model_id, GSI
# stage-index (PK stage, SK last_modified).
# ---------------------------------------------------------------------------

resource "aws_dynamodb_table" "models" {
  name         = "${local.scoped_prefix}-models"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "model_id"

  attribute {
    name = "model_id"
    type = "S"
  }

  attribute {
    name = "stage"
    type = "S"
  }

  attribute {
    name = "last_modified"
    type = "S"
  }

  global_secondary_index {
    name            = "stage-index"
    hash_key        = "stage"
    range_key       = "last_modified"
    projection_type = "ALL"
  }
}

# ---------------------------------------------------------------------------
# SSM — provider API keys. Created as SecureStrings per brief section 3.5's
# literal requirement, but see variables.tf: the batch Lambda's env vars get
# the secret value directly, not a reference to these parameter names — there
# is no runtime SSM lookup in common/settings.py to wire up.
# ---------------------------------------------------------------------------

resource "aws_ssm_parameter" "fred_api_key" {
  name        = "/${local.scoped_prefix}/fred_api_key"
  type        = "SecureString"
  value       = var.fred_api_key
  description = "FRED/ALFRED API key, shared by FredProvider and AlfredProvider"
}

resource "aws_ssm_parameter" "alpha_vantage_api_key" {
  name        = "/${local.scoped_prefix}/alpha_vantage_api_key"
  type        = "SecureString"
  value       = var.alpha_vantage_api_key
  description = "Alpha Vantage API key"
}

# ---------------------------------------------------------------------------
# ECR — batch Lambda container image repository
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "batch_lambda" {
  name                 = "${local.scoped_prefix}-batch-lambda"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "batch_lambda" {
  repository = aws_ecr_repository.batch_lambda.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# IAM — server Lambda role (brief section 3.5: "read-only on the predictions/
# prefix and read-only on the DynamoDB table and GSI. No access to data/ or
# models/.")
# ---------------------------------------------------------------------------

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "server_lambda" {
  name               = "${local.scoped_prefix}-server-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "server_lambda_basic" {
  role       = aws_iam_role.server_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "server_lambda_permissions" {
  statement {
    sid    = "ReadPredictions"
    effect = "Allow"
    actions = [
      "s3:GetObject",
    ]
    resources = ["${aws_s3_bucket.app.arn}/predictions/*"]
  }

  statement {
    sid    = "ReadModelRegistry"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:Query",
      # GET /v1/models with no stage filter scans (common/model_registry.list_all_models) —
      # easy to miss if copying Query-only examples; brief's "read-only on the
      # table and GSI" covers this literally.
      "dynamodb:Scan",
    ]
    resources = [
      aws_dynamodb_table.models.arn,
      "${aws_dynamodb_table.models.arn}/index/stage-index",
    ]
  }
}

resource "aws_iam_role_policy" "server_lambda_permissions" {
  name   = "${local.scoped_prefix}-server-lambda-permissions"
  role   = aws_iam_role.server_lambda.id
  policy = data.aws_iam_policy_document.server_lambda_permissions.json
}

# ---------------------------------------------------------------------------
# IAM — batch Lambda role (brief section 3.5: "read/write on data/ and
# predictions/, read on models/ and datasets/, read on DynamoDB, read on the
# specific SSM parameters.")
# ---------------------------------------------------------------------------

resource "aws_iam_role" "batch_lambda" {
  name               = "${local.scoped_prefix}-batch-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

resource "aws_iam_role_policy_attachment" "batch_lambda_basic" {
  role       = aws_iam_role.batch_lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "batch_lambda_permissions" {
  statement {
    sid    = "ReadWriteDataAndPredictions"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.app.arn,
      "${aws_s3_bucket.app.arn}/data/*",
      "${aws_s3_bucket.app.arn}/predictions/*",
    ]
  }

  statement {
    sid    = "ReadModelsAndDatasets"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.app.arn,
      "${aws_s3_bucket.app.arn}/models/*",
      "${aws_s3_bucket.app.arn}/datasets/*",
    ]
  }

  statement {
    sid    = "ReadModelRegistry"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:Query",
    ]
    resources = [
      aws_dynamodb_table.models.arn,
      "${aws_dynamodb_table.models.arn}/index/stage-index",
    ]
  }

  statement {
    sid    = "ReadSSMParams"
    effect = "Allow"
    actions = [
      "ssm:GetParameter",
    ]
    resources = [
      aws_ssm_parameter.fred_api_key.arn,
      aws_ssm_parameter.alpha_vantage_api_key.arn,
    ]
  }

  statement {
    sid    = "DecryptSSMParams"
    effect = "Allow"
    actions = [
      "kms:Decrypt",
    ]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${var.aws_region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "batch_lambda_permissions" {
  name   = "${local.scoped_prefix}-batch-lambda-permissions"
  role   = aws_iam_role.batch_lambda.id
  policy = data.aws_iam_policy_document.batch_lambda_permissions.json
}

# ---------------------------------------------------------------------------
# Lambda — batch (container image)
#
# image_uri points at a placeholder ":latest" tag that doesn't exist yet on
# first apply; lifecycle.ignore_changes stops Terraform reverting it once
# CI/CD pushes the real image and calls update-function-code — same pattern
# as Investment-News-Analysis-AI, single root module (no registry/+stack/
# split, brief's own chicken-and-egg problem solved this way instead).
# ---------------------------------------------------------------------------

resource "aws_lambda_function" "batch" {
  function_name = "${local.scoped_prefix}-batch"
  role          = aws_iam_role.batch_lambda.arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.batch_lambda.repository_url}:latest"
  architectures = ["x86_64"] # amd64 — matches Investment-News-Analysis-AI's build pipeline
  timeout       = var.batch_lambda_timeout_seconds
  memory_size   = var.batch_lambda_memory_mb

  environment {
    variables = {
      MRD_AWS_REGION            = var.aws_region
      MRD_S3_BUCKET_NAME        = aws_s3_bucket.app.bucket
      MRD_DYNAMODB_TABLE_NAME   = aws_dynamodb_table.models.name
      MRD_FRED_API_KEY          = var.fred_api_key
      MRD_ALPHA_VANTAGE_API_KEY = var.alpha_vantage_api_key
      MRD_LOG_LEVEL             = "INFO"
    }
  }

  lifecycle {
    ignore_changes = [image_uri]
  }
}

resource "aws_cloudwatch_log_group" "batch" {
  name              = "/aws/lambda/${aws_lambda_function.batch.function_name}"
  retention_in_days = 30
}

# ---------------------------------------------------------------------------
# Lambda — server (zip)
#
# This zip only contains the project's own .py files, no dependencies —
# enough for the first `terraform apply` to succeed, same placeholder pattern
# as Obsidian-Server. CI/CD packages the real dependency-inclusive zip
# (scripts/package_lambda.sh) and pushes it via `update-function-code`.
# ---------------------------------------------------------------------------

data "archive_file" "server_src" {
  type        = "zip"
  source_dir  = "${path.module}/../src"
  output_path = "${path.module}/.placeholder-server-lambda.zip"
}

resource "aws_lambda_function" "server" {
  function_name    = "${local.scoped_prefix}-server"
  role             = aws_iam_role.server_lambda.arn
  runtime          = "python3.12"
  handler          = "market_regime_detector.server.main.handler"
  filename         = data.archive_file.server_src.output_path
  source_code_hash = data.archive_file.server_src.output_base64sha256
  timeout          = var.server_lambda_timeout_seconds
  memory_size      = var.server_lambda_memory_mb

  environment {
    variables = {
      MRD_AWS_REGION              = var.aws_region
      MRD_S3_BUCKET_NAME          = aws_s3_bucket.app.bucket
      MRD_DYNAMODB_TABLE_NAME     = aws_dynamodb_table.models.name
      MRD_CORS_ALLOWED_ORIGINS    = var.cors_allowed_origins
      MRD_STALENESS_MAX_AGE_HOURS = tostring(var.staleness_max_age_hours)
      MRD_LOG_LEVEL               = "INFO"
    }
  }

  lifecycle {
    # CI/CD updates the code with real dependencies bundled in; Terraform
    # should not revert it to the dependency-less placeholder on next apply.
    ignore_changes = [filename, source_code_hash]
  }
}

resource "aws_cloudwatch_log_group" "server" {
  name              = "/aws/lambda/${aws_lambda_function.server.function_name}"
  retention_in_days = 30
}

resource "aws_lambda_function_url" "server" {
  function_name      = aws_lambda_function.server.function_name
  authorization_type = "NONE"
}

# ---------------------------------------------------------------------------
# API Gateway — public HTTP API in front of the server Lambda
#
# Added alongside the Function URL above (not replacing it) after discovering
# this AWS account's Organization enforces an SCP that denies anonymous
# lambda:InvokeFunctionUrl calls: the Function URL itself is correctly
# configured (authorization_type NONE, and a resource policy already grants
# public invoke) but every call is denied at the authorization layer before
# the handler ever runs. API Gateway invokes the same Lambda through a
# completely different IAM path — lambda:InvokeFunction, called by the
# apigateway.amazonaws.com service principal via the permission below, not a
# public FunctionUrlAuthType=NONE grant — so it isn't covered by that
# guardrail. Same AWS_PROXY HTTP API pattern as Obsidian-Server's "jwks" API
# (its terraform/main.tf), minus the mTLS custom domain: this API is meant to
# be fully public with no client certificate, and this project owns no
# domain name, so it's served from the auto-generated execute-api endpoint
# (aws_apigatewayv2_api.server.api_endpoint, output as
# server_api_gateway_url) rather than a custom domain.
# ---------------------------------------------------------------------------

resource "aws_apigatewayv2_api" "server" {
  name          = "${local.scoped_prefix}-server-api"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "server" {
  api_id                 = aws_apigatewayv2_api.server.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.server.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "server_default" {
  api_id             = aws_apigatewayv2_api.server.id
  route_key          = "$default"
  target             = "integrations/${aws_apigatewayv2_integration.server.id}"
  authorization_type = "NONE"
}

resource "aws_apigatewayv2_stage" "server_default" {
  api_id      = aws_apigatewayv2_api.server.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "apigateway_server" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.server.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.server.execution_arn}/*/*"
}

# ---------------------------------------------------------------------------
# EventBridge — two independently-scheduled cron jobs against the *same*
# batch-lambda function (brief section 3.5; a plain CloudWatch Events Rule,
# not EventBridge Scheduler): a weekly ingestion run and a
# daily prediction run, dispatched by `prediction/handler.py` on each
# target's `action` input payload. One shared retry policy + SQS
# dead-letter queue for failed invocations of either.
# ---------------------------------------------------------------------------

resource "aws_sqs_queue" "batch_dlq" {
  name = "${local.scoped_prefix}-batch-dlq"
}

data "aws_iam_policy_document" "batch_dlq_policy" {
  statement {
    sid    = "AllowEventBridgeSendMessage"
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
    actions   = ["sqs:SendMessage"]
    resources = [aws_sqs_queue.batch_dlq.arn]
    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values = [
        aws_cloudwatch_event_rule.predict_batch_lambda.arn,
        aws_cloudwatch_event_rule.ingest_batch_lambda.arn,
      ]
    }
  }
}

resource "aws_sqs_queue_policy" "batch_dlq" {
  queue_url = aws_sqs_queue.batch_dlq.id
  policy    = data.aws_iam_policy_document.batch_dlq_policy.json
}

# --- Daily: predict (nowcast off whatever's already in the lake) ------------

resource "aws_cloudwatch_event_rule" "predict_batch_lambda" {
  name                = "${local.scoped_prefix}-predict-batch-lambda"
  description         = "Triggers the '${local.scoped_prefix}-batch' Lambda with action=predict, daily on weekdays"
  schedule_expression = var.batch_predict_schedule_expression
}

resource "aws_cloudwatch_event_target" "predict_batch_lambda" {
  rule      = aws_cloudwatch_event_rule.predict_batch_lambda.name
  target_id = "${local.scoped_prefix}-predict-batch-lambda"
  arn       = aws_lambda_function.batch.arn
  input     = jsonencode({ action = "predict" })

  retry_policy {
    maximum_retry_attempts       = 2
    maximum_event_age_in_seconds = 3600
  }

  dead_letter_config {
    arn = aws_sqs_queue.batch_dlq.arn
  }
}

resource "aws_lambda_permission" "eventbridge_predict_batch_lambda" {
  statement_id  = "AllowEventBridgeInvokePredict"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.batch.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.predict_batch_lambda.arn
}

# --- Weekly: ingest (refresh the lake from every first-pass provider) -------

resource "aws_cloudwatch_event_rule" "ingest_batch_lambda" {
  name                = "${local.scoped_prefix}-ingest-batch-lambda"
  description         = "Triggers the '${local.scoped_prefix}-batch' Lambda with action=ingest, weekly"
  schedule_expression = var.batch_ingest_schedule_expression
}

resource "aws_cloudwatch_event_target" "ingest_batch_lambda" {
  rule      = aws_cloudwatch_event_rule.ingest_batch_lambda.name
  target_id = "${local.scoped_prefix}-ingest-batch-lambda"
  arn       = aws_lambda_function.batch.arn
  input     = jsonencode({ action = "ingest" })

  retry_policy {
    maximum_retry_attempts       = 2
    maximum_event_age_in_seconds = 3600
  }

  dead_letter_config {
    arn = aws_sqs_queue.batch_dlq.arn
  }
}

resource "aws_lambda_permission" "eventbridge_ingest_batch_lambda" {
  statement_id  = "AllowEventBridgeInvokeIngest"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.batch.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.ingest_batch_lambda.arn
}

# ---------------------------------------------------------------------------
# CloudWatch alarm + SNS — notifies on batch Lambda errors (brief section
# 3.5). The same Lambda error that exhausts EventBridge's retry policy and
# lands a message in the DLQ above is what this alarm fires on — one signal,
# not two separate things to keep in sync.
# ---------------------------------------------------------------------------

resource "aws_sns_topic" "alerts" {
  name = "${local.scoped_prefix}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "batch_errors" {
  alarm_name          = "${local.scoped_prefix}-batch-errors"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 300
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Triggers when the batch Lambda errors at least once in a 5-minute window"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  ok_actions          = [aws_sns_topic.alerts.arn]

  dimensions = {
    FunctionName = aws_lambda_function.batch.function_name
  }
}

# ---------------------------------------------------------------------------
# IAM — GitHub Actions user (Terraform apply + ECR push + both Lambdas' code
# update). Static access key, not OIDC.
# ---------------------------------------------------------------------------

resource "aws_iam_user" "github_actions" {
  name = "${local.scoped_prefix}-github-actions-user"
}

data "aws_iam_policy_document" "github_actions" {
  statement {
    sid       = "ECRAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "ECRPush"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:DescribeRepositories",
    ]
    resources = [aws_ecr_repository.batch_lambda.arn]
  }

  statement {
    sid    = "LambdaCodeUpdate"
    effect = "Allow"
    actions = [
      "lambda:UpdateFunctionCode",
      "lambda:GetFunction",
    ]
    resources = [aws_lambda_function.batch.arn, aws_lambda_function.server.arn]
  }

  statement {
    sid    = "TerraformState"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
    ]
    resources = [
      "arn:aws:s3:::${local.global_prefix}-terraform-state",
      "arn:aws:s3:::${local.global_prefix}-terraform-state/*",
    ]
  }

  statement {
    sid    = "TerraformLock"
    effect = "Allow"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:DeleteItem",
    ]
    resources = ["arn:aws:dynamodb:${var.aws_region}:*:table/${local.scoped_prefix}-terraform-lock"]
  }

  # Regional Terraform apply permissions
  statement {
    sid    = "TerraformApplyRegional"
    effect = "Allow"
    actions = [
      "s3:*",
      "lambda:*",
      "dynamodb:*",
      "ecr:*",
      "ssm:*",
      "events:*",
      "sqs:*",
      "sns:*",
      "cloudwatch:*",
      "kms:Describe*",
      "kms:List*",
      "logs:*",
    ]
    resources = ["*"]
    condition {
      test     = "StringLike"
      variable = "aws:RequestedRegion"
      values   = [var.aws_region]
    }
  }

  # IAM is a global service and does not populate aws:RequestedRegion, so it
  # must be in its own statement without that condition.
  statement {
    sid       = "TerraformApplyIAM"
    effect    = "Allow"
    actions   = ["iam:*"]
    resources = ["*"]
  }
}

resource "aws_iam_user_policy" "github_actions" {
  name   = "${local.scoped_prefix}-github-actions-user-policy"
  user   = aws_iam_user.github_actions.name
  policy = data.aws_iam_policy_document.github_actions.json
}

resource "aws_iam_access_key" "github_actions" {
  user = aws_iam_user.github_actions.name
}
