terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # No backend block here: scripts/bootstrap.sh applies this with local state
  # on the first run (the state bucket doesn't exist yet), then migrates the
  # state into that bucket via a generated _backend.tf (gitignored).
}

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-2"
}

variable "aws_account_id" {
  description = "AWS account ID to deploy into — Terraform will refuse to apply if the active credentials resolve to a different account"
  type        = string
}

variable "github_repository" {
  description = "owner/name of the GitHub repository whose Actions workflows may assume the CI role"
  type        = string
  default     = "jyjulianwong/Market-Regime-Detector"
}

variable "project" {
  description = "Short project identifier used to namespace resource names"
  type        = string
  default     = "jyjulianwong-mrd"
}

provider "aws" {
  region              = var.aws_region
  allowed_account_ids = [var.aws_account_id]
}

locals {
  global_prefix = "${var.aws_account_id}-${var.project}" # S3 bucket (globally scoped)
  scoped_prefix = var.project                            # DynamoDB table, IAM role (account-scoped)
}

resource "aws_s3_bucket" "tf_state" {
  bucket = "${local.global_prefix}-terraform-state"

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_s3_bucket_versioning" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tf_state" {
  bucket = aws_s3_bucket.tf_state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tf_state" {
  bucket                  = aws_s3_bucket.tf_state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tf_lock" {
  name         = "${local.scoped_prefix}-terraform-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  lifecycle {
    prevent_destroy = true
  }
}

# ---------------------------------------------------------------------------
# IAM — GitHub Actions role (OIDC)
# ---------------------------------------------------------------------------
# Used for Terraform apply of the main stack plus the deploy steps in
# .github/workflows/deploy-terraform.yml.
#
# Lives here, not in the main stack, so CI can never rewrite its own trust
# policy or permissions — only a local bootstrap apply can. That's also why
# the policy names main-stack resources by ARN pattern rather than by
# reference. The OIDC provider itself is an account-wide singleton owned by
# jyjulianwong/Infrastructure-GitHub's bootstrap.

data "aws_iam_openid_connect_provider" "github" {
  url = "https://token.actions.githubusercontent.com"
}

data "aws_iam_policy_document" "github_actions_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    # Pushes to main (apply) and pull requests (plan) only.
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values = [
        "repo:${var.github_repository}:ref:refs/heads/main",
        "repo:${var.github_repository}:pull_request",
      ]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "${local.scoped_prefix}-github-actions-role"
  assume_role_policy = data.aws_iam_policy_document.github_actions_trust.json
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
    resources = ["arn:aws:ecr:${var.aws_region}:${var.aws_account_id}:repository/${local.scoped_prefix}-batch-lambda"]
  }

  statement {
    sid    = "LambdaCodeUpdate"
    effect = "Allow"
    actions = [
      "lambda:UpdateFunctionCode",
      "lambda:GetFunction",
    ]
    resources = ["arn:aws:lambda:${var.aws_region}:${var.aws_account_id}:function:${local.scoped_prefix}-batch", "arn:aws:lambda:${var.aws_region}:${var.aws_account_id}:function:${local.scoped_prefix}-server"]
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
      "apigateway:*",
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

resource "aws_iam_role_policy" "github_actions" {
  name   = "${local.scoped_prefix}-github-actions-role-policy"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.github_actions.json
}

output "iac_tf_state_bucket_name" {
  value = aws_s3_bucket.tf_state.bucket
}

output "iac_tf_lock_table_name" {
  value = aws_dynamodb_table.tf_lock.name
}

output "github_actions_role_arn" {
  description = "IAM role GitHub Actions assumes via OIDC — stored as GitHub secret AWS_ROLE_ARN by scripts/bootstrap.sh"
  value       = aws_iam_role.github_actions.arn
}
