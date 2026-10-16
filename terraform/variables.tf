variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-2"
}

variable "aws_account_id" {
  description = "AWS account ID to deploy into — Terraform will refuse to apply if the active credentials resolve to a different account"
  type        = string
}

variable "project" {
  description = "Short project identifier used to namespace resource names"
  type        = string
  default     = "jyjulianwong-mrd"
}

# ---------------------------------------------------------------------------
# Secrets — injected via GitHub Actions secrets, never stored in the repo.
# Terraform still creates SSM SecureString parameters from these (brief
# section 3.5's literal requirement), but the batch Lambda's own env vars get
# the value directly from the same variable — common/settings.py (built in
# Phase 1) reads plain MRD_-prefixed env vars, not an SSM parameter name, so
# there is no runtime SSM lookup in the application code to wire up here.
# ---------------------------------------------------------------------------

variable "fred_api_key" {
  description = "FRED/ALFRED API key, shared by FredProvider and AlfredProvider"
  type        = string
  sensitive   = true
}

variable "alpha_vantage_api_key" {
  description = "Alpha Vantage API key (AlphaVantageProvider — not wired to any first-pass series yet)"
  type        = string
  sensitive   = true
}

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------

variable "cors_allowed_origins" {
  description = "Comma-separated exact origins allowed to call the API via CORS (a future GitHub Pages UI), e.g. https://jyjulianwong.github.io. Empty disables cross-origin access entirely."
  type        = string
  default     = ""
}

variable "staleness_max_age_hours" {
  description = "How old a prediction's generated_at can be before the API flags it stale at request time (brief section 3.4)"
  type        = number
  default     = 36
}

# ---------------------------------------------------------------------------
# Schedule and alerting
# ---------------------------------------------------------------------------

variable "batch_schedule_expression" {
  description = "EventBridge Rule cron expression for the daily batch Lambda trigger — 06:30 UTC, weekdays (after the prior US session's data is available)"
  type        = string
  default     = "cron(30 6 ? * MON-FRI *)"
}

variable "alert_email" {
  description = "Email address subscribed to the SNS topic for batch Lambda failure alarms and DLQ messages"
  type        = string
  default     = "julianwong2000@gmail.com"
}

# ---------------------------------------------------------------------------
# Lambda sizing — judgment calls (brief doesn't specify): the batch job does a
# handful of provider fetches plus a NumPy forward filter over small arrays,
# nothing like Investment-News-Analysis-AI's LLM-calling Lambda (900s/1024MB);
# the API Lambda is read-only with no inference, same ballpark as
# Obsidian-Server's auth Lambda.
# ---------------------------------------------------------------------------

variable "batch_lambda_timeout_seconds" {
  description = "Batch Lambda timeout — well under the 15-minute Lambda cap (brief section 3.5)"
  type        = number
  default     = 600
}

variable "batch_lambda_memory_mb" {
  type    = number
  default = 512
}

variable "api_lambda_timeout_seconds" {
  type    = number
  default = 10
}

variable "api_lambda_memory_mb" {
  type    = number
  default = 256
}
