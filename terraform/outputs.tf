output "app_bucket_name" {
  description = "S3 bucket holding data/, datasets/, models/, and predictions/ (brief section 4.1)"
  value       = aws_s3_bucket.app.bucket
}

output "dynamodb_table_name" {
  description = "Model registry table name"
  value       = aws_dynamodb_table.models.name
}

output "ecr_repository_url" {
  description = "ECR repository URL for the batch Lambda container image"
  value       = aws_ecr_repository.batch_lambda.repository_url
}

output "batch_lambda_function_name" {
  description = "Batch Lambda function name — for CI/CD update-function-code and manual invokes"
  value       = aws_lambda_function.batch.function_name
}

output "server_lambda_function_name" {
  description = "Server Lambda function name — for CI/CD update-function-code"
  value       = aws_lambda_function.server.function_name
}

output "server_function_url" {
  description = "Public Lambda Function URL for the read-only API (brief section 3.5's ASSUMPTION, resolved) — currently blocked by an org-level SCP denying anonymous lambda:InvokeFunctionUrl; use server_api_gateway_url instead until that's resolved"
  value       = aws_lambda_function_url.server.function_url
}

output "server_api_gateway_url" {
  description = "Public API Gateway URL for the read-only API — added after the Function URL above was found blocked by an org-level SCP (see terraform/main.tf's comment above aws_apigatewayv2_api.server); client/index.html's API_BASE should point here"
  value       = aws_apigatewayv2_api.server.api_endpoint
}

output "batch_dlq_url" {
  description = "SQS dead-letter queue URL for failed batch Lambda invocations"
  value       = aws_sqs_queue.batch_dlq.url
}

output "alerts_topic_arn" {
  description = "SNS topic ARN alert_email is subscribed to"
  value       = aws_sns_topic.alerts.arn
}

output "github_actions_access_key_id" {
  description = "AWS access key ID for GitHub Actions — store as GitHub secret AWS_ACCESS_KEY_ID"
  value       = aws_iam_access_key.github_actions.id
}

output "github_actions_secret_access_key" {
  description = "AWS secret access key for GitHub Actions — store as GitHub secret AWS_SECRET_ACCESS_KEY"
  value       = aws_iam_access_key.github_actions.secret
  sensitive   = true
}
