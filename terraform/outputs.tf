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

output "api_lambda_function_name" {
  description = "API Lambda function name — for CI/CD update-function-code"
  value       = aws_lambda_function.api.function_name
}

output "api_function_url" {
  description = "Public Lambda Function URL for the read-only API (brief section 3.5's ASSUMPTION, resolved)"
  value       = aws_lambda_function_url.api.function_url
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
