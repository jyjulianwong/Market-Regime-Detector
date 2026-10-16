terraform {
  # bucket and dynamodb_table are supplied at init time via -backend-config
  # so the same stack can target different accounts without code changes.
  backend "s3" {
    key     = "prod/terraform.tfstate"
    region  = "eu-west-2"
    encrypt = true
  }
}
