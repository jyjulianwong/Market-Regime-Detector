#!/usr/bin/env bash
# Packages the server Lambda zip (project + `server` extra deps) into build/lambda.zip.
#
# Usage:
#   ./scripts/package_lambda.sh
#
# CI/CD runs this after `terraform apply`, then pushes the result via
# `aws lambda update-function-code --zip-file fileb://build/lambda.zip` — the
# real, dependency-inclusive artifact replacing the placeholder zip Terraform
# creates on first apply (see terraform/main.tf's aws_lambda_function.server).
#
# The first non-Python script in scripts/ — a deliberate structural addition
# (brief section 3.7 only lists Python scripts), matching Obsidian-Server's
# equivalent packaging script.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -rf build/package build/lambda.zip
mkdir -p build/package

uv pip install --target build/package ".[server]"

cd build/package
zip -r -q ../lambda.zip . -x '*.pyc' -x '*/__pycache__/*'
cd ../..

echo "wrote build/lambda.zip"
