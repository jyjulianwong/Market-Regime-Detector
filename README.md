# Market Regime Detector

Nowcasts Bridgewater All Weather macro quadrants (growth × inflation) from a daily
batch pipeline of two independent Gaussian HMMs, served by a read-only API.

## Architecture

```
EventBridge Rule "ingest" (Mon–Fri, 04:00 UTC)
  └─► Batch Lambda, event {"action": "ingest"}
        incremental ingestion (FRED/ALFRED/yfinance/Alpha Vantage)
          └─► S3 app bucket: data/asset=<id>/year=<yyyy>/data.parquet

EventBridge Rule "predict" (Mon–Fri, 06:00 UTC)
  └─► Batch Lambda, event {"action": "predict"}
        ├─ reads the production model from the DynamoDB registry
        ├─ loads that model's artifacts from S3: models/<model_id>/
        ├─ causal NumPy forward filter (no hmmlearn in this image)
        │    — nowcasts off whatever's already in the lake; never ingests
        └─► S3 app bucket: predictions/model_id=<id>/{latest,history}.json
                           and snapshots/asof=<date>.json
                                                      │
Lambda Function URL (public, no auth)                 │
  └─► Server Lambda (zip) — read-only, no inference ─────┘
        reads the production model from DynamoDB, reads predictions/ from S3

Local only (never deployed): scripts/train_model.py, publish_model.py,
promote_model.py — fit the HMMs, publish to models/<id>/ + the registry,
promote to "production".
```

| Piece | Role |
|---|---|
| S3 app bucket | `data/`, `datasets/`, `models/`, `predictions/` — one bucket, versioned |
| DynamoDB `models` table | Model registry: `model_id` → stage (`development`/`staging`/`production`/`archived`) |
| ECR repository | Batch Lambda's container image |
| Batch Lambda (container image, amd64) | One function, two actions: `ingest` (daily) and `predict` (daily) — the only component that imports `hmmlearn`-free inference code; never imports `training/` |
| Server Lambda (zip) | Read-only `/v1` endpoints — never imports `pandas`/`numpy`/`hmmlearn` |
| 2x EventBridge Rule + shared SQS DLQ | Daily ingest trigger, daily predict trigger, retries, dead-letter queue for failed invocations of either |
| CloudWatch alarm + SNS | Emails `alert_email` on batch Lambda errors |
| SSM parameters | `SecureString` copies of the provider API keys (see note below) |
| GitHub Actions IAM user | Static access key — Terraform apply, ECR push, both Lambdas' code updates |

## Repository layout

```
.
├── src/market_regime_detector/   # application code
├── scripts/              # ingest_data.py, train_model.py, publish_model.py,
│                          # promote_model.py, run_prediction.py,
│                          # package_lambda.sh, bump_calver.py
├── docker/
│   └── batch-lambda.Dockerfile
├── terraform/
│   ├── bootstrap/         # One-time state infrastructure (run manually once)
│   └── *.tf               # Main infrastructure (deployed via CI/CD after first apply)
└── .github/workflows/     # deploy-terraform.yml
```

---

## Prerequisites

- **AWS CLI**, configured (`aws configure`) with credentials for the account you're
  deploying into.
- **Terraform** ≥ 1.6.
- **Docker** (with `buildx`), for the batch Lambda's container image.
- **uv** (Python package manager — `pip install uv`).
- A **FRED/ALFRED API key** ([free, instant signup](https://fred.stlouisfed.org/docs/api/api_key.html))
  and, optionally, an **Alpha Vantage API key** (not wired to any first-pass series
  yet, but the provider exists).

---

## Initial setup

Everything below is a **one-time** setup. After it's done, pushes to `main` that
touch `terraform/`, `docker/`, `src/`, `scripts/`, `pyproject.toml`, or `uv.lock`
redeploy automatically via GitHub Actions.

### 1. Find your AWS account ID

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo $ACCOUNT_ID
```

### 2. Bootstrap Terraform state infrastructure

```bash
cd terraform/bootstrap
terraform init
terraform apply -var="aws_account_id=$ACCOUNT_ID"
# Creates: 
# - ${ACCOUNT_ID}-jyjulianwong-mrd-terraform-state (S3)
# - jyjulianwong-mrd-terraform-lock (DynamoDB)
cd ../..
```

### 3. First-time Terraform apply

The batch Lambda is a container image — AWS validates that the image actually
exists in ECR *at function-creation time*, so (same as
`Investment-News-Analysis-AI`, which hits the exact same constraint) the very
first apply has to happen in two steps: create the ECR repository on its own,
push an initial image into it, then apply everything else. Every apply *after*
this one is a single `terraform apply` — `lifecycle { ignore_changes =
[image_uri] }` on the Lambda resource only protects against Terraform *reverting*
the image CI/CD pushes later; it doesn't help this first bootstrap.

**Phase 1 — create the ECR repository only:**

```bash
cd terraform
terraform init \
  -backend-config="bucket=${ACCOUNT_ID}-jyjulianwong-mrd-terraform-state" \
  -backend-config="dynamodb_table=jyjulianwong-mrd-terraform-lock"

terraform apply \
  -target=aws_ecr_repository.batch_lambda \
  -var="aws_account_id=$ACCOUNT_ID" \
  -var="fred_api_key=$MRD_FRED_API_KEY" \
  -var="alpha_vantage_api_key=$MRD_ALPHA_VANTAGE_API_KEY"
```

**Phase 2 — build and push an initial batch Lambda image:**

```bash
ECR_URL=$(terraform output -raw ecr_repository_url)

aws ecr get-login-password --region eu-west-2 | \
  docker login --username AWS --password-stdin "$ECR_URL"

docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --push \
  -f ../docker/batch-lambda.Dockerfile \
  -t "$ECR_URL:latest" \
  ..
```

**Phase 3 — complete the full apply:**

```bash
terraform apply \
  -var="aws_account_id=$ACCOUNT_ID" \
  -var="fred_api_key=$MRD_FRED_API_KEY" \
  -var="alpha_vantage_api_key=$MRD_ALPHA_VANTAGE_API_KEY"
```

This creates everything else: the S3 app bucket, the DynamoDB model registry, the
SSM parameters, both IAM roles, the batch Lambda (now pointing at the real image
you just pushed), the Server Lambda (a dependency-less placeholder zip — see step 4),
its public Function URL, the two EventBridge Rules (`ingest` daily, `predict`
daily) + shared DLQ, and the CloudWatch alarm + SNS topic.

Optional variables (all have sensible defaults — see `terraform/variables.tf`):
`cors_allowed_origins`, `staleness_max_age_hours`, `batch_predict_schedule_expression`,
`batch_ingest_schedule_expression`, `alert_email`.

### 4. Deploy the real Server Lambda code (manually, for the first time only)

Terraform's own zip for the Server Lambda has no dependencies bundled — just enough
for `terraform apply` to succeed. Build and push the real one once, before CI/CD
exists to do it for you:

```bash
cd ..   # repo root
./scripts/package_lambda.sh
aws lambda update-function-code \
  --function-name "$(terraform -chdir=terraform output -raw server_lambda_function_name)" \
  --zip-file fileb://build/lambda.zip \
  --region eu-west-2
```

### 5. Retrieve generated IAM credentials

```bash
cd terraform
terraform output github_actions_access_key_id
terraform output -raw github_actions_secret_access_key && echo
```

### 6. Add GitHub Actions secrets and variables

In your GitHub repository → **Settings → Secrets and variables → Actions**:

**Secrets:**

| Secret name | Value |
|---|---|
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID |
| `AWS_ACCESS_KEY_ID` | From step 5 |
| `AWS_SECRET_ACCESS_KEY` | From step 5 |
| `FRED_API_KEY` | Your FRED/ALFRED API key |
| `ALPHA_VANTAGE_API_KEY` | Your Alpha Vantage API key |

**Variables** (optional — all have defaults in the workflow if omitted):

| Variable name | Value |
|---|---|
| `CORS_ALLOWED_ORIGINS` | e.g. `https://jyjulianwong.github.io` (empty by default — no cross-origin access until set) |
| `STALENESS_MAX_AGE_HOURS` | Default `36` |
| `ALERT_EMAIL` | Default `julianwong2000@gmail.com` |
| `BATCH_PREDICT_SCHEDULE_EXPRESSION` | Default `cron(0 6 ? * MON-FRI *)` (daily, weekdays) |
| `BATCH_INGEST_SCHEDULE_EXPRESSION` | Default `cron(0 4 ? * MON-FRI *)` (daily, weekdays) |

These are what the workflow reads on every run — see [CI/CD](#cicd) below.

### 7. Confirm the SNS email subscription

AWS emails `alert_email` (default `julianwong2000@gmail.com`) a subscription
confirmation link the first time `terraform apply` creates the SNS topic — click
it, or alarm notifications will never arrive.

### 8. Train and publish a first model, then promote it to production

Everything above deploys infrastructure with **no model in the registry yet** —
the batch Lambda and the API will both correctly 404/log "no production model"
until you do this, once, locally:

```bash
uv sync --all-extras
uv run python scripts/ingest_data.py --start 2000-01-01
uv run python scripts/train_model.py --start 2000-01-01 --output-dir ./model
uv run python scripts/publish_model.py --model-dir ./model --manifest-path ./model/manifest.json --stage staging
uv run python scripts/promote_model.py --model-id <model_id printed above>
```
`--start 2000-01-01` is earlier than every first-pass series actually needs — it's
not an arbitrary round number, it's past the real binding constraint
(`T10YIE`/`DFII10`, 2003) and right at the other (`HG=F`/`GC=F`, 2000-08-30), so
going earlier buys nothing and going later would lose growth-axis history for no
reason:

| Series | Earliest available |
|---|---|
| INDPRO | 1919-01-01 |
| PAYEMS | 1939-01-01 |
| CPILFESL | 1957-01-01 |
| ICSA | 1967-01-07 |
| T10Y2Y | 1976-06-01 |
| DCOILWTICO | 1986-01-02 |
| HG=F, GC=F | 2000-08-30 |
| T10YIE, DFII10 | 2003-01-02 |

`promote_model.py` also best-effort async-invokes the batch Lambda so a first
prediction exists immediately rather than waiting for the next 06:00 UTC run.

### 9. Test it

```bash
curl "$(terraform -chdir=terraform output -raw server_function_url)v1/health"
curl "$(terraform -chdir=terraform output -raw server_function_url)v1/predictions/latest"
```

---

## CI/CD

`.github/workflows/deploy-terraform.yml`:
- **On PR** (touching `.github/**`, `terraform/**`, `docker/**`, `scripts/**`,
  `src/**`, `pyproject.toml`, or `uv.lock`): runs `terraform plan` and posts the
  output as a PR comment.
- **On merge to `main`**: bumps a calendar version and tags it (`version_bump`),
  then — checked out at that tag — runs `terraform apply`, builds and pushes the
  batch Lambda image to ECR, calls `update-function-code` for it, packages the
  real Server Lambda zip, and calls `update-function-code` for that too (`deploy`).

No `ruff`/`mypy`/`pytest`/`terraform validate` gate runs in GitHub Actions —
those are enforced locally via `.pre-commit-config.yaml` (see below); CI only
plans and deploys.

### Versioning

Every deploy to `main` is stamped with a calendar version (`YYYY.MM.DD.MINOR`,
e.g. `2026.10.17.0`) via `scripts/bump_calver.py`: it bumps `version` in the root
`pyproject.toml`, commits, and tags (`v2026.10.17.0`). The `deploy` job then
builds and ships that exact tagged commit — both the batch image and the API
zip — not whatever `main` happens to be at the moment the job runs, so a green
Actions run always corresponds to one, findable, reproducible commit.

### Pre-commit hooks

Install once per clone:

```bash
uv run pre-commit install
```

`.pre-commit-config.yaml` runs on every commit: `uv-lock`/`uv-sort` (keep
`uv.lock` and `pyproject.toml`'s dependency list in sync and sorted),
`ruff`/`ruff-format` (lint and format all Python), `terraform_fmt` (keeps
everything under `terraform/` canonically formatted).

---

## Useful AWS CLI commands

```bash
# Tail batch Lambda logs
aws logs tail /aws/lambda/jyjulianwong-mrd-batch --follow --region eu-west-2

# Tail Server Lambda logs
aws logs tail /aws/lambda/jyjulianwong-mrd-server --follow --region eu-west-2

# Manually trigger a prediction (production model, lake source, latest data)
aws lambda invoke \
  --function-name jyjulianwong-mrd-batch \
  --cli-binary-format raw-in-base64-out \
  --payload '{"action": "predict"}' \
  --region eu-west-2 \
  response.json && cat response.json && echo

# Manually trigger ingestion (every first-pass series)
aws lambda invoke \
  --function-name jyjulianwong-mrd-batch \
  --cli-binary-format raw-in-base64-out \
  --payload '{"action": "ingest"}' \
  --region eu-west-2 \
  response.json && cat response.json && echo

# Check a Lambda's status (works for either function name above)
aws lambda get-function \
  --function-name jyjulianwong-mrd-batch \
  --region eu-west-2 \
  --query "Configuration.{State:State,LastUpdateStatus:LastUpdateStatus}"

# List the model registry
aws dynamodb scan \
  --table-name "$(terraform -chdir=terraform output -raw dynamodb_table_name)" \
  --query "Items[].{model_id:model_id.S,stage:stage.S}"

# Browse the S3 app bucket
BUCKET=$(terraform -chdir=terraform output -raw app_bucket_name)
aws s3 ls "s3://$BUCKET/data/"
aws s3 ls "s3://$BUCKET/models/"
aws s3 ls "s3://$BUCKET/predictions/"

# Check the DLQ for failed batch invocations
aws sqs receive-message --queue-url "$(terraform -chdir=terraform output -raw batch_dlq_url)"

# Update SSM secrets manually (rotate an API key without re-running Terraform)
aws ssm put-parameter \
  --name "/jyjulianwong-mrd/fred_api_key" \
  --value "YOUR_NEW_KEY" \
  --type SecureString \
  --overwrite \
  --region eu-west-2
# Note: this does NOT update the running batch Lambda's env var — that's set
# directly from the Terraform variable, so also re-run
# `terraform apply -var="fred_api_key=YOUR_NEW_KEY" ...` to actually rotate it there.

# Log in to ECR (for a manual docker push)
aws ecr get-login-password --region eu-west-2 | \
  docker login --username AWS --password-stdin \
  "$(terraform -chdir=terraform output -raw ecr_repository_url | sed 's|/.*||')"
```

---

## Daily flow

| Time (UTC) | Event |
|---|---|
| 04:00, Mon–Fri | `ingest` EventBridge Rule invokes the batch Lambda with `{"action": "ingest"}` — incrementally ingests every first-pass series into the lake |
| 06:00, Mon–Fri | `predict` EventBridge Rule invokes the batch Lambda with `{"action": "predict"}` — loads the production model, runs the real recursive causal forward filter over month-end rows only, previews every other day in the lookback window with one `preview_posterior` step off its month's previous month-end (never a daily recursion — see `prediction/predict.py`'s `_axis_posteriors`), writes `predictions/model_id=<id>/{latest,history}.json` and a write-once `snapshots/asof=<date>.json` |
| Any time | `GET /v1/predictions/latest` (and `/history`) on the Server Lambda Function URL serve whatever the last `predict` run wrote |
| On failure | EventBridge retries, then the message lands in the shared SQS DLQ; a CloudWatch alarm on the batch Lambda's `Errors` metric emails `alert_email` |

---

## Local development

```bash
uv sync --all-extras                    # install everything
uv run pytest                           # run tests (with coverage)
uv run ruff check .                     # lint
uv run ruff format .                    # format
uv run mypy --strict src tests scripts  # type-check
uv run pre-commit run --all-files       # everything pre-commit enforces
```

The two commands below need real AWS credentials (and the deployed bucket/table
from above):

```bash
uv run uvicorn market_regime_detector.server.app:app --reload --port 8000
uv run python scripts/run_prediction.py   # defaults to production model, lake source, latest data
```
