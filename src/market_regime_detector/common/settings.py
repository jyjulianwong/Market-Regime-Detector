import boto3
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Shared runtime configuration for every deployable (API, batch, training scripts).

    Values are read from `MRD_`-prefixed environment variables, falling back to a
    git-ignored `.env` file for local development. Deployed code sets these as Lambda
    environment variables (see terraform/, added in Phase 8); nothing here reads SSM
    directly — that's the caller's job for anything secret.
    """

    model_config = SettingsConfigDict(env_prefix="MRD_", env_file=".env", extra="ignore")

    aws_region: str = "eu-west-2"
    # Match terraform/main.tf's actual deployed names (local.global_prefix /
    # local.scoped_prefix) — these defaults are for local dev only; Terraform
    # always sets the real values as Lambda env vars. Left empty, the bucket name
    # is derived from the AWS account the active credentials belong to (see
    # _default_s3_bucket_name), so local scripts follow whichever account you're
    # logged in to.
    s3_bucket_name: str = ""
    dynamodb_table_name: str = "jyjulianwong-mrd-models"
    log_level: str = "INFO"

    # Shared by FredProvider and AlfredProvider — both are the same underlying
    # FRED/ALFRED API, just different fredapi query methods.
    fred_api_key: str = ""
    alpha_vantage_api_key: str = ""

    # Set by Terraform (Phase 8) once the batch Lambda exists. Empty until then —
    # scripts/promote_model.py's Lambda invoke is best-effort and tolerates this.
    batch_lambda_function_name: str = ""

    # server/ only (brief section 3.4). Comma-separated exact origins, no
    # wildcards — same convention as Obsidian-Server/Investment-News-Analysis-AI's
    # server CORS settings. Empty means no cross-origin access until a GitHub
    # Pages origin is configured.
    cors_allowed_origins: str = ""
    # How old `generated_at` can be before a prediction response is flagged stale
    # at request time — distinct from the batch job's own generation-time
    # `is_stale` (brief section 3.3); the server ORs the two together.
    staleness_max_age_hours: float = 36.0

    @model_validator(mode="after")
    def _default_s3_bucket_name(self) -> "Settings":
        if not self.s3_bucket_name:
            sts = boto3.client("sts", region_name=self.aws_region)
            account_id = sts.get_caller_identity()["Account"]
            self.s3_bucket_name = f"{account_id}-jyjulianwong-mrd"
        return self


def get_settings() -> Settings:
    return Settings()
