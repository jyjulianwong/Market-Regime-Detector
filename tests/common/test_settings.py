import pytest
from moto import mock_aws

from market_regime_detector.common.settings import Settings


@mock_aws
def test_defaults_load_without_any_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MRD_S3_BUCKET_NAME", raising=False)  # set by tests/conftest.py

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.aws_region == "eu-west-2"
    # moto's default account ID — the bucket name follows the active credentials.
    assert settings.s3_bucket_name == "123456789012-jyjulianwong-mrd"
    assert settings.dynamodb_table_name == "jyjulianwong-mrd-models"
    assert settings.log_level == "INFO"


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRD_AWS_REGION", "us-east-1")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.aws_region == "us-east-1"


def test_explicit_bucket_name_skips_account_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRD_S3_BUCKET_NAME", "some-bucket")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.s3_bucket_name == "some-bucket"
