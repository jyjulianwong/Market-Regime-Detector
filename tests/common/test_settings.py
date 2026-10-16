import pytest

from market_regime_detector.common.settings import Settings


def test_defaults_load_without_any_env_vars() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.aws_region == "eu-west-2"
    assert settings.s3_bucket_name == "market-regime-detector"
    assert settings.dynamodb_table_name == "market-regime-detector-models"
    assert settings.log_level == "INFO"


def test_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRD_AWS_REGION", "us-east-1")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.aws_region == "us-east-1"
