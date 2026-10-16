from datetime import date, timedelta
from unittest.mock import MagicMock, patch

from market_regime_detector.ingestion import ingest


@patch("market_regime_detector.ingestion.ingest.write_to_lake")
@patch("market_regime_detector.ingestion.ingest.get_s3_client")
@patch("market_regime_detector.ingestion.ingest.build_provider")
def test_ingest_series_fetches_from_provider_and_writes_to_lake(
    mock_build_provider: MagicMock,
    mock_get_s3_client: MagicMock,
    mock_write_to_lake: MagicMock,
) -> None:
    mock_provider = MagicMock()
    mock_build_provider.return_value = mock_provider
    mock_provider.fetch.return_value = "fake-dataframe"
    mock_get_s3_client.return_value = "fake-s3-client"
    start, end, asof = date(2024, 1, 1), date(2024, 2, 1), date(2024, 1, 15)

    ingest.ingest_series("TEST", "fred", start, end, "test-bucket", asof=asof)

    mock_build_provider.assert_called_once_with("fred")
    mock_provider.fetch.assert_called_once_with("TEST", start, end, asof=asof)
    mock_write_to_lake.assert_called_once_with("fake-s3-client", "test-bucket", "fake-dataframe")


@patch("market_regime_detector.ingestion.ingest.get_s3_client")
@patch("market_regime_detector.ingestion.ingest.get_last_ingested_date")
@patch("market_regime_detector.ingestion.ingest.ingest_series")
def test_incremental_uses_last_date_minus_overlap(
    mock_ingest_series: MagicMock,
    mock_get_last_ingested_date: MagicMock,
    mock_get_s3_client: MagicMock,
) -> None:
    mock_get_last_ingested_date.return_value = date(2024, 6, 1)
    end = date(2024, 6, 15)

    ingest.ingest_incremental("TEST", "fred", "test-bucket", end=end, overlap_days=10)

    expected_start = date(2024, 6, 1) - timedelta(days=10)
    mock_ingest_series.assert_called_once_with("TEST", "fred", expected_start, end, "test-bucket")


@patch("market_regime_detector.ingestion.ingest.get_s3_client")
@patch("market_regime_detector.ingestion.ingest.get_last_ingested_date")
@patch("market_regime_detector.ingestion.ingest.ingest_series")
def test_incremental_falls_back_to_long_lookback_when_series_is_new(
    mock_ingest_series: MagicMock,
    mock_get_last_ingested_date: MagicMock,
    mock_get_s3_client: MagicMock,
) -> None:
    mock_get_last_ingested_date.return_value = None
    end = date(2024, 6, 15)

    ingest.ingest_incremental("TEST", "fred", "test-bucket", end=end)

    expected_start = end - timedelta(days=365 * 10)
    mock_ingest_series.assert_called_once_with("TEST", "fred", expected_start, end, "test-bucket")
