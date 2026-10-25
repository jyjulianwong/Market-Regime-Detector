#!/usr/bin/env python3
"""Ingest the brief's first-pass series (or a subset) into the S3 data lake.

Usage:
    uv run python scripts/ingest_data.py --start 2015-01-01 --end 2026-10-16
    uv run python scripts/ingest_data.py --incremental
    uv run python scripts/ingest_data.py --series INDPRO PAYEMS --start 2020-01-01 --end 2024-01-01
    uv run python scripts/ingest_data.py --series CFNAIMA3 CPIAUCSL USREC --start 1990-01-01

A thin wrapper around `market_regime_detector.ingestion.ingest` — no logic of its own.
"""

import argparse
import logging
from datetime import date

from market_regime_detector.common.logging import configure_logging
from market_regime_detector.common.settings import get_settings
from market_regime_detector.ingestion.ingest import ingest_incremental, ingest_series
from market_regime_detector.ingestion.series_catalog import EVALUATION_SERIES, FIRST_PASS_SERIES

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        nargs="+",
        default=None,
        help="Series IDs to ingest (must exist in series_catalog.py's FIRST_PASS_SERIES or "
        "EVALUATION_SERIES). Defaults to all first-pass series — evaluation-only series "
        "(CFNAIMA3, CPIAUCSL, USREC) are never ingested by default, only when named "
        "explicitly, since the weekly production ingest never needs them.",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        help="Start date (YYYY-MM-DD). Ignored with --incremental.",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Ingest from (last lake date - overlap) through --end instead of --start.",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=30,
        help="Overlap window for --incremental, so recent revisions are re-fetched.",
    )
    args = parser.parse_args()
    if not args.incremental and args.start is None:
        parser.error("--start is required unless --incremental is set")
    return args


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args()

    # The default (no --series) stays FIRST_PASS_SERIES only — EVALUATION_SERIES is
    # reachable only by naming it explicitly, so a plain `ingest_data.py --start ...`
    # run (what the weekly production schedule's intent mirrors locally) never
    # pulls in ground-truth label series it has no use for.
    catalog = {spec.series_id: spec for spec in FIRST_PASS_SERIES + EVALUATION_SERIES}
    series_ids: list[str] = (
        args.series if args.series is not None else [spec.series_id for spec in FIRST_PASS_SERIES]
    )

    for series_id in series_ids:
        if series_id not in catalog:
            raise ValueError(
                f"{series_id!r} is not in series_catalog.FIRST_PASS_SERIES or "
                "series_catalog.EVALUATION_SERIES"
            )
        spec = catalog[series_id]

        logger.info("ingesting %s from %s", series_id, spec.source)
        if args.incremental:
            ingest_incremental(
                series_id,
                spec.source,
                settings.s3_bucket_name,
                end=args.end,
                overlap_days=args.overlap_days,
            )
        else:
            end = args.end if args.end is not None else date.today()
            ingest_series(series_id, spec.source, args.start, end, settings.s3_bucket_name)


if __name__ == "__main__":
    main()
