import numpy as np
import pandas as pd

from market_regime_detector.common.feature_config import (
    Axis,
    FeatureConfig,
    FeatureDefinition,
    Frequency,
)
from market_regime_detector.training.build_model import HMMHyperparameters
from market_regime_detector.training.harness import (
    AXES,
    MODEL_NAMES,
    render_evaluation_markdown,
    run_evaluation,
)
from market_regime_detector.training.labels import LabelSet


def _feature(name: str, axis: Axis, sign: int) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        axis=axis,
        series_id=name.upper(),
        source="fred",
        native_frequency=Frequency.MONTHLY,
        vintage_aware=False,
        publication_lag_days=0,
        transform="identity",
        transform_params={},
        sign=sign,
    )


def _synthetic_two_regime_df(
    seed: int, n_per_block: int = 10, n_blocks: int = 8
) -> tuple[pd.DataFrame, pd.Series]:
    """Reuses `tests/training/test_build_model.py`'s synthetic two-regime pattern:
    alternating blocks of months with mean +2.0 ("up") / -2.0 ("down"), plus the
    ground-truth label that pattern is standing in for (perfectly known here,
    since this is synthetic — unlike `training/labels.py`'s real CFNAI/CPI-derived
    labels, which are themselves noisy proxies).
    """
    rng = np.random.default_rng(seed)
    blocks = []
    block_labels = []
    for block in range(n_blocks):
        is_up = block % 2 == 0
        mean = 2.0 if is_up else -2.0
        blocks.append(rng.normal(loc=mean, scale=0.3, size=(n_per_block, 2)))
        block_labels.extend([1.0 if is_up else 0.0] * n_per_block)
    data = np.concatenate(blocks, axis=0)
    dates = pd.date_range("2000-01-31", periods=len(data), freq="ME")
    axis_df = pd.DataFrame(data, columns=["f1", "f2"], index=dates)
    label = pd.Series(block_labels, index=dates)
    return axis_df, label


def _feature_config() -> FeatureConfig:
    return FeatureConfig(
        version="test",
        features=(
            _feature("f1", Axis.GROWTH, sign=1),
            _feature("f2", Axis.GROWTH, sign=1),
            _feature("f1", Axis.INFLATION, sign=1),
            _feature("f2", Axis.INFLATION, sign=1),
        ),
    )


def _hyperparameters() -> HMMHyperparameters:
    return HMMHyperparameters(
        n_restarts=2, initial_window_months=48, refit_every_months=12, random_state=0
    )


def _label_set(growth_label: pd.Series, inflation_label: pd.Series) -> LabelSet:
    return LabelSet(
        growth=growth_label,
        growth_recession=growth_label,  # reused for the test; semantics don't matter here
        inflation=inflation_label,
        label_lag_months=1,
    )


def test_run_evaluation_report_structure() -> None:
    growth_df, growth_label = _synthetic_two_regime_df(seed=0)
    inflation_df, inflation_label = _synthetic_two_regime_df(seed=1)
    labels = _label_set(growth_label, inflation_label)

    report = run_evaluation(
        growth_df,
        inflation_df,
        labels,
        _feature_config(),
        _hyperparameters(),
        dev_end=growth_df.index[60],
    )

    assert set(report["axes"]) == set(AXES)
    for axis_name in AXES:
        axis_report = report["axes"][axis_name]
        assert set(axis_report["splits"]) == {"dev", "holdout", "all"}
        for split in ("dev", "holdout", "all"):
            split_metrics = axis_report["splits"][split]
            assert set(split_metrics) == set(MODEL_NAMES)
            for model in MODEL_NAMES:
                m = split_metrics[model]
                for key in (
                    "n_months",
                    "log_loss",
                    "brier",
                    "skill_vs_climatology",
                    "skill_vs_climatology_ci_low",
                    "skill_vs_climatology_ci_high",
                    "accuracy",
                    "expected_calibration_error",
                    "calibration_table",
                    "flip_rate_per_year",
                    "mean_regime_duration_months",
                ):
                    assert key in m
        assert set(axis_report["coverage"]) == set(MODEL_NAMES)
        assert set(axis_report["detection_lag_months"]) == set(MODEL_NAMES)
    assert set(report["quadrant_log_loss"]) == {"dev", "holdout", "all"}
    for split_quadrant in report["quadrant_log_loss"].values():
        assert set(split_quadrant) == set(MODEL_NAMES)


def test_run_evaluation_intersection_is_shared_across_models_in_a_split() -> None:
    """Every model's `n_months_in_comparison` for a given axis must be identical —
    it is, by construction, the same intersection index for every model, not a
    per-model figure. `n_months_available` can legitimately differ (e.g. a model
    with skipped folds)."""
    growth_df, growth_label = _synthetic_two_regime_df(seed=0)
    inflation_df, inflation_label = _synthetic_two_regime_df(seed=1)
    labels = _label_set(growth_label, inflation_label)

    report = run_evaluation(
        growth_df,
        inflation_df,
        labels,
        _feature_config(),
        _hyperparameters(),
        dev_end=growth_df.index[60],
    )

    for axis_name in AXES:
        coverage = report["axes"][axis_name]["coverage"]
        n_in_comparison = {coverage[model]["n_months_in_comparison"] for model in MODEL_NAMES}
        assert len(n_in_comparison) == 1
        # And every split's n_months must be <= that shared intersection size.
        shared = next(iter(n_in_comparison))
        for split in ("dev", "holdout", "all"):
            for model in MODEL_NAMES:
                assert report["axes"][axis_name]["splits"][split][model]["n_months"] <= shared


def test_run_evaluation_include_holdout_false_omits_holdout_entirely() -> None:
    growth_df, growth_label = _synthetic_two_regime_df(seed=0)
    inflation_df, inflation_label = _synthetic_two_regime_df(seed=1)
    labels = _label_set(growth_label, inflation_label)

    report = run_evaluation(
        growth_df,
        inflation_df,
        labels,
        _feature_config(),
        _hyperparameters(),
        dev_end=growth_df.index[60],
        include_holdout=False,
    )

    assert report["include_holdout"] is False
    for axis_name in AXES:
        assert set(report["axes"][axis_name]["splits"]) == {"dev", "all"}
    assert set(report["quadrant_log_loss"]) == {"dev", "all"}


def test_render_evaluation_markdown_smoke_test() -> None:
    growth_df, growth_label = _synthetic_two_regime_df(seed=0)
    inflation_df, inflation_label = _synthetic_two_regime_df(seed=1)
    labels = _label_set(growth_label, inflation_label)

    report = run_evaluation(
        growth_df,
        inflation_df,
        labels,
        _feature_config(),
        _hyperparameters(),
        dev_end=growth_df.index[60],
    )
    metadata = {
        "dataset_id": "ds-test",
        "feature_config_version": "test",
        "feature_config_hash": "deadbeef",
        "git_sha": "abc123",
        "date_start": "2000-01-31",
        "date_end": "2006-08-31",
    }

    markdown = render_evaluation_markdown(report, metadata)

    assert "# Evaluation report" in markdown
    assert "## Growth axis" in markdown
    assert "## Inflation axis" in markdown
    assert "## Quadrant-level log loss" in markdown
    assert "## Recession detection lag" in markdown
    assert "## Walk-forward coverage gaps" in markdown
    for model in MODEL_NAMES:
        assert model in markdown


def test_render_evaluation_markdown_without_holdout() -> None:
    growth_df, growth_label = _synthetic_two_regime_df(seed=0)
    inflation_df, inflation_label = _synthetic_two_regime_df(seed=1)
    labels = _label_set(growth_label, inflation_label)

    report = run_evaluation(
        growth_df,
        inflation_df,
        labels,
        _feature_config(),
        _hyperparameters(),
        dev_end=growth_df.index[60],
        include_holdout=False,
    )

    markdown = render_evaluation_markdown(report, metadata={})

    assert "### holdout" not in markdown
    assert "### dev" in markdown
    assert "### all" in markdown
