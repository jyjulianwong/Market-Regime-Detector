import pandas as pd
import pytest

from market_regime_detector.training.report import (
    combine_quadrant_probabilities,
    compare_to_baseline,
    episode_report,
)


def test_compare_to_baseline_computes_agreement() -> None:
    walk_forward_df = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "p_up": [0.9, 0.1, 0.8],
            "label": ["UP", "DOWN", "UP"],
            "fold_id": [0, 0, 0],
        }
    )
    baseline_series = pd.Series(
        ["UP", "UP", "UP"],
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
    )

    result = compare_to_baseline(walk_forward_df, baseline_series)

    assert result["n_out_of_sample_months"] == 3
    assert result["label_agreement_with_baseline"] == pytest.approx(2 / 3)


def test_compare_to_baseline_handles_no_overlap() -> None:
    walk_forward_df = pd.DataFrame(columns=["decision_date", "p_up", "label", "fold_id"])
    baseline_series = pd.Series([], dtype=object)

    result = compare_to_baseline(walk_forward_df, baseline_series)

    assert result["n_out_of_sample_months"] == 0
    assert result["label_agreement_with_baseline"] is None


def _walk_forward_df(dates: list[str], p_up: list[float]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "decision_date": pd.to_datetime(dates),
            "p_up": p_up,
            "label": ["UP" if p >= 0.5 else "DOWN" for p in p_up],
            "fold_id": [0] * n,
        }
    )


def test_combine_quadrant_probabilities_sums_to_one_and_picks_highest_regime() -> None:
    growth = _walk_forward_df(["2020-01-31", "2020-02-29"], [0.7, 0.3])
    inflation = _walk_forward_df(["2020-01-31", "2020-02-29"], [0.4, 0.6])

    result = combine_quadrant_probabilities(growth, inflation)

    quadrant_cols = [
        "GROWTH_UP_INFLATION_DOWN",
        "GROWTH_UP_INFLATION_UP",
        "GROWTH_DOWN_INFLATION_UP",
        "GROWTH_DOWN_INFLATION_DOWN",
    ]
    sums = result[quadrant_cols].sum(axis=1)
    assert (sums - 1.0).abs().max() < 1e-9
    # 0.7 * (1 - 0.4) = 0.42 is the largest of the four quadrant probabilities.
    assert result.iloc[0]["regime"] == "GROWTH_UP_INFLATION_DOWN"
    assert result["confidence"].between(0, 1).all()


def test_combine_quadrant_probabilities_only_uses_common_dates() -> None:
    growth = _walk_forward_df(["2020-01-31", "2020-02-29"], [0.7, 0.3])
    inflation = _walk_forward_df(["2020-02-29"], [0.5])

    result = combine_quadrant_probabilities(growth, inflation)

    assert len(result) == 1


def test_episode_report_shape() -> None:
    timeline = pd.DataFrame(
        {"confidence": [0.6, 0.7]}, index=pd.to_datetime(["2008-06-30", "2020-06-30"])
    )
    episodes = {"gfc": ("2008-01-01", "2009-12-31"), "covid": ("2020-01-01", "2020-12-31")}

    result = episode_report(timeline, value_column="confidence", episodes=episodes)

    assert list(result["episode"]) == ["gfc", "covid"]
    assert result.loc[0, "n_months"] == 1
    assert result.loc[0, "mean_confidence"] == pytest.approx(0.6)


def test_episode_report_handles_empty_window() -> None:
    timeline = pd.DataFrame({"confidence": [0.6]}, index=pd.to_datetime(["2020-06-30"]))
    episodes = {"gfc": ("2008-01-01", "2009-12-31")}

    result = episode_report(timeline, value_column="confidence", episodes=episodes)

    assert result.loc[0, "n_months"] == 0
    assert result.loc[0, "mean_confidence"] is None
