import numpy as np
import pandas as pd

from market_regime_detector.common.feature_config import Axis, FeatureDefinition, Frequency
from market_regime_detector.training.evaluation import walk_forward_evaluate
from market_regime_detector.training.occupancy import build_occupancy_report


def _feature(name: str, sign: int) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        axis=Axis.GROWTH,
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
    n_per_block: int = 12, n_blocks: int = 10, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    blocks = []
    for block in range(n_blocks):
        mean = 2.0 if block % 2 == 0 else -2.0
        blocks.append(rng.normal(loc=mean, scale=0.3, size=(n_per_block, 2)))
    data = np.concatenate(blocks, axis=0)
    dates = pd.date_range("2000-01-31", periods=len(data), freq="ME")
    return pd.DataFrame(data, columns=["f1", "f2"], index=dates)


FEATURE_DEFS = [_feature("f1", sign=1), _feature("f2", sign=1)]


def test_build_occupancy_report_empty_result_returns_empty_report() -> None:
    report = build_occupancy_report(pd.DataFrame())
    assert report == {"overall": {}, "by_fold": []}


def test_build_occupancy_report_overall_matches_label_counts() -> None:
    df = _synthetic_two_regime_df()
    result = walk_forward_evaluate(
        df, FEATURE_DEFS, initial_window_months=60, refit_every_months=12, random_state=0
    )

    report = build_occupancy_report(result)

    overall = report["overall"]
    assert set(overall) <= {"UP", "DOWN"}
    assert sum(stats["n_months"] for stats in overall.values()) == len(result)
    for state, stats in overall.items():
        rows = result[result["label"] == state]
        assert stats["n_months"] == len(rows)
        assert stats["fraction"] == len(rows) / len(result)
        assert stats["first_date"] == str(rows["decision_date"].min().date())
        assert stats["last_date"] == str(rows["decision_date"].max().date())


def test_build_occupancy_report_by_fold_has_both_states_and_feature_means() -> None:
    df = _synthetic_two_regime_df()
    result = walk_forward_evaluate(
        df, FEATURE_DEFS, initial_window_months=60, refit_every_months=12, random_state=0
    )

    report = build_occupancy_report(result)

    assert len(report["by_fold"]) == result["fold_id"].nunique()
    for fold in report["by_fold"]:
        # Every fit always labels both states (trainer.fit_hmm), but a given
        # fold's out-of-sample test window can land entirely in one state (e.g.
        # this synthetic data's 12-month blocks line up with the 12-month refit
        # cadence) — so `occupancy` can legitimately be a strict subset of
        # `state_means`'s keys.
        assert set(fold["state_means"]) == {"UP", "DOWN"}
        for means in fold["state_means"].values():
            assert set(means) == {"f1", "f2"}
        assert set(fold["occupancy"]) <= {"UP", "DOWN"}
        fractions = [stats["fraction"] for stats in fold["occupancy"].values()]
        assert abs(sum(fractions) - 1.0) < 1e-9


def test_build_occupancy_report_down_state_is_not_only_one_episode() -> None:
    """A loose sanity check standing in for the plan's "is DOWN just COVID?"
    question: on this alternating synthetic pattern, DOWN should span more than
    one block, i.e. the overall first/last DOWN dates are far apart.
    """
    df = _synthetic_two_regime_df(n_per_block=12, n_blocks=10)
    result = walk_forward_evaluate(
        df, FEATURE_DEFS, initial_window_months=60, refit_every_months=12, random_state=0
    )

    report = build_occupancy_report(result)
    down = report["overall"]["DOWN"]
    first = pd.Timestamp(down["first_date"])
    last = pd.Timestamp(down["last_date"])
    assert (last - first).days > 180
