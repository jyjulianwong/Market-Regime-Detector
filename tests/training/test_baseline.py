import pandas as pd

from market_regime_detector.common.feature_config import Axis, FeatureDefinition, Frequency
from market_regime_detector.training.baseline import baseline_signal


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


def test_baseline_signal_sign_weighted_average() -> None:
    feature_defs = [_feature("a", sign=1), _feature("b", sign=-1)]
    axis_df = pd.DataFrame(
        {"a": [1.0, -1.0], "b": [1.0, 1.0]},
        index=pd.to_datetime(["2024-01-31", "2024-02-29"]),
    )

    result = baseline_signal(axis_df, feature_defs)

    # row0: mean(1*1, 1*-1) = 0 -> tie rounds to UP
    # row1: mean(-1*1, 1*-1) = -1 -> DOWN
    assert result.iloc[0] == "UP"
    assert result.iloc[1] == "DOWN"


def test_baseline_signal_ignores_features_not_in_axis_df() -> None:
    feature_defs = [_feature("a", sign=1), _feature("unused", sign=1)]
    axis_df = pd.DataFrame({"a": [2.0]}, index=pd.to_datetime(["2024-01-31"]))

    result = baseline_signal(axis_df, feature_defs)

    assert result.iloc[0] == "UP"
