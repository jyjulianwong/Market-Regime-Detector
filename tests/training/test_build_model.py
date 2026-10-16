from pathlib import Path

import numpy as np
import pandas as pd

from market_regime_detector.common.feature_config import (
    Axis,
    FeatureConfig,
    FeatureDefinition,
    Frequency,
)
from market_regime_detector.training.build_model import HMMHyperparameters, train_and_save_model


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


def _synthetic_two_regime_df(seed: int, n_per_block: int = 10, n_blocks: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    blocks = []
    for block in range(n_blocks):
        mean = 2.0 if block % 2 == 0 else -2.0
        blocks.append(rng.normal(loc=mean, scale=0.3, size=(n_per_block, 2)))
    data = np.concatenate(blocks, axis=0)
    dates = pd.date_range("2000-01-31", periods=len(data), freq="ME")
    return pd.DataFrame(data, columns=["f1", "f2"], index=dates)


def test_train_and_save_model_writes_all_six_files(tmp_path: Path) -> None:
    growth_df = _synthetic_two_regime_df(seed=0)
    inflation_df = _synthetic_two_regime_df(seed=1)
    feature_config = FeatureConfig(
        version="test",
        features=(
            _feature("f1", Axis.GROWTH, sign=1),
            _feature("f2", Axis.GROWTH, sign=1),
            _feature("f1", Axis.INFLATION, sign=1),
            _feature("f2", Axis.INFLATION, sign=1),
        ),
    )
    output_dir = tmp_path / "model"

    train_and_save_model(
        output_dir,
        growth_df,
        inflation_df,
        feature_config=feature_config,
        hyperparameters=HMMHyperparameters(
            n_restarts=3, initial_window_months=60, refit_every_months=12, random_state=0
        ),
    )

    expected_files = {
        "growth_hmm.npz",
        "inflation_hmm.npz",
        "scalers.npz",
        "labels.json",
        "config.json",
        "metrics.json",
        "model_card.md",
    }
    assert {p.name for p in output_dir.iterdir()} == expected_files

    growth_npz = np.load(output_dir / "growth_hmm.npz")
    assert growth_npz["means"].shape == (2, 2)
    assert growth_npz["variances"].shape == (2, 2)

    model_card = (output_dir / "model_card.md").read_text()
    assert "Model Card" in model_card
    assert "Limitations" in model_card
