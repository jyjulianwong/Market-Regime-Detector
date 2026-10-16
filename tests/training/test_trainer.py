import numpy as np
import pandas as pd
import pytest

from market_regime_detector.common.feature_config import Axis, FeatureDefinition, Frequency
from market_regime_detector.training.trainer import fit_hmm


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
    """Alternating blocks of two well-separated, persistent Gaussian regimes — a
    small HMM should reliably recover this structure.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for block in range(n_blocks):
        mean = 2.0 if block % 2 == 0 else -2.0
        blocks.append(rng.normal(loc=mean, scale=0.3, size=(n_per_block, 2)))
    data = np.concatenate(blocks, axis=0)
    dates = pd.date_range("2000-01-31", periods=len(data), freq="ME")
    return pd.DataFrame(data, columns=["f1", "f2"], index=dates)


FEATURE_DEFS = [_feature("f1", sign=1), _feature("f2", sign=1)]


def test_fit_hmm_labels_up_state_as_higher_composite() -> None:
    df = _synthetic_two_regime_df()

    result = fit_hmm(df, FEATURE_DEFS, random_state=0)

    up_state = next(s for s, label in result.state_labels.items() if label == "UP")
    down_state = next(s for s, label in result.state_labels.items() if label == "DOWN")
    assert result.composites[up_state] > result.composites[down_state]


def test_fit_hmm_state_labelling_is_deterministic_across_refits() -> None:
    """Brief section 3.8, explicitly required: same data + same random_state produces
    the same state->label mapping every time.
    """
    df = _synthetic_two_regime_df()

    first = fit_hmm(df, FEATURE_DEFS, random_state=0)
    second = fit_hmm(df, FEATURE_DEFS, random_state=0)

    assert first.state_labels == second.state_labels
    assert first.composites == pytest.approx(second.composites)
    assert first.log_likelihood == pytest.approx(second.log_likelihood)


def test_fit_hmm_raises_when_no_restart_passes_sanity_checks() -> None:
    df = _synthetic_two_regime_df()

    with pytest.raises(ValueError, match="no restart"):
        fit_hmm(df, FEATURE_DEFS, random_state=0, composite_separation_margin=1_000.0)


def test_fit_hmm_result_includes_feature_names_in_order() -> None:
    df = _synthetic_two_regime_df()

    result = fit_hmm(df, FEATURE_DEFS, random_state=0)

    assert result.feature_names == ["f1", "f2"]
    assert result.means.shape == (2, 2)
    assert result.variances.shape == (2, 2)
    assert result.transition_matrix.shape == (2, 2)
    assert result.start_prob.shape == (2,)
