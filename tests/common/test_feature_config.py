import dataclasses

import pytest

from market_regime_detector.common.feature_config import (
    FEATURE_CONFIG_V1,
    FEATURE_CONFIGS,
    Axis,
    FeatureConfig,
    FeatureDefinition,
    Frequency,
    used_series_ids,
)


def _definition(**overrides: object) -> FeatureDefinition:
    base: dict[str, object] = {
        "name": "test_feature",
        "axis": Axis.GROWTH,
        "series_id": "TEST",
        "source": "fred",
        "native_frequency": Frequency.DAILY,
        "vintage_aware": False,
        "publication_lag_days": 0,
        "transform": "identity",
        "transform_params": {},
        "sign": 1,
    }
    base.update(overrides)
    return FeatureDefinition(**base)  # type: ignore[arg-type]


def test_sign_must_be_plus_or_minus_one() -> None:
    with pytest.raises(ValueError, match="sign must be 1 or -1"):
        _definition(sign=0)


def test_ratio_legs_must_be_set_together() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        _definition(numerator_series_id="HG=F")


def test_is_ratio_true_only_when_both_legs_set() -> None:
    assert not _definition().is_ratio
    assert _definition(numerator_series_id="HG=F", denominator_series_id="GC=F").is_ratio


def test_content_hash_is_stable_for_equal_configs() -> None:
    config_a = FeatureConfig(version="v1", features=(_definition(),))
    config_b = FeatureConfig(version="v1", features=(_definition(),))

    assert config_a.content_hash() == config_b.content_hash()


def test_content_hash_changes_when_a_field_changes() -> None:
    config_a = FeatureConfig(version="v1", features=(_definition(),))
    config_b = FeatureConfig(version="v1", features=(_definition(sign=-1),))

    assert config_a.content_hash() != config_b.content_hash()


def test_feature_configs_registry_resolves_v1_by_name() -> None:
    assert FEATURE_CONFIGS["v1"] is FEATURE_CONFIG_V1
    assert FEATURE_CONFIGS[FEATURE_CONFIG_V1.version] is FEATURE_CONFIG_V1


def test_first_pass_config_has_ten_features_from_nine_series() -> None:
    assert len(FEATURE_CONFIG_V1.features) == 10

    series_ids = {f.series_id for f in FEATURE_CONFIG_V1.features}
    assert len(series_ids) == 9  # T10YIE appears twice (1m, 3m change)


def test_first_pass_config_ratio_feature_has_both_legs() -> None:
    ratio_features = [f for f in FEATURE_CONFIG_V1.features if f.is_ratio]

    assert len(ratio_features) == 1
    assert ratio_features[0].numerator_series_id == "HG=F"
    assert ratio_features[0].denominator_series_id == "GC=F"


def test_feature_definition_is_frozen() -> None:
    feature = _definition()
    with pytest.raises(dataclasses.FrozenInstanceError):
        feature.sign = -1  # type: ignore[misc]


def test_from_dict_round_trips_to_dict_for_a_single_feature() -> None:
    config = FeatureConfig(version="v1", features=(_definition(),))

    round_tripped = FeatureConfig.from_dict(config.to_dict())

    assert round_tripped == config


def test_from_dict_round_trips_the_first_pass_config_exactly() -> None:
    round_tripped = FeatureConfig.from_dict(FEATURE_CONFIG_V1.to_dict())

    assert round_tripped == FEATURE_CONFIG_V1
    assert round_tripped.content_hash() == FEATURE_CONFIG_V1.content_hash()


def test_from_dict_preserves_ratio_feature_legs() -> None:
    config = FeatureConfig(
        version="v1",
        features=(_definition(numerator_series_id="HG=F", denominator_series_id="GC=F"),),
    )

    round_tripped = FeatureConfig.from_dict(config.to_dict())

    assert round_tripped.features[0].numerator_series_id == "HG=F"
    assert round_tripped.features[0].denominator_series_id == "GC=F"
    assert round_tripped.features[0].is_ratio


def test_used_series_ids_includes_both_ratio_legs_and_dedupes() -> None:
    config = FeatureConfig(
        version="v1",
        features=(
            _definition(series_id="A"),
            _definition(series_id="A"),  # duplicate on purpose
            _definition(
                numerator_series_id="HG=F", denominator_series_id="GC=F", series_id="HG=F/GC=F"
            ),
        ),
    )

    assert used_series_ids(config) == ["A", "GC=F", "HG=F"]


def test_used_series_ids_resolves_ratio_feature_to_its_two_real_legs() -> None:
    # 9 distinct series_id values (brief section 2.4) include the ratio feature's
    # composite label "HG=F/GC=F" as one entry; used_series_ids resolves that into
    # the two real series it actually needs to fetch, so the true count is 10.
    assert len(used_series_ids(FEATURE_CONFIG_V1)) == 10
    assert "HG=F" in used_series_ids(FEATURE_CONFIG_V1)
    assert "GC=F" in used_series_ids(FEATURE_CONFIG_V1)
    assert "HG=F/GC=F" not in used_series_ids(FEATURE_CONFIG_V1)
