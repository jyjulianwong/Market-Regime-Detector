import pytest

from market_regime_detector.common.quadrants import best_regime, quadrant_probabilities


def test_quadrant_probabilities_sums_to_one() -> None:
    quadrants = quadrant_probabilities(0.7, 0.4)

    assert sum(quadrants.values()) == pytest.approx(1.0)


def test_quadrant_probabilities_matches_hand_computed_formula() -> None:
    quadrants = quadrant_probabilities(0.7, 0.4)

    assert quadrants["GROWTH_UP_INFLATION_DOWN"] == pytest.approx(0.7 * 0.6)
    assert quadrants["GROWTH_UP_INFLATION_UP"] == pytest.approx(0.7 * 0.4)
    assert quadrants["GROWTH_DOWN_INFLATION_UP"] == pytest.approx(0.3 * 0.4)
    assert quadrants["GROWTH_DOWN_INFLATION_DOWN"] == pytest.approx(0.3 * 0.6)


def test_best_regime_picks_the_maximum() -> None:
    quadrants = quadrant_probabilities(0.7, 0.4)

    regime, confidence = best_regime(quadrants)

    assert regime == "GROWTH_UP_INFLATION_DOWN"
    assert confidence == pytest.approx(0.42)


def test_best_regime_at_extremes() -> None:
    regime, confidence = best_regime(quadrant_probabilities(1.0, 0.0))

    assert regime == "GROWTH_UP_INFLATION_DOWN"
    assert confidence == pytest.approx(1.0)
