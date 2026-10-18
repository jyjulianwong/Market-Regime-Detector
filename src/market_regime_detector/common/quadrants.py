QUADRANT_NAMES = (
    "GROWTH_UP_INFLATION_DOWN",
    "GROWTH_UP_INFLATION_UP",
    "GROWTH_DOWN_INFLATION_UP",
    "GROWTH_DOWN_INFLATION_DOWN",
)


def quadrant_probabilities(p_growth_up: float, p_inflation_up: float) -> dict[str, float]:
    """Brief section 2.2's independence-assumption product — `P(quadrant) =
    P(axis_up) × P(other_axis_up_or_down)`. Shared by `training/report.py` (across a
    whole walk-forward timeline) and `prediction/predict.py` (a single day), so it
    lives in `common/` rather than being duplicated or imported across the
    `training/` → `hmmlearn` boundary.
    """
    return {
        "GROWTH_UP_INFLATION_DOWN": p_growth_up * (1 - p_inflation_up),
        "GROWTH_UP_INFLATION_UP": p_growth_up * p_inflation_up,
        "GROWTH_DOWN_INFLATION_UP": (1 - p_growth_up) * p_inflation_up,
        "GROWTH_DOWN_INFLATION_DOWN": (1 - p_growth_up) * (1 - p_inflation_up),
    }


def best_regime(quadrants: dict[str, float]) -> tuple[str, float]:
    """The highest-probability quadrant and its probability — brief section 5's
    `regime`/`confidence` fields.
    """
    regime = max(quadrants, key=lambda name: quadrants[name])
    return regime, quadrants[regime]
