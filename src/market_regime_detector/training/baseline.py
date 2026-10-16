import pandas as pd

from market_regime_detector.common.feature_config import FeatureDefinition


def baseline_signal(axis_df: pd.DataFrame, feature_defs: list[FeatureDefinition]) -> pd.Series:
    """Brief section 2.6's baseline: "a simple rule-based classifier (sign of the
    3-month change in a growth composite and an inflation composite)."

    Every `FEATURE_CONFIG_V1` feature is already a rate-of-change (section 2.4), so
    the composite here is a sign-weighted average of those already-computed feature
    values — the direct reading of "change in a composite" available from this
    project's inputs, rather than inventing a second, unspecified level-based
    composite to re-differentiate. Documented as a judgment call in CLAUDE.md.

    Returns `"UP"`/`"DOWN"` per decision date; a composite of exactly 0 rounds to
    `"UP"` (a rare tie, not worth a third label).
    """
    relevant = [f for f in feature_defs if f.name in axis_df.columns]
    signs = pd.Series({f.name: f.sign for f in relevant})
    composite = (axis_df[list(signs.index)] * signs).mean(axis=1)
    return composite.apply(lambda x: "UP" if x >= 0 else "DOWN")
