import pandas as pd

from market_regime_detector.common.feature_config import FeatureDefinition

# Same warm-up `benchmarks.fixed_rule_baseline` uses for the evaluation harness's
# own causal "rule" baseline — this module exists for a separate consumer
# (`training/build_model.py`'s model-card comparison), not a different convention.
MIN_OBSERVATIONS = 24


def baseline_signal(
    axis_df: pd.DataFrame,
    feature_defs: list[FeatureDefinition],
    min_observations: int = MIN_OBSERVATIONS,
) -> pd.Series:
    """Brief section 2.6's baseline: "a simple rule-based classifier (sign of the
    3-month change in a growth composite and an inflation composite)."

    Every `FEATURE_CONFIG_V1` feature is already a rate-of-change (section 2.4), so
    the composite here is a sign-weighted average of those already-computed feature
    values — the direct reading of "change in a composite" available from this
    project's inputs, rather than inventing a second, unspecified level-based
    composite to re-differentiate — a deliberate judgment call.

    Standardizes every feature with its own **expanding** (causal) mean/std
    before combining, rather than averaging raw feature values directly
    (`HMM_ACCURACY_PLAN.md` Phase 2) — otherwise whichever feature happens to
    have the largest native units (e.g. ICSA's raw claims count in the
    thousands) swamps every rate-of-change feature in the composite, making
    "agreement with baseline" (`report.compare_to_baseline`) close to
    meaningless. Each row's z-score uses only that row's own expanding window up
    to and including it — never a later row — so this stays a fair,
    look-ahead-free comparison. A row is dropped, not computed, until every
    feature has at least `min_observations` expanding-window observations.

    Returns `"UP"`/`"DOWN"` per decision date; a composite of exactly 0 rounds to
    `"UP"` (a rare tie, not worth a third label).
    """
    relevant = [f for f in feature_defs if f.name in axis_df.columns]
    signs = pd.Series({f.name: f.sign for f in relevant})
    sub = axis_df[list(signs.index)]

    expanding_mean = sub.expanding(min_periods=min_observations).mean()
    expanding_std = sub.expanding(min_periods=min_observations).std()
    z = (sub - expanding_mean) / expanding_std

    composite = (z * signs).mean(axis=1).dropna()
    return composite.apply(lambda x: "UP" if x >= 0 else "DOWN")
