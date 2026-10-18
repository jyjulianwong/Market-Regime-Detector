import pandas as pd

from market_regime_detector.common.feature_config import FeatureConfig
from market_regime_detector.common.quadrants import quadrant_probabilities
from market_regime_detector.training.trainer import FitResult

# Brief section 2.6: "validation against known episodes (2008-09, 2020, 2021-23
# inflation shock)."
KNOWN_EPISODES: dict[str, tuple[str, str]] = {
    "2008-09 financial crisis": ("2008-01-01", "2009-12-31"),
    "2020 covid shock": ("2020-01-01", "2020-12-31"),
    "2021-23 inflation shock": ("2021-01-01", "2023-12-31"),
}


def compare_to_baseline(
    walk_forward_df: pd.DataFrame, baseline_series: pd.Series
) -> dict[str, object]:
    """Brief section 2.6: "the HMMs must be compared to [the baseline]... If the HMMs
    do not beat the baseline out of sample, say so."

    There's no ground-truth regime label to score either model against, so "beat"
    isn't directly computable — what's reported honestly instead is how often the
    HMM's walk-forward out-of-sample label agrees with the baseline's label for the
    same date. High agreement means the HMM mostly reproduces the (much cheaper)
    baseline's reading; it is not, by itself, evidence the HMM is "more right."
    """
    aligned = walk_forward_df.set_index("decision_date")["label"]
    common_dates = aligned.index.intersection(baseline_series.index)
    if len(common_dates) == 0:
        return {"n_out_of_sample_months": 0, "label_agreement_with_baseline": None}

    agreement = (aligned.loc[common_dates] == baseline_series.loc[common_dates]).mean()
    return {
        "n_out_of_sample_months": int(len(common_dates)),
        "label_agreement_with_baseline": float(agreement),
    }


def combine_quadrant_probabilities(
    growth_walk_forward_df: pd.DataFrame, inflation_walk_forward_df: pd.DataFrame
) -> pd.DataFrame:
    """Brief section 2.2's independence-assumption product, applied to the two axes'
    out-of-sample `p_up` series on the dates common to both (their walk-forward
    out-of-sample windows don't necessarily start on the same date — each axis drops
    its own warm-up rows independently, brief section 3.2/Phase 3).
    """
    growth = growth_walk_forward_df.set_index("decision_date")["p_up"]
    inflation = inflation_walk_forward_df.set_index("decision_date")["p_up"]
    common_dates = growth.index.intersection(inflation.index)

    p_growth_up = growth.loc[common_dates]
    p_inflation_up = inflation.loc[common_dates]

    # Same formula as common/quadrants.quadrant_probabilities, applied vectorized
    # across the whole timeline rather than one date at a time — prediction/predict.py
    # uses that scalar version directly for a single day.
    probs = quadrant_probabilities(p_growth_up, p_inflation_up)  # type: ignore[arg-type]
    quadrants = pd.DataFrame(probs, index=common_dates)
    quadrants["regime"] = quadrants.idxmax(axis=1)
    quadrants["confidence"] = quadrants.drop(columns="regime").max(axis=1)
    return quadrants


def episode_report(
    timeline_df: pd.DataFrame,
    value_column: str,
    episodes: dict[str, tuple[str, str]] = KNOWN_EPISODES,
) -> pd.DataFrame:
    """A data artifact (brief section 2.6), not a pass/fail assertion: `value_column`
    (e.g. `p_up`, or a quadrant's probability) averaged over each named historical
    window, for a human to read in the model card.
    """
    rows: list[dict[str, object]] = []
    for name, (start, end) in episodes.items():
        window = timeline_df.loc[pd.Timestamp(start) : pd.Timestamp(end)]
        rows.append(
            {
                "episode": name,
                "start": start,
                "end": end,
                "n_months": len(window),
                f"mean_{value_column}": float(window[value_column].mean())
                if not window.empty
                else None,
            }
        )
    return pd.DataFrame(rows)


def render_model_card(
    feature_config: FeatureConfig,
    growth_fit: FitResult,
    inflation_fit: FitResult,
    growth_baseline_comparison: dict[str, object],
    inflation_baseline_comparison: dict[str, object],
    quadrant_episode_report: pd.DataFrame,
    hyperparameters: dict[str, object],
    growth_skipped_folds: list[dict[str, object]] | None = None,
    inflation_skipped_folds: list[dict[str, object]] | None = None,
) -> str:
    """Brief section 4.1's `model_card.md`."""

    def _state_table(fit: FitResult) -> str:
        lines = ["| State | Label | Composite | Expected duration (months) |", "|---|---|---|---|"]
        durations = 1.0 / (1.0 - fit.transition_matrix.diagonal())
        for state in sorted(fit.state_labels):
            lines.append(
                f"| {state} | {fit.state_labels[state]} | {fit.composites[state]:.3f} "
                f"| {durations[state]:.1f} |"
            )
        return "\n".join(lines)

    def _skipped_folds_section() -> str:
        growth_skipped = growth_skipped_folds or []
        inflation_skipped = inflation_skipped_folds or []
        if not growth_skipped and not inflation_skipped:
            return ""
        lines = [
            "\n## Walk-forward coverage gaps\n",
            "The following expanding-window folds found no separable two-state "
            "structure (brief section 2.3's sanity checks) and were excluded from "
            "the out-of-sample metrics above, rather than relaxing the checks or "
            "aborting the whole evaluation:\n",
        ]
        for axis_name, skipped in (("Growth", growth_skipped), ("Inflation", inflation_skipped)):
            for fold in skipped:
                lines.append(
                    f"- **{axis_name}**, fold {fold['fold_id']} (test "
                    f"{fold['test_start']} to {fold['test_end']}, trained on data "
                    f"through {fold['train_end']}): {fold['reason']}"
                )
        return "\n".join(lines) + "\n"

    return f"""# Model Card

## Architecture

Two independent 2-state Gaussian HMMs (growth, inflation), `covariance_type=
"{growth_fit.covariance_type}"`. Quadrant probabilities are the product of the two
axes' marginals, assuming independence between growth and inflation — only
approximately true; see "Limitations" below.

Inference uses a causal forward (filtered) posterior only, in training's
walk-forward evaluation and at prediction time alike — never `hmmlearn`'s smoothed
`predict_proba` or a Viterbi path.

## Feature config

Version `{feature_config.version}`, hash `{feature_config.content_hash()}`,
{len(feature_config.features)} features.

## Growth HMM

{_state_table(growth_fit)}

Log-likelihood (full-history fit): {growth_fit.log_likelihood:.2f}

## Inflation HMM

{_state_table(inflation_fit)}

Log-likelihood (full-history fit): {inflation_fit.log_likelihood:.2f}

## Baseline comparison (walk-forward, out-of-sample)

Baseline: sign of a sign-weighted average of each axis's already-computed
rate-of-change features (brief section 2.6 — this reading of "change in a
composite" was chosen over a second, unspecified level-based composite).

- Growth: {growth_baseline_comparison["n_out_of_sample_months"]} out-of-sample
  months, {growth_baseline_comparison["label_agreement_with_baseline"]} label
  agreement with the baseline.
- Inflation: {inflation_baseline_comparison["n_out_of_sample_months"]} out-of-sample
  months, {inflation_baseline_comparison["label_agreement_with_baseline"]} label
  agreement with the baseline.

High agreement means the HMM mostly reproduces the baseline's reading on the same
data — it is not, by itself, evidence the HMM is more accurate; there is no ground
truth regime label to score either model against.

## Known-episode validation (walk-forward quadrant, out-of-sample)

{quadrant_episode_report.to_markdown(index=False)}
{_skipped_folds_section()}
## Hyperparameters

```
{hyperparameters}
```

## Limitations

- Independence assumption between growth and inflation axes is only approximate.
- The model trains on month-end decision dates; the daily batch job (Phase 6)
  evaluates features as of the run date (trailing windows ending that day, using
  only data published by then), not a strict month-end — see brief section 3.3.
- `covariance_type="diag"` and the baseline composite definition are judgment calls,
  not brief-specified choices.
"""
