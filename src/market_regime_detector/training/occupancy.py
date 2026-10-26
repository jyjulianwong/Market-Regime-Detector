from typing import Any

import pandas as pd


def build_occupancy_report(result: pd.DataFrame) -> dict[str, Any]:
    """Builds a state-occupancy / state-means report straight from one axis's
    `walk_forward_evaluate` output (`decision_date`/`p_up`/`label`/`fold_id`, plus
    the `fold_fits` attr `walk_forward_evaluate` stashes alongside
    `skipped_folds`) — Phase 0 of `HMM_ACCURACY_PLAN.md` wants "is DOWN just
    COVID?" (growth) / "is DOWN/UP just 2021-23?" (inflation) as a visible output
    rather than left implicit in `flip_rate_per_year`/`mean_regime_duration_months`.

    `overall` groups every evaluated month by its assigned label across all folds
    — fold boundaries don't change what "UP"/"DOWN" means, since the label comes
    from each fold's own composite score (brief section 2.3), not an arbitrary
    state index, so aggregating across refits is meaningful. `by_fold` keeps each
    fold's own per-feature emission means in z-units, so a state that only shows
    up because one feature's value is extreme in that fold is visible fold by
    fold, not just in the aggregate.
    """
    if result.empty:
        return {"overall": {}, "by_fold": []}

    overall: dict[str, Any] = {}
    for label_name, group in result.groupby("label"):
        dates = pd.DatetimeIndex(group["decision_date"])
        overall[str(label_name)] = {
            "n_months": int(len(group)),
            "fraction": float(len(group) / len(result)),
            "first_date": str(dates.min().date()),
            "last_date": str(dates.max().date()),
        }

    by_fold: list[dict[str, Any]] = []
    for fold_summary in result.attrs.get("fold_fits", []):
        fold_rows = result[result["fold_id"] == fold_summary["fold_id"]]
        fold_occupancy: dict[str, Any] = {}
        for label_name, group in fold_rows.groupby("label"):
            dates = pd.DatetimeIndex(group["decision_date"])
            fold_occupancy[str(label_name)] = {
                "n_months": int(len(group)),
                "fraction": float(len(group) / len(fold_rows)),
                "first_date": str(dates.min().date()),
                "last_date": str(dates.max().date()),
            }
        by_fold.append(
            {
                "fold_id": fold_summary["fold_id"],
                "test_start": fold_summary["test_start"],
                "test_end": fold_summary["test_end"],
                "occupancy": fold_occupancy,
                "state_means": fold_summary["state_means"],
            }
        )

    return {"overall": overall, "by_fold": by_fold}
