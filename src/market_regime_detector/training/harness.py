from typing import Any

import numpy as np
import pandas as pd

from market_regime_detector.common.feature_config import Axis, FeatureConfig
from market_regime_detector.training.benchmarks import (
    calibrated_persistence_baseline,
    climatology_baseline,
    fixed_rule_baseline,
    logistic_baseline,
)
from market_regime_detector.training.build_model import HMMHyperparameters
from market_regime_detector.training.evaluation import walk_forward_evaluate
from market_regime_detector.training.labels import LabelSet
from market_regime_detector.training.occupancy import build_occupancy_report
from market_regime_detector.training.scoring import (
    CLIP_EPS,
    accuracy,
    auc,
    brier,
    calibration_table,
    detection_lag,
    flip_rate_per_year,
    log_loss,
    mean_regime_duration_months,
    quadrant_log_loss,
)

AXES = ("growth", "inflation")
MODEL_NAMES = ("climatology", "persistence", "rule", "logistic", "hmm_current")
DEFAULT_DEV_END = pd.Timestamp("2016-12-31")


def _walk_forward_series(result: pd.DataFrame) -> pd.Series:
    """`walk_forward_evaluate`/`logistic_baseline` both return a `decision_date`/
    `p_up`/`label`/`fold_id` DataFrame — this is the one place both get turned
    into the plain `p_up` series every other function in this module works with.
    """
    if result.empty:
        return pd.Series(dtype=float)
    series: pd.Series = pd.Series(
        result["p_up"].to_numpy(), index=pd.DatetimeIndex(result["decision_date"])
    )
    return series


def _axis_model_series(
    axis_df: pd.DataFrame,
    feature_config: FeatureConfig,
    axis: Axis,
    label: pd.Series,
    label_lag_months: int,
    hyperparameters: HMMHyperparameters,
) -> tuple[dict[str, pd.Series], dict[str, list[dict[str, Any]]], pd.DataFrame]:
    """Every model's `p_up` series for one axis, each model's skipped-fold
    records (empty for the three baselines that never "fit" in a way that can
    fail — only the HMM and the logistic baseline have a fold-fitting step that
    can be skipped), and the HMM's own raw walk-forward result (kept whole,
    with its `fold_fits`/`fold_id` attrs/column intact, for `build_occupancy_report`
    — `_walk_forward_series` above collapses those away).
    """
    feature_defs = [f for f in feature_config.features if f.axis == axis]
    hp = hyperparameters

    hmm_result = walk_forward_evaluate(
        axis_df,
        feature_defs,
        initial_window_months=hp.initial_window_months,
        refit_every_months=hp.refit_every_months,
        n_restarts=hp.n_restarts,
        covariance_type=hp.covariance_type,
        min_state_duration_months=hp.min_state_duration_months,
        composite_separation_margin=hp.composite_separation_margin,
        random_state=hp.random_state,
        scaler_clip=hp.scaler_clip,
        scaler_method=hp.scaler_method,
    )
    logistic_result = logistic_baseline(
        axis_df,
        feature_defs,
        label,
        label_lag_months,
        initial_window_months=hp.initial_window_months,
        refit_every_months=hp.refit_every_months,
        scaler_clip=hp.scaler_clip,
        scaler_method=hp.scaler_method,
    )

    month_end_index = pd.DatetimeIndex(axis_df.index)
    series = {
        "climatology": climatology_baseline(
            label, month_end_index, label_lag_months, hp.initial_window_months
        ),
        "persistence": calibrated_persistence_baseline(
            label, month_end_index, label_lag_months, hp.initial_window_months
        ),
        "rule": fixed_rule_baseline(axis_df, feature_defs, hp.initial_window_months),
        "logistic": _walk_forward_series(logistic_result),
        "hmm_current": _walk_forward_series(hmm_result),
    }
    skipped_folds = {
        "climatology": [],
        "persistence": [],
        "rule": [],
        "logistic": logistic_result.attrs.get("skipped_folds", []),
        "hmm_current": hmm_result.attrs.get("skipped_folds", []),
    }
    return series, skipped_folds, hmm_result


def _per_row_log_loss(p: pd.Series, y: pd.Series) -> pd.Series:
    """Per-month log loss contributions, aligned and clipped the same way
    `scoring.log_loss` is — needed (rather than just the scalar mean) for the
    block-bootstrapped skill-score CI below.
    """
    aligned = pd.concat([p.rename("p"), y.rename("y")], axis=1, join="inner").dropna()
    p_clipped = aligned["p"].clip(CLIP_EPS, 1 - CLIP_EPS)
    losses: pd.Series = -(
        aligned["y"] * np.log(p_clipped) + (1 - aligned["y"]) * np.log(1 - p_clipped)
    )
    return losses


def _skill_ci(
    model_p: pd.Series, climatology_p: pd.Series, label: pd.Series, block: int, seed: int
) -> tuple[float, float, float]:
    """Block-bootstrap CI for the skill score `1 - mean(model_loss)/
    mean(climatology_loss)`, resampling the same month-blocks for both series'
    per-month losses together so each bootstrap replicate recomputes the full
    ratio — a plain CI on the mean *difference* of losses would answer a related
    but different question (an additive, not a relative, improvement).
    """
    model_losses = _per_row_log_loss(model_p, label)
    climatology_losses = _per_row_log_loss(climatology_p, label)
    common = model_losses.index.intersection(climatology_losses.index)
    model_arr = model_losses.loc[common].to_numpy()
    climatology_arr = climatology_losses.loc[common].to_numpy()
    n = len(common)

    point = 1.0 - model_arr.mean() / climatology_arr.mean() if n > 0 else float("nan")
    if n < block:
        return point, point, point

    rng = np.random.default_rng(seed)
    n_boot = 2000
    n_blocks_needed = -(-n // block)
    boot_skills = np.empty(n_boot)
    for i in range(n_boot):
        block_starts = rng.integers(0, n - block + 1, size=n_blocks_needed)
        model_resample = np.concatenate([model_arr[s : s + block] for s in block_starts])[:n]
        climatology_resample = np.concatenate(
            [climatology_arr[s : s + block] for s in block_starts]
        )[:n]
        boot_skills[i] = 1.0 - model_resample.mean() / climatology_resample.mean()

    low, high = np.percentile(boot_skills, [2.5, 97.5])
    return point, float(low), float(high)


def _model_metrics(
    model_p: pd.Series, climatology_p: pd.Series, label: pd.Series
) -> dict[str, Any]:
    table, ece = calibration_table(model_p, label)
    table_records = table.assign(bin=table["bin"].astype(str)).to_dict(orient="records")
    skill_point, skill_low, skill_high = _skill_ci(model_p, climatology_p, label, block=6, seed=0)

    aligned = pd.concat([model_p.rename("p"), label.rename("y")], axis=1, join="inner").dropna()
    return {
        "n_months": len(aligned),
        "log_loss": log_loss(model_p, label),
        "brier": brier(model_p, label),
        "skill_vs_climatology": skill_point,
        "skill_vs_climatology_ci_low": skill_low,
        "skill_vs_climatology_ci_high": skill_high,
        "auc": auc(model_p, label),
        "accuracy": accuracy(model_p, label),
        "expected_calibration_error": ece,
        "calibration_table": table_records,
        "flip_rate_per_year": flip_rate_per_year(model_p),
        "mean_regime_duration_months": mean_regime_duration_months(model_p),
    }


def _split_index(common: pd.DatetimeIndex, dev_end: pd.Timestamp, split: str) -> pd.DatetimeIndex:
    if split == "all":
        return common
    if split == "dev":
        return common[common <= dev_end]
    if split == "holdout":
        return common[common > dev_end]
    raise ValueError(f"unknown split {split!r}; expected 'dev', 'holdout', or 'all'")


def run_evaluation(
    growth_df: pd.DataFrame,
    inflation_df: pd.DataFrame,
    labels: LabelSet,
    feature_config: FeatureConfig,
    hyperparameters: HMMHyperparameters,
    dev_end: pd.Timestamp = DEFAULT_DEV_END,
    include_holdout: bool = True,
) -> dict[str, Any]:
    """Scores the current (unmodified) model configuration against four causal
    baselines, over the exact same walk-forward test months the HMM itself
    produces. Returns a single JSON-serializable report dict — see
    `scripts/evaluate_model.py` for how it becomes `evaluation.json`/`.md`.

    Every number is computed on the **intersection** of months where every model
    being compared in that table has a prediction and the relevant label exists
    (the "same-month comparability rule") — each model's own raw coverage before
    that intersection is reported alongside it, never silently discarded without
    a trace.

    `axis_report["occupancy"]` (one per axis) is the HMM's own state-occupancy /
    state-means report (`training/occupancy.build_occupancy_report`) — independent
    of any split, since it answers "what did the HMM actually learn" rather than
    "how good were its predictions."
    """
    axis_dfs = {Axis.GROWTH: growth_df, Axis.INFLATION: inflation_df}
    axis_labels = {Axis.GROWTH: labels.growth, Axis.INFLATION: labels.inflation}
    usrec_monthly = 1.0 - labels.growth_recession

    splits = ("dev", "holdout", "all") if include_holdout else ("dev", "all")

    report: dict[str, Any] = {
        "label_lag_months": labels.label_lag_months,
        "dev_end": dev_end.date().isoformat(),
        "include_holdout": include_holdout,
        "axes": {},
    }

    series_by_axis: dict[Axis, dict[str, pd.Series]] = {}
    for axis in (Axis.GROWTH, Axis.INFLATION):
        axis_df = axis_dfs[axis]
        label = axis_labels[axis]
        series, skipped_folds, hmm_result = _axis_model_series(
            axis_df, feature_config, axis, label, labels.label_lag_months, hyperparameters
        )
        series_by_axis[axis] = series

        common = pd.DatetimeIndex(label.dropna().index)
        for model_series in series.values():
            common = common.intersection(pd.DatetimeIndex(model_series.dropna().index))

        axis_report: dict[str, Any] = {
            "coverage": {
                model: {
                    "n_months_available": int(series[model].dropna().shape[0]),
                    "n_months_in_comparison": int(len(common)),
                }
                for model in MODEL_NAMES
            },
            "skipped_folds": skipped_folds,
            "occupancy": build_occupancy_report(hmm_result),
            "splits": {},
        }

        for split in splits:
            split_index = _split_index(common, dev_end, split)
            climatology_p = series["climatology"].loc[split_index]
            split_label = label.loc[split_index]
            axis_report["splits"][split] = {
                model: _model_metrics(series[model].loc[split_index], climatology_p, split_label)
                for model in MODEL_NAMES
            }

        axis_report["detection_lag_months"] = {
            model: detection_lag(series[model], usrec_monthly) for model in MODEL_NAMES
        }

        report["axes"][axis.value] = axis_report

    # Quadrant-level log loss per model: needs both axes' predictions AND both
    # axes' true labels on the same month, a separate (narrower) intersection
    # from either axis's own comparison window above.
    quadrant_by_split: dict[str, dict[str, float]] = {split: {} for split in splits}
    growth_series = series_by_axis[Axis.GROWTH]
    inflation_series = series_by_axis[Axis.INFLATION]
    for model in MODEL_NAMES:
        common_quadrant = (
            pd.DatetimeIndex(growth_series[model].dropna().index)
            .intersection(pd.DatetimeIndex(inflation_series[model].dropna().index))
            .intersection(pd.DatetimeIndex(labels.growth.dropna().index))
            .intersection(pd.DatetimeIndex(labels.inflation.dropna().index))
        )
        for split in splits:
            split_index = _split_index(common_quadrant, dev_end, split)
            quadrant_by_split[split][model] = quadrant_log_loss(
                growth_series[model].loc[split_index],
                inflation_series[model].loc[split_index],
                labels.growth.loc[split_index],
                labels.inflation.loc[split_index],
            )
    report["quadrant_log_loss"] = quadrant_by_split

    return report


def _axis_split_table(axis_report: dict[str, Any], split: str) -> str:
    splits = axis_report["splits"]
    split_metrics = splits[split]
    rows = []
    for model in MODEL_NAMES:
        m = split_metrics[model]
        skill = f"{m['skill_vs_climatology']:.3f} [{m['skill_vs_climatology_ci_low']:.3f}, {m['skill_vs_climatology_ci_high']:.3f}]"
        rows.append(
            {
                "model": model,
                "n": m["n_months"],
                "log_loss": round(m["log_loss"], 4),
                "skill_vs_climatology [95% CI]": skill,
                "auc": round(m["auc"], 3) if m["auc"] is not None else None,
                "accuracy": round(m["accuracy"], 3),
                "ece": round(m["expected_calibration_error"], 4),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def _behavior_table(axis_report: dict[str, Any], split: str) -> str:
    splits = axis_report["splits"]
    split_metrics = splits[split]
    rows = []
    for model in MODEL_NAMES:
        m = split_metrics[model]
        rows.append(
            {
                "model": model,
                "flip_rate_per_year": round(m["flip_rate_per_year"], 2),
                "mean_regime_duration_months": round(m["mean_regime_duration_months"], 2),
            }
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def _coverage_table(axis_report: dict[str, Any]) -> str:
    coverage = axis_report["coverage"]
    rows = [
        {
            "model": model,
            "n_months_available": coverage[model]["n_months_available"],
            "n_months_in_comparison": coverage[model]["n_months_in_comparison"],
        }
        for model in MODEL_NAMES
    ]
    return pd.DataFrame(rows).to_markdown(index=False)


def _occupancy_overall_table(axis_report: dict[str, Any]) -> str:
    overall = axis_report["occupancy"]["overall"]
    rows = [
        {
            "state": state,
            "n_months": stats["n_months"],
            "fraction": round(stats["fraction"], 3),
            "first_date": stats["first_date"],
            "last_date": stats["last_date"],
        }
        for state, stats in overall.items()
    ]
    if not rows:
        return "No evaluated months (every fold skipped)."
    return pd.DataFrame(rows).to_markdown(index=False)


def _occupancy_by_fold_table(axis_report: dict[str, Any]) -> str:
    by_fold = axis_report["occupancy"]["by_fold"]
    if not by_fold:
        return "No evaluated folds."
    feature_names = sorted(next(iter(by_fold[0]["state_means"].values())).keys())
    rows = []
    for fold in by_fold:
        for state, means in fold["state_means"].items():
            occupancy = fold["occupancy"].get(state, {})
            row: dict[str, Any] = {
                "fold_id": fold["fold_id"],
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
                "state": state,
                "fraction": round(occupancy["fraction"], 3) if occupancy else 0.0,
            }
            for name in feature_names:
                row[name] = round(means[name], 2)
            rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def _quadrant_table(report: dict[str, Any]) -> str:
    quadrant = report["quadrant_log_loss"]
    rows = []
    for model in MODEL_NAMES:
        row: dict[str, Any] = {"model": model}
        for split, by_model in quadrant.items():
            row[f"{split}_log_loss"] = round(by_model[model], 4)
        rows.append(row)
    return pd.DataFrame(rows).to_markdown(index=False)


def _detection_lag_section(report: dict[str, Any]) -> str:
    growth_report = report["axes"]["growth"]
    lags = growth_report["detection_lag_months"]
    lines = ["| model | detection lags (months, per recession) |", "|---|---|"]
    for model in MODEL_NAMES:
        values = ", ".join("never" if lag is None else str(lag) for lag in lags[model])
        lines.append(f"| {model} | {values or '(no recessions in scored window)'} |")
    return "\n".join(lines)


def _skipped_folds_section(report: dict[str, Any]) -> str:
    lines = []
    for axis_name in AXES:
        axis_report = report["axes"][axis_name]
        for model in MODEL_NAMES:
            for fold in axis_report["skipped_folds"][model]:
                lines.append(
                    f"- **{axis_name}/{model}**, fold {fold['fold_id']} "
                    f"(test {fold['test_start']} to {fold['test_end']}): {fold['reason']}"
                )
    return "\n".join(lines) if lines else "None."


def render_evaluation_markdown(report: dict[str, Any], metadata: dict[str, Any]) -> str:
    """Human-readable companion to the full `evaluation.json` this report dict
    becomes — one table per axis x split (rows=models), plus quadrant, behaviour,
    coverage/skips, and the exact label definitions + provenance, matching
    `training/report.render_model_card`'s own "give a human the numbers, not just
    a verdict" style.
    """
    splits_present = ("dev", "holdout", "all") if report["include_holdout"] else ("dev", "all")

    axis_sections = []
    for axis_name in AXES:
        axis_report = report["axes"][axis_name]
        split_sections = []
        for split in splits_present:
            split_sections.append(
                f"### {split}\n\n{_axis_split_table(axis_report, split)}\n\n"
                f"#### Behaviour ({split})\n\n{_behavior_table(axis_report, split)}"
            )
        axis_sections.append(
            f"## {axis_name.capitalize()} axis\n\n"
            + "\n\n".join(split_sections)
            + f"\n\n### Coverage\n\n{_coverage_table(axis_report)}"
            + "\n\n### HMM state occupancy (is a state just one episode?)\n\n"
            f"{_occupancy_overall_table(axis_report)}\n\n"
            "#### By fold (per-feature emission means, z-units)\n\n"
            f"{_occupancy_by_fold_table(axis_report)}"
        )

    return f"""# Evaluation report

Dataset `{metadata.get("dataset_id")}`, feature config `{metadata.get("feature_config_version")}`
(hash `{metadata.get("feature_config_hash")}`), git sha `{metadata.get("git_sha")}`.
Scored `{metadata.get("date_start")}` to `{metadata.get("date_end")}`.
Label lag: {report["label_lag_months"]} month(s). Dev/holdout split at `{report["dev_end"]}`
({"holdout included" if report["include_holdout"] else "holdout omitted"}).

## Label definitions (pre-registered, see training/labels.py)

- **growth**: 1.0 if CFNAI-MA3 > 0, else 0.0.
- **growth_recession** (secondary, sanity check only): 1 - USREC.
- **inflation**: 1.0 if 3-month-annualized CPI inflation exceeds its own
  trailing year-over-year rate 3 months prior, else 0.0.

## Caveats

- Walk-forward's filtered posterior is re-run from the first month through each
  fold's test window under that fold's refit parameters, keeping only the test
  months (`training/evaluation.walk_forward_evaluate`), so the filter's memory
  carries across refit boundaries, as in production inference.
- Small sample: wide confidence intervals are expected and should not be
  over-interpreted as "no difference."

{"".join(s + chr(10) + chr(10) for s in axis_sections)}## Quadrant-level log loss

{_quadrant_table(report)}

## Recession detection lag (growth axis)

{_detection_lag_section(report)}

## Walk-forward coverage gaps (skipped folds)

{_skipped_folds_section(report)}
"""
