"""v4 posterior calibration (`docs/model_cards/v4/MODEL_CARD.md`, experiment E1).

`prediction/predict.py::_axis_posteriors` emits the HMM's raw filtered
posterior, `P(axis up)`. Phase 4's walk-forward evaluation (F1 in the v4
report) found this to be well-ranked (decent AUC) but badly calibrated
(negative skill vs. climatology, high ECE on both axes) — the probabilities
themselves don't match observed frequencies, even though the model correctly
orders which months are more/less likely UP.

**Outcome (see the v4 model card for the full comparison table): adopted
`LinearShrinkCalibrator` for production, rejected Platt scaling.** Both were
evaluated causally (fit only on strictly-earlier OOS posteriors, applied to
later ones — `expanding_window_calibrate` for Platt, the same discipline
inlined in `scripts/evaluate_calibration.py` for shrink) against the same
acceptance bar (section 6: log loss, AUC and skill must all improve, AUC in
particular must not regress, since AUC-neutral is the whole point of a
*calibration* fix as opposed to a ranking fix). Platt, refit at every
walk-forward fold boundary, measurably changed AUC (it isn't one global
monotone map once different folds get different `a, b`) and left growth's
skill still negative — it does not clear the bar. A single global linear
shrink toward the label's historical base rate, `weight * p_raw + (1 -
weight) * base_rate`, is *exactly* rank-preserving (one affine map applied
everywhere), and took both axes from negative to positive skill with a large
ECE drop. Platt scaling is kept in this module, fully causal and tested, as a
documented negative result, not dead code — see the model card for why a
reader might reasonably expect it to win and didn't here (mainly: ~90-150
OOS points, refit from scratch at every expanding-window step, is too few for
a freshly-fit 2-parameter logistic regression to be stable, even after fixing
the separation-induced divergence noted on `fit_platt_calibrator` below).

`identity_calibrator()` / `LinearShrinkCalibrator(weight=1.0, ...)` is the
artifact default — reproduces the raw posterior unchanged, so a model
published before this module existed (no calibrator keys in its
`scalers.npz`) predicts unchanged, per the v4 report's backward-compatibility
constraint (section 6).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Keeps `logit` finite for p in {0, 1} — the raw HMM posterior can be an exact
# 0 or 1 in principle (see `common/forward_filter.py`'s log(0)-is-correct note);
# clipping here, not there, since this module is the one taking a logit.
_PROB_EPS = 1e-6


@dataclass(frozen=True)
class PlattCalibrator:
    a: float
    b: float

    def apply(self, p_raw: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
        return _sigmoid(self.a * _logit(p_raw) + self.b)


def identity_calibrator() -> PlattCalibrator:
    return PlattCalibrator(a=1.0, b=0.0)


def _logit(p: pd.Series | np.ndarray) -> np.ndarray:
    p_arr = np.asarray(p, dtype=float)
    clipped = np.clip(p_arr, _PROB_EPS, 1 - _PROB_EPS)
    result: np.ndarray = np.log(clipped / (1 - clipped))
    return result


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt_calibrator(
    p_raw: pd.Series, label: pd.Series, *, max_iter: int = 100, tol: float = 1e-10
) -> PlattCalibrator:
    """Fits `a, b` by Newton's method on the 2-parameter logistic regression
    log-likelihood, `x = logit(p_raw)` as the lone feature. `p_raw` and `label`
    are aligned on their index first (an expanding-window caller only ever
    passes the OOS posteriors available up to some point, never future ones —
    enforcing that is the caller's job, e.g. `expanding_window_calibrate`
    below, not this function's).

    Fits against **Platt's (1999) soft targets**, not the raw 0/1 `label`
    directly: `t = (n_pos + 1) / (n_pos + 2)` where `label == 1`, `t = 1 /
    (n_neg + 2)` where `label == 0`. A first implementation here used raw 0/1
    targets and it diverged on growth's ~90-point OOS sample — the raw HMM
    posterior is well-ranked enough (AUC ~0.81) to be *quasi-separable* on a
    sample this small, and unregularized logistic regression on separable data
    has no finite maximum-likelihood solution (the fit pushes `|a|` to
    infinity chasing a vanishing gradient). Platt's own paper introduces this
    smoothing for exactly that reason, not as an approximation of convenience —
    without it, Platt scaling is not well-posed on a small sample, which this
    project's growth axis (`HMM_ACCURACY_PLAN.md`'s ~150-row sample) is.

    Falls back to the identity map when there isn't enough signal to fit: no
    rows after alignment, or the label has no variation (all UP or all DOWN) —
    the soft-target smoothing above still needs both classes to be non-empty.
    """
    aligned = pd.concat([p_raw.rename("p"), label.rename("y")], axis=1).dropna()
    if aligned.empty or aligned["y"].nunique() < 2:
        return identity_calibrator()

    x = _logit(aligned["p"].to_numpy())
    y_raw = aligned["y"].to_numpy(dtype=float)
    n_pos = float(np.sum(y_raw == 1))
    n_neg = float(np.sum(y_raw == 0))
    target_pos = (n_pos + 1.0) / (n_pos + 2.0)
    target_neg = 1.0 / (n_neg + 2.0)
    y = np.where(y_raw == 1, target_pos, target_neg)

    def _neg_log_lik(a_: float, b_: float) -> float:
        p_ = np.clip(_sigmoid(a_ * x + b_), 1e-12, 1 - 1e-12)
        return float(-np.sum(y * np.log(p_) + (1 - y) * np.log(1 - p_)))

    a, b = 1.0, 0.0
    current_nll = _neg_log_lik(a, b)
    for _ in range(max_iter):
        p = _sigmoid(a * x + b)
        grad_a = float(np.sum((y - p) * x))
        grad_b = float(np.sum(y - p))
        w = np.clip(p * (1 - p), 1e-10, None)
        hess_aa = -float(np.sum(w * x * x))
        hess_ab = -float(np.sum(w * x))
        hess_bb = -float(np.sum(w))
        det = hess_aa * hess_bb - hess_ab * hess_ab
        if abs(det) < 1e-12:
            break
        delta_a = -(hess_bb * grad_a - hess_ab * grad_b) / det
        delta_b = -(-hess_ab * grad_a + hess_aa * grad_b) / det

        # Backtracking line search: a full Newton step on this tiny, nearly
        # separable sample can overshoot wildly (confirmed empirically — the
        # undamped version of this loop diverged to |a| > 1e5 on growth's
        # ~90-point OOS sample). Halving the step until the negative
        # log-likelihood actually improves is the standard safeguard, and is
        # what keeps this a convex-optimization routine rather than a fragile
        # one that only works when the first step happens to land well.
        step = 1.0
        for _ in range(30):
            a_new, b_new = a + step * delta_a, b + step * delta_b
            new_nll = _neg_log_lik(a_new, b_new)
            if new_nll <= current_nll + 1e-12:
                break
            step *= 0.5
        else:
            break

        converged = abs(step * delta_a) < tol and abs(step * delta_b) < tol
        a, b, current_nll = a_new, b_new, new_nll
        if converged:
            break

    return PlattCalibrator(a=float(a), b=float(b))


def expanding_window_calibrate(walk_forward_df: pd.DataFrame, true_label: pd.Series) -> pd.Series:
    """Applies Platt calibration to `walk_forward_evaluate`'s `p_up` column
    causally: fold `k`'s rows are calibrated using a calibrator fit only on
    folds `< k`'s OOS posteriors, paired with `true_label` (the pre-registered
    ground truth from `training/labels.py` — e.g. `growth_label(cfnai_ma3)` —
    **not** `walk_forward_df["label"]`, which is just the HMM's own UP/DOWN
    classification string at `p_up >= 0.5` and would make this circular). Fold
    0 (no prior folds) gets the identity map — there's nothing earlier to fit
    on, same reasoning as `fit_platt_calibrator`'s own fallback.

    This is the causal evaluation path only, reusing the walk-forward's
    existing fold boundaries as the expanding-window refit cadence (so it
    never invents a second, inconsistent notion of "how often to refit") — it
    is **not** what a published model uses: `training/build_model.py` fits one
    calibrator on the *entire* OOS series for the production artifact, the
    same way every other trained parameter in this project is fit once at
    train time and then applied going forward.
    """
    indexed = walk_forward_df.set_index("decision_date")
    out = pd.Series(index=walk_forward_df.index, dtype=float)
    fold_ids = sorted(walk_forward_df["fold_id"].dropna().unique())
    for fold_id in fold_ids:
        is_fold = walk_forward_df["fold_id"] == fold_id
        prior = indexed.loc[indexed["fold_id"] < fold_id]
        if prior.empty:
            out.loc[is_fold] = walk_forward_df.loc[is_fold, "p_up"]
            continue
        calibrator = fit_platt_calibrator(prior["p_up"], true_label.reindex(prior.index))
        raw = walk_forward_df.loc[is_fold, "p_up"]
        calibrated = calibrator.apply(raw.to_numpy())
        out.loc[is_fold] = calibrated
    return out


#: The constant used throughout E1 — a round, pre-chosen number (same spirit
#: as this project's other documented judgment-call constants, e.g.
#: `common/scaling.DEFAULT_CLIP`), not fit/tuned against this sample. The v4
#: report proposed "a simple shrink toward the base rate" as a deliberately
#: dumb comparison point for Platt scaling, not a parameter to search over —
#: searching `weight` for the best score would be exactly the "tune on
#: holdout" the project's constraints (section 6) rule out.
DEFAULT_SHRINK_WEIGHT = 0.5


@dataclass(frozen=True)
class LinearShrinkCalibrator:
    """The production calibrator adopted for v4 (see module docstring):
    `calibrated = weight * p_raw + (1 - weight) * base_rate`, a single global
    affine map in probability space (not logit space — the HMM's raw
    posterior is already a probability, and this map's whole appeal is that
    it's simple enough to need no per-fold refitting). `weight = 1.0` is the
    identity map regardless of `base_rate`, which is what a pre-v4 artifact's
    missing calibrator keys default to.
    """

    weight: float
    base_rate: float

    def apply(self, p_raw: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
        return self.weight * p_raw + (1 - self.weight) * self.base_rate


def identity_shrink_calibrator() -> LinearShrinkCalibrator:
    return LinearShrinkCalibrator(weight=1.0, base_rate=0.0)


def fit_shrink_calibrator(
    label: pd.Series, *, weight: float = DEFAULT_SHRINK_WEIGHT
) -> LinearShrinkCalibrator:
    """`base_rate` is the only fitted quantity — the historical UP fraction of
    `label` (same information a climatology baseline already uses elsewhere in
    this project), computed once over all available label history at train
    time, then baked into the artifact like any other trained parameter
    (`growth_scaler_mean`/`scale`, the HMM's own `means`/`variances`). It is
    never refit per fold or per prediction — that's what makes this safe to
    apply with zero look-ahead at inference time: by the time a given month's
    label would be knowable, the artifact (and its `base_rate`) was already
    fixed at training time, strictly before that month.
    """
    return LinearShrinkCalibrator(weight=weight, base_rate=float(label.dropna().mean()))


def shrink_to_base_rate(p_raw: pd.Series, label: pd.Series, *, weight: float = 0.5) -> pd.Series:
    """A **non-causal, upper-bound** comparison number only: fits one
    `base_rate` from the *entire* OOS sample's label (including, for an early
    test month, labels from months that come later) and applies it everywhere.
    `scripts/evaluate_calibration.py` reports this alongside
    `expanding_window_shrink` below (the fair, apples-to-apples-with-Platt
    comparison) specifically so the gap between the two is visible — if this
    non-causal version and the causal version score similarly, that's
    evidence the base rate is stable enough over time that the causal
    restriction isn't costing much; it is not itself evidence that the
    non-causal number is safe to use as the actual acceptance-bar result.
    """
    calibrator = fit_shrink_calibrator(label, weight=weight)
    result: pd.Series = calibrator.apply(p_raw)  # type: ignore[assignment]
    return result


def expanding_window_shrink(
    walk_forward_df: pd.DataFrame, true_label: pd.Series, *, weight: float = DEFAULT_SHRINK_WEIGHT
) -> pd.Series:
    """The causal counterpart to `shrink_to_base_rate`, fit the same way
    `expanding_window_calibrate` fits Platt: fold `k`'s rows get a `base_rate`
    computed only from folds `< k`'s true labels. Fold 0 gets the identity map
    (nothing earlier to compute a base rate from). This is the fair
    comparison against Platt's own causal numbers — `shrink_to_base_rate`
    above is a non-causal upper bound, kept only as a sanity check on how much
    the causal restriction costs.
    """
    indexed = walk_forward_df.set_index("decision_date")
    out = pd.Series(index=walk_forward_df.index, dtype=float)
    fold_ids = sorted(walk_forward_df["fold_id"].dropna().unique())
    for fold_id in fold_ids:
        is_fold = walk_forward_df["fold_id"] == fold_id
        prior = indexed.loc[indexed["fold_id"] < fold_id]
        if prior.empty:
            out.loc[is_fold] = walk_forward_df.loc[is_fold, "p_up"]
            continue
        prior_label = true_label.reindex(prior.index).dropna()
        if prior_label.empty:
            out.loc[is_fold] = walk_forward_df.loc[is_fold, "p_up"]
            continue
        calibrator = fit_shrink_calibrator(prior_label, weight=weight)
        raw = walk_forward_df.loc[is_fold, "p_up"]
        out.loc[is_fold] = calibrator.apply(raw.to_numpy())
    return out


def hard_cap(p_raw: pd.Series, *, low: float = 0.10, high: float = 0.90) -> pd.Series:
    """The v4 report's stopgap fallback if Platt calibration doesn't clear its
    acceptance bar on an axis: clip displayed/served probabilities into
    `[low, high]`. Implemented as its own function (not expressed as
    `PlattCalibrator` params) because a hard clip isn't a sigmoid-of-affine-
    logit map — it has a flat region a 2-parameter Platt fit can't represent.
    """
    return p_raw.clip(lower=low, upper=high)
