import numpy as np

# `HMM_ACCURACY_PLAN.md` Phase 1 default: clip standardized z-scores to +/-3.
# Chosen by a real-data ablation over clip in {1.5, 2, 2.5, 3, 4, 5} x method in
# {winsorized, median_mad} (recorded in ARCHITECTURE.md) — 3.0 beat every looser
# clip on every metric (log loss, AUC, accuracy, skill vs. climatology, skipped
# folds) on both axes, and beat every tighter clip tried too. Change the
# ARCHITECTURE.md note first if this default ever moves.
DEFAULT_CLIP = 3.0

# The constant that makes MAD (median absolute deviation) a consistent estimator
# of the standard deviation under a normal distribution: 1 / Phi^-1(3/4).
_MAD_TO_STD = 1.4826


def fit_robust_scaler(
    X: np.ndarray, *, clip: float = DEFAULT_CLIP, method: str = "winsorized"
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column robust center/scale, replacing a plain
    `sklearn.preprocessing.StandardScaler` wherever a training-window outlier
    would otherwise inflate that column's scale for every row it's ever applied
    to (`HMM_ACCURACY_PLAN.md` Phase 1 problem #1 — e.g. ICSA's March 2020 spike
    leaves ICSA near zero everywhere else once the whole training window shares
    one outlier-inflated std).

    `method="winsorized"` (default): clips each column to its own 1st/99th
    percentile, then takes the mean/std of the *clipped* column — the outlier
    still exists in the data but no longer dominates `scale`.
    `method="median_mad"`: center = median, scale = MAD * 1.4826 — immune to the
    outlier entirely, at the cost of being a less efficient estimator when a
    column really is close to normal with no outliers. Phase 1's ablation
    (see ARCHITECTURE.md) picked between the two on this project's real features.

    `clip` is accepted here (not just by `apply_scaler`) purely so a caller can
    pass one `clip` value to both functions without it living in two places —
    fitting itself never clips anything; only `apply_scaler` does. Guards
    `scale == 0` (a constant column) to `1.0`, never dividing by zero or
    propagating a NaN from a feature that happens not to vary in-sample.
    """
    if method == "winsorized":
        lower = np.percentile(X, 1, axis=0)
        upper = np.percentile(X, 99, axis=0)
        winsorized = np.clip(X, lower, upper)
        center = winsorized.mean(axis=0)
        scale = winsorized.std(axis=0)
    elif method == "median_mad":
        center = np.median(X, axis=0)
        scale = np.median(np.abs(X - center), axis=0) * _MAD_TO_STD
    else:
        raise ValueError(f"unknown method {method!r}; expected 'winsorized' or 'median_mad'")

    scale = np.where(scale == 0, 1.0, scale)
    result: tuple[np.ndarray, np.ndarray] = (center, scale)
    return result


def apply_scaler(
    X: np.ndarray, center: np.ndarray, scale: np.ndarray, clip: float = DEFAULT_CLIP
) -> np.ndarray:
    """Standardizes `X` with an already-fit `center`/`scale`, then clips the
    result to `[-clip, clip]` — applied identically at HMM fit time, walk-forward
    scoring time, the logistic benchmark, and live prediction, so an outlier
    month is bounded the same way everywhere it's ever scored, not just where it
    was fit (`HMM_ACCURACY_PLAN.md` Phase 1: "clipping must also be applied at
    scoring time, not just fit time").
    """
    z = (X - center) / scale
    result: np.ndarray = np.clip(z, -clip, clip)
    return result
