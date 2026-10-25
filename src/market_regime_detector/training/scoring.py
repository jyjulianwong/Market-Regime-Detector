import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Floor/ceiling for a probability before it enters a log — an exact 0 or 1 would
# give log loss an infinite penalty for a single miss, which would make one
# saturated prediction dominate an entire split's score.
CLIP_EPS = 1e-4


def _align_and_drop_na(p: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Common alignment for every pairwise metric below: an inner join on index
    (never a positional zip — callers may have gaps from skipped walk-forward
    folds or a missing label), then drop any row where either side is `NaN`.
    """
    aligned = pd.concat([p.rename("p"), y.rename("y")], axis=1, join="inner").dropna()
    return aligned["p"].to_numpy(dtype=float), aligned["y"].to_numpy(dtype=float)


def log_loss(p: pd.Series, y: pd.Series) -> float:
    """`-mean(y*log(p) + (1-y)*log(1-p))` — the primary scoring metric. Proper and
    strictly penalizes overconfident wrong calls, unlike accuracy.
    """
    p_arr, y_arr = _align_and_drop_na(p, y)
    p_clipped = np.clip(p_arr, CLIP_EPS, 1 - CLIP_EPS)
    return float(-np.mean(y_arr * np.log(p_clipped) + (1 - y_arr) * np.log(1 - p_clipped)))


def brier(p: pd.Series, y: pd.Series) -> float:
    p_arr, y_arr = _align_and_drop_na(p, y)
    return float(np.mean((p_arr - y_arr) ** 2))


def skill_score(model_loss: float, climatology_loss: float) -> float:
    """`1 - model/climatology`: positive means the model beats always-predicting-
    the-historical-base-rate, zero means it's exactly as good, negative means
    worse. The headline comparison number — a raw log loss alone isn't
    interpretable without a reference.
    """
    return 1.0 - model_loss / climatology_loss


def auc(p: pd.Series, y: pd.Series) -> float | None:
    """`None` when the aligned sample has only one class present — AUC is
    undefined there, not trivially 0 or 1. A caller seeing `None` a lot for one
    axis/split has a label-imbalance problem worth flagging, not a bug.
    """
    p_arr, y_arr = _align_and_drop_na(p, y)
    if len(np.unique(y_arr)) < 2:
        return None
    return float(roc_auc_score(y_arr, p_arr))


def accuracy(p: pd.Series, y: pd.Series) -> float:
    p_arr, y_arr = _align_and_drop_na(p, y)
    predicted_up = (p_arr >= 0.5).astype(float)
    return float(np.mean(predicted_up == y_arr))


def calibration_table(p: pd.Series, y: pd.Series, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Bins `p` into `n_bins` equal-width `[0, 1]` buckets. Returns `(table, ece)`:
    `table` has one row per non-empty bin (`bin`, `n`, `mean_p`, `mean_y`); `ece`
    (expected calibration error) is the `n`-weighted mean absolute gap between
    `mean_p` and `mean_y` across those rows — `nan` if there's no data to bin.
    """
    p_arr, y_arr = _align_and_drop_na(p, y)
    if len(p_arr) == 0:
        return pd.DataFrame(columns=["bin", "n", "mean_p", "mean_y"]), float("nan")

    bins = pd.cut(p_arr, bins=list(np.linspace(0, 1, n_bins + 1)), include_lowest=True)
    frame = pd.DataFrame({"bin": bins, "p": p_arr, "y": y_arr})
    table = (
        frame.groupby("bin", observed=True)
        .agg(n=("p", "size"), mean_p=("p", "mean"), mean_y=("y", "mean"))
        .reset_index()
    )
    ece = float(np.average(np.abs(table["mean_p"] - table["mean_y"]), weights=table["n"]))
    return table, ece


def quadrant_log_loss(
    p_growth: pd.Series,
    p_inflation: pd.Series,
    y_growth: pd.Series,
    y_inflation: pd.Series,
) -> float:
    """`-mean(log P(true quadrant))`, with `P` from the same independence-product
    formula as `common/quadrants.quadrant_probabilities` (vectorized directly here
    rather than calling that dict-returning function row by row) — scores the full
    joint quadrant call, not each axis in isolation.
    """
    frame = pd.concat(
        [
            p_growth.rename("pg"),
            p_inflation.rename("pi"),
            y_growth.rename("yg"),
            y_inflation.rename("yi"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    pg = frame["pg"].clip(CLIP_EPS, 1 - CLIP_EPS)
    pi = frame["pi"].clip(CLIP_EPS, 1 - CLIP_EPS)
    yg = frame["yg"]
    yi = frame["yi"]
    p_true = (pg**yg * (1 - pg) ** (1 - yg)) * (pi**yi * (1 - pi) ** (1 - yi))
    return float(-np.log(p_true).mean())


def block_bootstrap_ci(
    per_month_values_a: pd.Series,
    per_month_values_b: pd.Series | None = None,
    block: int = 6,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Moving-block bootstrap `ci`-level interval for the mean of
    `per_month_values_a` (or the mean difference `a - b`, paired on `a`'s index,
    when `per_month_values_b` is given). Regimes are strongly autocorrelated month
    to month, so an i.i.d. bootstrap would be far too narrow — resampling
    contiguous blocks of `block` months preserves that autocorrelation structure.

    Returns `(point_estimate, ci_low, ci_high)`. Falls back to a degenerate
    `(point_estimate, point_estimate, point_estimate)` when there are fewer
    observations than `block` — too little data to block-resample meaningfully.
    """
    if per_month_values_b is not None:
        paired = pd.concat(
            [per_month_values_a.rename("a"), per_month_values_b.rename("b")],
            axis=1,
            join="inner",
        ).dropna()
        values = (paired["a"] - paired["b"]).to_numpy(dtype=float)
    else:
        values = per_month_values_a.dropna().to_numpy(dtype=float)

    n = len(values)
    point_estimate = float(np.mean(values)) if n > 0 else float("nan")
    if n < block:
        return point_estimate, point_estimate, point_estimate

    rng = np.random.default_rng(seed)
    n_blocks_needed = -(-n // block)  # ceil(n / block)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        block_starts = rng.integers(0, n - block + 1, size=n_blocks_needed)
        resample = np.concatenate([values[s : s + block] for s in block_starts])[:n]
        boot_means[i] = resample.mean()

    low, high = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return point_estimate, float(low), float(high)


def flip_rate_per_year(p: pd.Series) -> float:
    """Number of `p >= 0.5` sign flips, annualized from however many months `p`
    actually spans — comparable across models/splits with different month counts.
    """
    p_sorted = p.dropna().sort_index()
    if len(p_sorted) < 2:
        return 0.0
    state = (p_sorted >= 0.5).astype(int)
    n_flips = int((state.diff().abs() == 1).sum())
    return n_flips * 12.0 / len(p_sorted)


def mean_regime_duration_months(p: pd.Series) -> float:
    """Average length, in months, of a run where `p >= 0.5` stays constant — a
    duration-flavored complement to `flip_rate_per_year`, not just its reciprocal
    (a model that's confidently up for 11 months then down for 1 has the same flip
    rate as one that alternates every 6, but very different average duration).
    """
    p_sorted = p.dropna().sort_index()
    if p_sorted.empty:
        return float("nan")
    state = (p_sorted >= 0.5).astype(int)
    run_id = (state != state.shift()).cumsum()
    run_lengths = state.groupby(run_id).size()
    return float(run_lengths.mean())


def detection_lag(p_growth_up: pd.Series, usrec: pd.Series) -> list[int | None]:
    """For each NBER recession start inside `p_growth_up`'s own scored window (a
    month where `usrec` goes `0 -> 1`), the number of months from that start to
    the first later month with `p_growth_up < 0.5` — `None` if growth never dips
    below 0.5 before `p_growth_up`'s series ends. A recession start before or after
    `p_growth_up`'s own index range is excluded, not counted as "never detected."
    """
    usrec_sorted = usrec.dropna().sort_index()
    starts = usrec_sorted.index[(usrec_sorted == 1) & (usrec_sorted.shift(1) == 0)]

    lags: list[int | None] = []
    for start in starts:
        if start not in p_growth_up.index:
            continue
        window = p_growth_up.loc[start:]
        below = window[window < 0.5]
        if below.empty:
            lags.append(None)
            continue
        detected_at = below.index[0]
        lags.append((detected_at.year - start.year) * 12 + (detected_at.month - start.month))
    return lags
