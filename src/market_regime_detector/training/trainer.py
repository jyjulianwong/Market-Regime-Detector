from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from market_regime_detector.common.feature_config import FeatureDefinition

Label = Literal["UP", "DOWN"]


@dataclass
class FitResult:
    feature_names: list[str]
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    start_prob: np.ndarray
    transition_matrix: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    state_labels: dict[int, Label]
    composites: dict[int, float]
    log_likelihood: float
    covariance_type: str
    n_restarts: int


def _composite_scores(means: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Brief section 2.3: each state's composite = the mean, over features, of that
    state's standardized emission mean times the feature's sign (+1 if higher means
    "up" for this axis). `means` and `signs` must share the same feature ordering.
    """
    result: np.ndarray = (means * signs).mean(axis=1)
    return result


def _expected_durations(transition_matrix: np.ndarray) -> np.ndarray:
    self_transition = np.diag(transition_matrix)
    result: np.ndarray = 1.0 / (1.0 - self_transition)
    return result


def _diag_variances(covars: np.ndarray) -> np.ndarray:
    """`hmmlearn`'s `covars_` getter always returns full `(n_states, n_features,
    n_features)` matrices for display purposes, even when `covariance_type="diag"`
    internally — extract just the diagonal so this project's own artifact format
    (and `common/forward_filter.py`) only ever deals with per-feature variances.
    """
    if covars.ndim == 2:
        return covars
    result: np.ndarray = np.diagonal(covars, axis1=1, axis2=2)
    return result


def fit_hmm(
    df: pd.DataFrame,
    feature_defs: list[FeatureDefinition],
    *,
    n_restarts: int = 10,
    covariance_type: str = "diag",
    min_state_duration_months: float = 2.0,
    composite_separation_margin: float = 0.1,
    random_state: int | None = None,
) -> FitResult:
    """Fits a 2-state Gaussian HMM on `df` (one row per monthly decision date, one
    column per feature on a single axis) and applies brief section 2.3's
    deterministic, no-label-switching labelling.

    `covariance_type="diag"` is a judgment call (the brief only specifies
    `GaussianHMM(n_components=2)`) — the standard, stable default for a handful of
    features over a few hundred monthly points.

    Multiple random restarts; among restarts whose fit **passes both sanity checks**
    (composite separation > `composite_separation_margin`, both states' expected
    duration `1/(1-self_transition_prob)` in months > `min_state_duration_months`),
    keeps the best log-likelihood. Raises `ValueError` if none pass — never silently
    returns a flickering or label-ambiguous fit.
    """
    feature_names = [f.name for f in feature_defs if f.name in df.columns]
    signs = np.array([f.sign for f in feature_defs if f.name in df.columns], dtype=float)

    scaler = StandardScaler()
    observations = scaler.fit_transform(df[feature_names].to_numpy())

    rng = np.random.default_rng(random_state)
    best: FitResult | None = None

    for _ in range(n_restarts):
        seed = int(rng.integers(0, 2**31 - 1))
        model = GaussianHMM(
            n_components=2,
            covariance_type=covariance_type,
            n_iter=100,
            random_state=seed,
        )
        model.fit(observations)

        composites = _composite_scores(model.means_, signs)
        durations = _expected_durations(model.transmat_)

        separation_ok = abs(composites[0] - composites[1]) > composite_separation_margin
        duration_ok = bool(np.all(durations > min_state_duration_months))
        if not (separation_ok and duration_ok):
            continue

        log_likelihood = float(model.score(observations))
        if best is not None and log_likelihood <= best.log_likelihood:
            continue

        up_state = int(np.argmax(composites))
        down_state = 1 - up_state
        state_labels: dict[int, Label] = {up_state: "UP", down_state: "DOWN"}

        best = FitResult(
            feature_names=feature_names,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            start_prob=model.startprob_,
            transition_matrix=model.transmat_,
            means=model.means_,
            variances=_diag_variances(model.covars_),
            state_labels=state_labels,
            composites={s: float(c) for s, c in enumerate(composites)},
            log_likelihood=log_likelihood,
            covariance_type=covariance_type,
            n_restarts=n_restarts,
        )

    if best is None:
        raise ValueError(
            f"no restart out of {n_restarts} passed sanity checks "
            f"(composite separation > {composite_separation_margin}, expected state "
            f"duration > {min_state_duration_months} months) — this feature set may "
            "not show a clear two-regime structure over this window"
        )

    return best
