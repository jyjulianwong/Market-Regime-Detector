import numpy as np


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """A minimal `logsumexp` — avoids adding scipy as a dependency for one function.

    A real bug, caught by `preview_posterior`'s zero-transition-probability test,
    not a hypothetical: when every value along `axis` is `-inf` (a genuinely
    impossible outcome — e.g. a state with zero probability of being reached from
    any predecessor), `a_max` is itself `-inf`, and `a - a_max` becomes `-inf -
    (-inf) = nan` under IEEE arithmetic, poisoning the result that should just be
    `-inf`. `safe_max` substitutes a finite placeholder for that subtraction only;
    the real (possibly `-inf`) `a_max` is still what gets added back below, so an
    all-impossible row correctly comes out `-inf`, not `nan`.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    safe_max = np.where(np.isneginf(a_max), 0.0, a_max)
    result = a_max + np.log(np.sum(np.exp(a - safe_max), axis=axis, keepdims=True))
    squeezed: np.ndarray = np.squeeze(result, axis=axis)
    return squeezed


def _log_gaussian_diag(
    observations: np.ndarray, means: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """Log density of a diagonal-covariance Gaussian: `(T, n_states)`."""
    n_states = means.shape[0]
    log_probs = np.zeros((observations.shape[0], n_states))
    for state in range(n_states):
        diff = observations - means[state]
        log_probs[:, state] = -0.5 * np.sum(
            np.log(2 * np.pi * variances[state]) + diff**2 / variances[state], axis=1
        )
    return log_probs


def filtered_posterior(
    observations: np.ndarray,
    start_prob: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Causal forward (filtered) posterior for a diagonal-Gaussian-emission HMM.

    `P(state_t | obs_1..obs_t)` for every `t` — never uses future observations, unlike
    `hmmlearn`'s `predict_proba` (smoothed) or a Viterbi path. This is the only
    inference primitive this project uses (brief section 2.7): training's
    walk-forward evaluation and the Phase 6 batch Lambda both call this directly, not
    `hmmlearn` at inference time.

    Args:
        observations: `(T, n_features)`, already standardized by the same scaler
            used at training time.
        start_prob: `(n_states,)`.
        transition_matrix: `(n_states, n_states)`, row `i` = `P(state_t=j |
            state_{t-1}=i)`.
        means: `(n_states, n_features)`.
        variances: `(n_states, n_features)` — diagonal covariance only.

    Returns:
        `(T, n_states)`, each row summing to 1.
    """
    log_emission = _log_gaussian_diag(observations, means, variances)
    n_timesteps, n_states = log_emission.shape

    # A fitted HMM legitimately has exact-zero start/transition probabilities (e.g.
    # a state never used as an initial state); log(0) = -inf is the correct log-space
    # representation of that, not an error, so the resulting NumPy warning is
    # suppressed rather than left as noise on every normal call.
    with np.errstate(divide="ignore"):
        log_transition = np.log(transition_matrix)
        log_alpha = np.zeros((n_timesteps, n_states))
        log_alpha[0] = np.log(start_prob) + log_emission[0]
    log_alpha[0] -= _logsumexp(log_alpha[0], axis=0)

    for t in range(1, n_timesteps):
        prev = log_alpha[t - 1][:, None] + log_transition
        log_alpha[t] = _logsumexp(prev, axis=0) + log_emission[t]
        log_alpha[t] -= _logsumexp(log_alpha[t], axis=0)

    return np.exp(log_alpha)


def preview_posterior(
    prev_alpha: np.ndarray,
    transition_matrix: np.ndarray,
    observations: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """One forward-filter step applied independently to each row of `observations`,
    from a fixed prior `prev_alpha` — never recursed across rows.

    Used by `prediction/predict.py` to turn a monthly filtered posterior into a
    daily preview: every day in month `M` previews `P(state | month M-1's filtered
    posterior, that day's own partial-month features)`, so the transition matrix
    (learned on monthly steps) is applied exactly once per calendar month no matter
    how many daily rows are requested for that month — unlike feeding ~30
    near-identical daily observations through `filtered_posterior` directly, which
    applies a month's worth of transition probability per day.

    When `observations` is the single month-end row that produces `prev_alpha`'s
    successor under `filtered_posterior`, this returns exactly that successor row —
    it is the same recursion step, just factored out so it can be evaluated against
    many candidate observations (a month's worth of days) without feeding any of
    them back into each other.

    Args:
        prev_alpha: `(n_states,)` (broadcast to every observation row) or `(N,
            n_states)` (one anchor row per observation).
        transition_matrix: `(n_states, n_states)`.
        observations: `(N, n_features)`, already standardized.
        means: `(n_states, n_features)`.
        variances: `(n_states, n_features)` — diagonal covariance only.

    Returns:
        `(N, n_states)`, each row summing to 1.
    """
    log_emission = _log_gaussian_diag(observations, means, variances)
    n_observations, n_states = log_emission.shape

    prev_alpha = np.asarray(prev_alpha, dtype=float)
    if prev_alpha.ndim == 1:
        prev_alpha = np.broadcast_to(prev_alpha, (n_observations, n_states))

    # Same log(0) = -inf reasoning as `filtered_posterior` above.
    with np.errstate(divide="ignore"):
        log_prev_alpha = np.log(prev_alpha)
        log_transition = np.log(transition_matrix)
        predicted = log_prev_alpha[:, :, None] + log_transition[None, :, :]
        log_predicted = _logsumexp(predicted, axis=1)
        log_posterior = log_predicted + log_emission
        log_posterior -= _logsumexp(log_posterior, axis=1)[:, None]

    result: np.ndarray = np.exp(log_posterior)
    return result
