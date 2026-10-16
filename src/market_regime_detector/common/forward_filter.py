import numpy as np


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """A minimal `logsumexp` — avoids adding scipy as a dependency for one function."""
    a_max = np.max(a, axis=axis, keepdims=True)
    result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
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
