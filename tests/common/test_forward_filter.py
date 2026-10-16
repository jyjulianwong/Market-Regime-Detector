import numpy as np
import pytest
from hmmlearn.hmm import GaussianHMM

from market_regime_detector.common.forward_filter import filtered_posterior

START_PROB = np.array([0.6, 0.4])
TRANSITION_MATRIX = np.array([[0.9, 0.1], [0.2, 0.8]])
MEANS = np.array([[-1.5, -1.0], [1.5, 1.0]])
VARIANCES = np.array([[0.5, 0.5], [0.5, 0.5]])


def _sample_observations(n_timesteps: int, seed: int) -> np.ndarray:
    model = GaussianHMM(n_components=2, covariance_type="diag", random_state=seed)
    model.startprob_ = START_PROB
    model.transmat_ = TRANSITION_MATRIX
    model.means_ = MEANS
    model.covars_ = VARIANCES
    observations, _states = model.sample(n_timesteps)
    result: np.ndarray = observations
    return result


def _gaussian_pdf_diag(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    return float(np.prod(1 / np.sqrt(2 * np.pi * var) * np.exp(-0.5 * (x - mean) ** 2 / var)))


def _brute_force_posterior(
    observations: np.ndarray,
    start_prob: np.ndarray,
    transition_matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """An independent, deliberately naive reference: plain probabilities, no
    log-space trick, straight off the forward-algorithm definition.
    """
    n_timesteps = observations.shape[0]
    n_states = means.shape[0]
    emission = np.array(
        [
            [_gaussian_pdf_diag(observations[t], means[s], variances[s]) for s in range(n_states)]
            for t in range(n_timesteps)
        ]
    )

    alpha = np.zeros((n_timesteps, n_states))
    alpha[0] = start_prob * emission[0]
    alpha[0] /= alpha[0].sum()
    for t in range(1, n_timesteps):
        alpha[t] = (alpha[t - 1] @ transition_matrix) * emission[t]
        alpha[t] /= alpha[t].sum()
    return alpha


def test_matches_hmmlearn_predict_proba_at_last_timestep() -> None:
    observations = _sample_observations(n_timesteps=50, seed=42)

    model = GaussianHMM(n_components=2, covariance_type="diag")
    model.startprob_ = START_PROB
    model.transmat_ = TRANSITION_MATRIX
    model.means_ = MEANS
    model.covars_ = VARIANCES
    smoothed = model.predict_proba(observations)

    filtered = filtered_posterior(observations, START_PROB, TRANSITION_MATRIX, MEANS, VARIANCES)

    # Smoothed and filtered posteriors coincide only at the final timestep — there's
    # no "future" beyond it for the smoother to use.
    np.testing.assert_allclose(filtered[-1], smoothed[-1], atol=1e-8)


def test_matches_brute_force_reference_at_every_step() -> None:
    observations = _sample_observations(n_timesteps=30, seed=7)

    filtered = filtered_posterior(observations, START_PROB, TRANSITION_MATRIX, MEANS, VARIANCES)
    reference = _brute_force_posterior(
        observations, START_PROB, TRANSITION_MATRIX, MEANS, VARIANCES
    )

    np.testing.assert_allclose(filtered, reference, atol=1e-8)


def test_each_row_sums_to_one() -> None:
    observations = _sample_observations(n_timesteps=20, seed=1)

    filtered = filtered_posterior(observations, START_PROB, TRANSITION_MATRIX, MEANS, VARIANCES)

    np.testing.assert_allclose(filtered.sum(axis=1), np.ones(20), atol=1e-8)


def test_single_timestep_matches_start_prob_posterior() -> None:
    observations = _sample_observations(n_timesteps=1, seed=3)

    filtered = filtered_posterior(observations, START_PROB, TRANSITION_MATRIX, MEANS, VARIANCES)

    emission = np.array(
        [_gaussian_pdf_diag(observations[0], MEANS[s], VARIANCES[s]) for s in range(2)]
    )
    expected = START_PROB * emission
    expected /= expected.sum()
    np.testing.assert_allclose(filtered[0], expected, atol=1e-8)


@pytest.mark.parametrize("n_timesteps", [5, 10, 100])
def test_various_sequence_lengths(n_timesteps: int) -> None:
    observations = _sample_observations(n_timesteps=n_timesteps, seed=99)

    filtered = filtered_posterior(observations, START_PROB, TRANSITION_MATRIX, MEANS, VARIANCES)

    assert filtered.shape == (n_timesteps, 2)
