"""Shared pytest fixtures and synthetic data factories."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lazystats.regimes.core import FitResult


# ---------------------------------------------------------------------------
# Synthetic data factories — no network calls, deterministic
# ---------------------------------------------------------------------------

def make_2state_1d(T: int = 300, seed: int = 0) -> np.ndarray:
    """Two-state volatility-switching Gaussian series, shape (T, 1)."""
    rng = np.random.RandomState(seed)
    states = np.zeros(T, dtype=int)
    states[T // 3 : 2 * T // 3] = 1
    x = np.where(states == 0, rng.normal(0, 0.5, T), rng.normal(0, 3.0, T))
    return x.reshape(-1, 1)


def make_3state_1d(T: int = 400, seed: int = 1) -> np.ndarray:
    """Three-state Gaussian series with clear vol separation, shape (T, 1)."""
    rng = np.random.RandomState(seed)
    states = np.zeros(T, dtype=int)
    states[100:150] = 1
    states[250:300] = 2
    scales = [0.4, 1.5, 4.0]
    x = np.array([rng.normal(0, scales[s]) for s in states])
    return x.reshape(-1, 1)


def make_multivariate_df(T: int = 200, k: int = 2, seed: int = 10) -> pd.DataFrame:
    """DataFrame with two-state volatility dynamics across k columns."""
    rng = np.random.RandomState(seed)
    states = np.zeros(T, dtype=int)
    states[T // 3 : 2 * T // 3] = 1
    cols = {f"X{i}": np.where(states == 0, rng.normal(0, 0.5, T), rng.normal(0, 2.5, T))
            for i in range(k)}
    idx = pd.date_range("2010-01-01", periods=T, freq="W-FRI")
    return pd.DataFrame(cols, index=idx)


def make_discrete_1d(T: int = 200, K: int = 4, seed: int = 3) -> np.ndarray:
    """Categorical observations, shape (T,)."""
    return np.random.RandomState(seed).randint(0, K, size=T)


def make_discrete_2d(T: int = 200, D: int = 2, seed: int = 4) -> np.ndarray:
    """2-D categorical observations, shape (T, D)."""
    return np.random.RandomState(seed).randint(0, 3, size=(T, D))


def make_fake_fitresult(
    S: int = 2,
    T: int = 100,
    k: int = 1,
    cov_type: str = "diag",
    seed: int = 0,
) -> FitResult:
    """Construct a minimal FitResult suitable for reordering / unit tests.

    State 0 is deliberately assigned HIGH variance, state 1 LOW variance so
    that after reorder(ascending=True) they swap — making tests meaningful.
    """
    rng = np.random.RandomState(seed)
    means = rng.randn(S, k)

    if cov_type == "diag":
        # state 0 → high vol, state 1 → low vol, rest → medium
        vols = np.linspace(4.0, 0.1, S)
        covars = np.array([[v] * k for v in vols])
    else:
        vols = np.linspace(4.0, 0.1, S)
        covars = np.zeros((S, k, k))
        for s in range(S):
            covars[s] = np.eye(k) * vols[s]

    trans = np.full((S, S), 1.0 / S)
    start = np.full(S, 1.0 / S)
    gamma = rng.dirichlet(np.ones(S), size=T)
    vpath = rng.randint(0, S, size=T) if S > 1 else np.zeros(T, dtype=int)

    return FitResult(
        S=S,
        cov_type=cov_type,
        shared_mean=False,
        seed=seed,
        converged=True,
        n_iter=10,
        loglik=-500.0,
        bic=1000.0,
        startprob_=start,
        transmat_=trans,
        means_=means,
        covars_=covars,
        gamma_=gamma,
        viterbi_path_=vpath,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def Y2_1d():
    """Pre-computed 2-state 1-D series (session scope — computed once)."""
    return make_2state_1d(T=300)


@pytest.fixture(scope="session")
def Y3_1d():
    return make_3state_1d(T=400)


@pytest.fixture(scope="session")
def multivar_df():
    return make_multivariate_df(T=200, k=2)


@pytest.fixture(scope="session")
def discrete_1d():
    return make_discrete_1d(T=200, K=3)


@pytest.fixture(scope="session")
def discrete_2d():
    return make_discrete_2d(T=200, D=2)
