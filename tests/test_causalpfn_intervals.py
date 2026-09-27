"""CausalPFN can report credible intervals.

`causalpfn.CATEEstimator.estimate_ate_CI` raised KeyError on every call in the
0.1.4 PyPI release; vdblm/CausalPFN#14 fixed it, and this repo pins a commit
that includes the fix. `CausalPFNWrapper.estimate_att_ci` relies on it.

Without intervals `coverage_95` cannot be computed for this model, which is why
that column is empty for CausalPFN in `data/benchmark_results_*.csv`.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_bench import CausalPFNWrapper
from causal_bench.metrics import coverage_95

pytestmark = pytest.mark.skipif(
    not CausalPFNWrapper.is_available(), reason="causalpfn not installed"
)

N_SAMPLES = 512  # keep the posterior draw cheap; these are behavioural tests


def _data(true_effect, n_c=900, n_t=150, seed=1):
    rng = np.random.default_rng(seed)
    X = np.vstack(
        [rng.normal(size=(n_c, 5)), rng.normal(size=(n_t, 5)) + 0.3]
    ).astype(np.float32)
    T = np.concatenate([np.zeros(n_c), np.ones(n_t)]).astype(np.float32)
    Y = (X[:, 0] * 0.5 + true_effect * T + rng.normal(scale=1.0, size=n_c + n_t)).astype(
        np.float32
    )
    return X, T, Y


def test_upstream_estimate_ate_ci_works():
    """Fails with KeyError on the 0.1.4 PyPI release: a sign the installed
    causalpfn predates the pinned commit (re-run `uv sync`)."""
    from causalpfn import CATEEstimator

    X, T, Y = _data(0.0, n_c=300, n_t=60)
    est = CATEEstimator(device="cpu", verbose=False, num_neighbours=60)
    est.fit(X, T, Y)
    out = est.estimate_ate_CI(X[T == 1], alpha=0.05, n_samples=64)
    assert out["ate"] == est.estimate_ate(X[T == 1])
    assert out["lower_bound"][0] <= out["upper_bound"][0]


@pytest.mark.slow
def test_att_interval_brackets_the_point_estimate():
    X, T, Y = _data(0.8)
    w = CausalPFNWrapper(device="cpu", interval_n_samples=N_SAMPLES)
    w.fit(X, T, Y)
    point, lo, hi = w.estimate_att_ci(X[T == 1], alpha=0.05)
    assert lo < point < hi
    assert np.isfinite([point, lo, hi]).all()


@pytest.mark.slow
def test_a_true_null_produces_an_interval_containing_zero():
    """The point of intervals here is to bound a null, so a genuine null has
    to come back as an interval straddling zero rather than a bare 'no effect'."""
    X, T, Y = _data(0.0)
    w = CausalPFNWrapper(device="cpu", interval_n_samples=N_SAMPLES)
    w.fit(X, T, Y)
    _, lo, hi = w.estimate_att_ci(X[T == 1], alpha=0.05)
    assert lo <= 0.0 <= hi


@pytest.mark.slow
def test_predict_returns_intervals_only_when_alpha_is_set():
    X, T, Y = _data(0.8)
    plain = CausalPFNWrapper(device="cpu")
    plain.fit(X, T, Y)
    tau, lo, hi = plain.predict(X[T == 1])
    assert lo is None and hi is None

    with_ci = CausalPFNWrapper(device="cpu", alpha=0.05, interval_n_samples=N_SAMPLES)
    with_ci.fit(X, T, Y)
    tau, lo, hi = with_ci.predict(X[T == 1])
    assert lo is not None and hi is not None
    assert lo.shape == tau.shape == hi.shape
    assert (lo <= hi).all()
    # coverage_95 is computable now, which is the point of the change
    cov = coverage_95(np.full_like(tau, 0.8), lo, hi)
    assert cov is not None and 0.0 <= cov <= 1.0


def test_alpha_is_validated():
    with pytest.raises(ValueError):
        CausalPFNWrapper(alpha=0.0)
    with pytest.raises(ValueError):
        CausalPFNWrapper(alpha=1.0)
    with pytest.raises(ValueError):
        CausalPFNWrapper(interval_n_samples=0)
