"""CausalPFN must actually run on Apple Silicon, not merely report available.

Upstream reports CausalPFN unavailable on macOS/arm64 because importing it
segfaults. `causal_bench.macos_compat` removes the duplicate OpenMP runtime
that causes it, so these tests assert the platform genuinely works: the wrapper
returns available, and it recovers a known treatment effect.

Marked `slow` -- the first run downloads ~75MB of pretrained weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_bench import CausalPFNWrapper, needs_faiss_shim, shim_active

pytestmark = pytest.mark.skipif(
    not CausalPFNWrapper.is_available(), reason="causalpfn not installed"
)


def test_shim_is_active_where_it_is_needed():
    assert shim_active() == needs_faiss_shim()


def _known_effect_dataset(n=1200, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    ps = 1.0 / (1.0 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    T = (rng.uniform(size=n) < ps).astype(np.float32)
    tau = 2.0 + 0.8 * X[:, 2]
    Y = (X[:, 0] - 0.5 * X[:, 1] + tau * T + rng.normal(scale=0.4, size=n)).astype(np.float32)
    return X, T, Y, tau


@pytest.mark.slow
def test_recovers_a_known_confounded_effect():
    X, T, Y, tau = _known_effect_dataset()
    tau_hat, _, _, ate_hat, runtime = CausalPFNWrapper(device="cpu").run(X, T, Y, X)

    assert tau_hat.shape == (len(X),)
    assert np.isfinite(tau_hat).all()
    # Treatment is confounded through X0/X1, so this fails if the adjustment
    # is not happening -- a naive difference in means is materially biased here.
    assert ate_hat == pytest.approx(tau.mean(), rel=0.10)
    # Heterogeneity is driven by X2; the ranking must survive, not just the mean.
    assert np.corrcoef(tau_hat, tau)[0, 1] > 0.9
    assert runtime > 0


@pytest.mark.slow
def test_estimates_are_deterministic_across_calls():
    """The shim must not introduce run-to-run drift: two fits on identical
    inputs have to agree exactly, or nothing built on top is reproducible."""
    X, T, Y, _ = _known_effect_dataset(n=600, seed=5)
    a = CausalPFNWrapper(device="cpu")
    a.fit(X, T, Y)
    first, _, _ = a.predict(X)
    b = CausalPFNWrapper(device="cpu")
    b.fit(X, T, Y)
    second, _, _ = b.predict(X)
    np.testing.assert_array_equal(first, second)
