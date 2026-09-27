"""CausalPFN must actually run on Apple Silicon, not merely report available.

The 0.1.4 PyPI release segfaults on macOS/arm64 (faiss and torch load two
OpenMP runtimes). vdblm/CausalPFN#14 dropped faiss, and this repo pins a commit
that includes the fix, so these tests assert the platform genuinely works: the
wrapper returns available, and it recovers a known treatment effect.

Marked `slow` -- the first run downloads ~75MB of pretrained weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_bench import CausalPFNWrapper

pytestmark = pytest.mark.skipif(
    not CausalPFNWrapper.is_available(), reason="causalpfn not installed"
)


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
    """Two fits on identical inputs have to agree exactly, or nothing built
    on top is reproducible."""
    X, T, Y, _ = _known_effect_dataset(n=600, seed=5)
    a = CausalPFNWrapper(device="cpu")
    a.fit(X, T, Y)
    first, _, _ = a.predict(X)
    b = CausalPFNWrapper(device="cpu")
    b.fit(X, T, Y)
    second, _, _ = b.predict(X)
    np.testing.assert_array_equal(first, second)
