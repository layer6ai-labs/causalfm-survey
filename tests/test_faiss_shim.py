"""The NumPy faiss shim must match real faiss exactly, or CausalPFN's
neighbour strata change and every number downstream moves with them.

Real faiss and torch cannot coexist on macOS/arm64 -- that collision is the
whole reason the shim exists -- so the comparison against real faiss runs in a
torch-free subprocess. The shim's own behaviour is tested in-process.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from causal_bench.faiss_shim import IndexFlatL2, install

REPO_ROOT = Path(__file__).resolve().parents[1]
FLT_MAX = float(np.finfo(np.float32).max)


def test_basic_1d_search():
    idx = IndexFlatL2(1)
    idx.add(np.array([[0.0], [1.0], [2.0]], dtype=np.float32))
    dist, ind = idx.search(np.array([[0.4]], dtype=np.float32), 2)
    assert ind.tolist() == [[0, 1]]
    assert np.allclose(dist, [[0.16, 0.36]], atol=1e-5)
    assert idx.ntotal == 3 and idx.d == 1 and idx.is_trained


def test_k_greater_than_ntotal_uses_faiss_sentinels():
    """CausalPFN 0.1.4 asks for more neighbours than a small arm holds, so
    the -1 / FLT_MAX padding is on the live path, not a corner case."""
    idx = IndexFlatL2(1)
    idx.add(np.array([[0.0], [1.0]], dtype=np.float32))
    dist, ind = idx.search(np.array([[0.5]], dtype=np.float32), 5)
    assert ind.tolist() == [[0, 1, -1, -1, -1]]
    assert dist[0, 2] == pytest.approx(FLT_MAX)


def test_add_is_cumulative():
    idx = IndexFlatL2(2)
    idx.add(np.zeros((3, 2), dtype=np.float32))
    idx.add(np.ones((4, 2), dtype=np.float32))
    assert idx.ntotal == 7


def test_reset_empties_the_index():
    idx = IndexFlatL2(1)
    idx.add(np.zeros((5, 1), dtype=np.float32))
    idx.reset()
    assert idx.ntotal == 0
    dist, ind = idx.search(np.zeros((2, 1), dtype=np.float32), 3)
    assert (ind == -1).all()


def test_results_are_distance_ordered():
    rng = np.random.default_rng(0)
    xb = rng.normal(size=(200, 1)).astype(np.float32)
    idx = IndexFlatL2(1)
    idx.add(xb)
    dist, _ = idx.search(rng.normal(size=(20, 1)).astype(np.float32), 10)
    assert (np.diff(dist, axis=1) >= -1e-6).all()


def test_dimension_mismatch_rejected():
    idx = IndexFlatL2(3)
    with pytest.raises(ValueError):
        idx.add(np.zeros((4, 2), dtype=np.float32))
    idx.add(np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        idx.search(np.zeros((2, 2), dtype=np.float32), 1)


def test_install_refuses_to_displace_real_faiss():
    import sys as _sys
    import types

    sentinel = types.ModuleType("faiss")
    saved = _sys.modules.get("faiss")
    _sys.modules["faiss"] = sentinel
    try:
        assert install() is False
        assert _sys.modules["faiss"] is sentinel
        assert install(force=True) is True
        assert _sys.modules["faiss"] is not sentinel
    finally:
        if saved is None:
            _sys.modules.pop("faiss", None)
        else:
            _sys.modules["faiss"] = saved


_CROSSCHECK = textwrap.dedent(
    """
    import sys, json
    import numpy as np
    sys.path.insert(0, {root!r})
    assert "torch" not in sys.modules, "this check must stay torch-free"
    import faiss
    from causal_bench.faiss_shim import IndexFlatL2

    rng = np.random.default_rng(7)
    bad = []
    cases = [(1, 50, 20, 5), (1, 1000, 200, 64), (1, 7, 5, 32), (1, 1, 3, 4),
             (1, 300, 300, 300), (3, 500, 100, 16), (5, 64, 64, 64)]
    for d, nb, nq, k in cases:
        xb = rng.normal(size=(nb, d)).astype(np.float32)
        xq = rng.normal(size=(nq, d)).astype(np.float32)
        fi = faiss.IndexFlatL2(d); fi.add(np.ascontiguousarray(xb))
        fd, fidx = fi.search(np.ascontiguousarray(xq), k)
        si = IndexFlatL2(d); si.add(xb)
        sd, sidx = si.search(xq, k)
        if not np.array_equal(fidx == -1, sidx == -1):
            bad.append([d, nb, nq, k, "sentinel"]); continue
        m = fidx != -1
        if not np.allclose(fd[m], sd[m], rtol=1e-4, atol=1e-5):
            bad.append([d, nb, nq, k, "distance"]); continue
        # ties may order differently between implementations; the neighbour
        # *set* is what CausalPFN consumes, so that is what must match.
        for i in range(nq):
            if sorted(fidx[i][fidx[i] != -1].tolist()) != sorted(sidx[i][sidx[i] != -1].tolist()):
                bad.append([d, nb, nq, k, "index-set"]); break
        if fi.ntotal != si.ntotal:
            bad.append([d, nb, nq, k, "ntotal"])
    print(json.dumps({{"n_cases": len(cases), "bad": bad}}))
    """
).format(root=str(REPO_ROOT))


def test_matches_real_faiss_in_a_torch_free_subprocess():
    proc = subprocess.run(
        [sys.executable, "-c", _CROSSCHECK], capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        if "No module named 'faiss'" in proc.stderr:
            pytest.skip("real faiss not installed")
        pytest.fail(f"cross-check crashed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    import json

    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report["bad"] == [], f"shim diverges from real faiss: {report['bad']}"
    assert report["n_cases"] >= 7
