"""NumPy stand-in for the sliver of FAISS that CausalPFN actually uses.

Why this exists
---------------
On macOS/arm64, ``import causalpfn`` reliably segfaults the interpreter. The
cause is not CausalPFN and not an unstable attention kernel: it is a duplicate
OpenMP runtime. ``faiss-cpu`` and ``torch`` each ship their own
``libomp.dylib``, and ``causalpfn/causal_estimator.py`` imports faiss (line 7)
*before* torch (line 9) purely because isort sorts alphabetically. Whichever
libomp loads first wins the symbol table, and the loser's parallel loops then
scribble over memory -- the crash lands in ``torch``'s ``param.copy_`` inside
``load_state_dict``, far from the actual fault.

Two workarounds exist and both are bad. ``OMP_NUM_THREADS=1`` works but costs
every core. ``KMP_DUPLICATE_LIB_OK=TRUE`` is documented by LLVM itself as
"unsafe, unsupported, undocumented ... may cause crashes or silently produce
incorrect results" -- disqualifying for anything whose numbers get reported.

The third option is to notice how little of FAISS is on the path. CausalPFN
calls it in exactly one place (``CausalEstimator._predict_cepo``), as
``IndexFlatL2(1)`` -- an exact, brute-force, *one-dimensional* L2 index over
weak-learner effect estimates. That is a sort, and NumPy already links the same
BLAS torch does. Removing faiss from the process removes the second OpenMP
runtime, so torch keeps every thread.

Semantics reproduced from ``faiss.IndexFlatL2``
-----------------------------------------------
* ``search`` returns ``(distances, indices)`` with **squared** L2 distances.
* Results are ordered by increasing distance.
* When ``k > ntotal``, missing slots are padded with index ``-1`` and distance
  ``FLT_MAX`` (not ``inf`` -- faiss uses the float32 max as its sentinel) -- CausalPFN 0.1.4 does request more neighbours than an arm holds
  when one treatment arm is small, so the sentinel path is live, not
  theoretical.
* ``d``, ``ntotal`` and ``is_trained`` are exposed as attributes.

Verified against real FAISS 1.15.0 in ``tests/test_faiss_shim.py``, which runs
in a torch-free subprocess (the only place both libraries can coexist).
"""

from __future__ import annotations

import numpy as np

__all__ = ["IndexFlatL2", "install"]

_FLT_MAX = float(np.finfo(np.float32).max)  # faiss pads absent neighbours with this


class IndexFlatL2:
    """Exact brute-force L2 index. API-compatible subset of faiss.IndexFlatL2."""

    def __init__(self, d: int):
        if not isinstance(d, (int, np.integer)) or d <= 0:
            raise ValueError(f"d must be a positive integer, got {d!r}")
        self.d = int(d)
        self.is_trained = True
        self._xb = np.empty((0, self.d), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._xb.shape[0])

    def train(self, x) -> None:  # noqa: ARG002 - flat indexes need no training
        return None

    def reset(self) -> None:
        self._xb = np.empty((0, self.d), dtype=np.float32)

    def add(self, x) -> None:
        xb = np.ascontiguousarray(x, dtype=np.float32)
        if xb.ndim != 2 or xb.shape[1] != self.d:
            raise ValueError(f"expected [n, {self.d}] array, got shape {xb.shape}")
        self._xb = xb.copy() if self.ntotal == 0 else np.vstack([self._xb, xb])

    def search(self, x, k: int):
        xq = np.ascontiguousarray(x, dtype=np.float32)
        if xq.ndim != 2 or xq.shape[1] != self.d:
            raise ValueError(f"expected [n, {self.d}] array, got shape {xq.shape}")
        k = int(k)
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        nq, nb = xq.shape[0], self.ntotal
        kk = min(k, nb)

        dist = np.full((nq, k), _FLT_MAX, dtype=np.float32)
        idx = np.full((nq, k), -1, dtype=np.int64)
        if nq == 0 or kk == 0:
            return dist, idx

        # ||q - b||^2 = ||q||^2 - 2 q.b + ||b||^2, in float64 so the expansion
        # does not lose the small differences this is being asked to rank.
        xq64, xb64 = xq.astype(np.float64), self._xb.astype(np.float64)
        d2 = (
            (xq64 * xq64).sum(1)[:, None]
            - 2.0 * (xq64 @ xb64.T)
            + (xb64 * xb64).sum(1)[None, :]
        )
        np.maximum(d2, 0.0, out=d2)  # clamp expansion round-off below zero

        if kk < nb:
            part = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
            part_d = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(part_d, axis=1, kind="stable")
            top = np.take_along_axis(part, order, axis=1)
        else:
            top = np.argsort(d2, axis=1, kind="stable")[:, :kk]

        idx[:, :kk] = top
        dist[:, :kk] = np.take_along_axis(d2, top, axis=1).astype(np.float32)
        return dist, idx


def install(force: bool = False) -> bool:
    """Register this module as ``faiss`` in ``sys.modules``.

    Must run *before* ``import causalpfn``. Returns True if the shim was
    installed. By default it is a no-op when the real faiss is already
    imported, since displacing a live extension module mid-process is worse
    than the problem it solves.
    """
    import sys
    import types

    if "faiss" in sys.modules and not force:
        return False

    mod = types.ModuleType("faiss")
    mod.IndexFlatL2 = IndexFlatL2
    mod.__doc__ = __doc__
    mod.__version__ = "0.0.0+cfms-numpy-shim"
    mod.omp_set_num_threads = lambda n: None
    mod.omp_get_max_threads = lambda: 1
    sys.modules["faiss"] = mod
    return True
