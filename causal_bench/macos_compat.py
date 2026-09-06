"""Make CausalPFN runnable on Apple Silicon by removing the duplicate OpenMP runtime.

``import causalpfn`` segfaults on macOS/arm64. The fault is an environment
collision, not a model or kernel problem: ``faiss-cpu`` and ``torch`` each
bundle their own ``libomp.dylib``, and ``causalpfn/causal_estimator.py``
happens to import faiss (line 7) before torch (line 9). The first libomp to
load claims the symbols and the second one's parallel loops then corrupt
memory. The crash surfaces in ``torch``'s ``param.copy_`` during
``load_state_dict`` -- nowhere near the actual fault, which is why it reads
like a broken attention kernel.

The fix is to stop loading two OpenMP runtimes. CausalPFN touches faiss in
exactly one place -- an ``IndexFlatL2(1)`` exact k-NN over one-dimensional
weak-learner effect estimates -- so :mod:`causal_bench.faiss_shim` reimplements
that in NumPy and this module registers it as ``faiss`` before CausalPFN can
import the real one.

Verified on Darwin 25.6.0 / arm64, Python 3.12, torch 2.12.1, causalpfn 0.1.4,
against real faiss 1.15.0: CATE estimates are **bitwise identical** over 1500
query points, and the run is ~1.5x faster than the ``OMP_NUM_THREADS=1``
workaround because torch keeps all its threads.

Rejected alternatives:

``OMP_NUM_THREADS=1``
    Works, but serializes torch on every core for a conflict that lives in a
    k-NN call.

``KMP_DUPLICATE_LIB_OK=TRUE``
    LLVM's own runtime describes this as "unsafe, unsupported, undocumented"
    and warns it "may cause crashes or silently produce incorrect results".
    Not acceptable for numbers anyone reports.

Set ``CFMS_NO_FAISS_SHIM=1`` to opt out and get upstream behaviour (i.e. the
segfault) back.
"""

from __future__ import annotations

import os
import platform
import sys

__all__ = ["needs_faiss_shim", "ensure_causalpfn_importable", "shim_active"]


def needs_faiss_shim() -> bool:
    """True on the platform where faiss and torch cannot share a process."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def shim_active() -> bool:
    """True if ``sys.modules['faiss']`` is our NumPy stand-in."""
    mod = sys.modules.get("faiss")
    return getattr(mod, "__version__", "").endswith("cfms-numpy-shim")


def ensure_causalpfn_importable() -> bool:
    """Install the faiss shim if this platform needs it. Idempotent.

    Returns True if CausalPFN can be imported safely after this call. On
    platforms that never had the problem this returns True without touching
    ``sys.modules``.
    """
    if not needs_faiss_shim():
        return True
    if os.environ.get("CFMS_NO_FAISS_SHIM") == "1":
        return False
    if shim_active():
        return True

    from .faiss_shim import install

    if install():
        return True
    # Real faiss already imported: displacing a live extension module
    # mid-process is worse than refusing. The caller must import
    # causal_bench before faiss/causalpfn.
    return False
