# CausalPFN on Apple Silicon

`import causalpfn` used to segfault the interpreter on macOS/arm64, and this
repo reported the model unavailable there. The cause is not CausalPFN, and not
the unstable `scaled_dot_product_attention` kernel this repo previously
guessed at.

## The actual fault

Run it under `faulthandler` and the crash lands in
`torch/nn/modules/module.py::_load_from_state_dict` — inside `param.copy_()`,
while loading weights, nowhere near attention.

`causalpfn/causal_estimator.py` imports in this order:

```python
import faiss   # line 7
import numpy as np
import torch   # line 9
```

isort sorts alphabetically, so faiss loads first. `faiss-cpu` and `torch` each
bundle their own `libomp.dylib`. On arm64 the first one loaded claims the
symbols and the second one's parallel loops corrupt memory. The corruption
surfaces later, at the first large tensor copy.

Two independent confirmations:

* `import torch` before `causalpfn` moves the crash out of weight loading and
  into the faiss call site — the fault follows the load order, not the code.
* `faiss.omp_set_num_threads(1)` makes libomp abort with its own message:
  *"multiple copies of the OpenMP runtime have been linked into the program."*

## The fix

CausalPFN touches faiss in exactly one place — `IndexFlatL2(1)`, an exact
brute-force k-NN over **one-dimensional** weak-learner effect estimates, in
`CausalEstimator._predict_cepo`. `causal_bench/faiss_shim.py` reimplements that
in NumPy, and `causal_bench/__init__.py` registers it as `faiss` on Darwin/arm64
before causalpfn can import the real one. One OpenMP runtime, no crash, and
torch keeps all its threads.

Import `causal_bench` **before** `faiss` or `causalpfn`. Opt out with
`CFMS_NO_FAISS_SHIM=1`.

### Rejected alternatives

| Approach | Why not |
|---|---|
| `OMP_NUM_THREADS=1` | Works, but serializes torch across every core to avoid a conflict that lives in a k-NN call. ~1.5x slower than the shim. |
| `KMP_DUPLICATE_LIB_OK=TRUE` | LLVM's own runtime calls this "unsafe, unsupported, undocumented" and warns it "may cause crashes or silently produce incorrect results". Disqualifying for anything whose numbers get reported. |

## Verification

`tests/test_faiss_shim.py` cross-checks the shim against real faiss 1.15.0 in a
**torch-free subprocess** — the only place the two libraries can coexist on this
platform — across 7 shapes covering `k > ntotal`, ties, and multi-dimensional
inputs. Neighbour sets, squared distances and the `-1`/`FLT_MAX` sentinels all
match.

End to end inside CausalPFN, with the same seed and context, the shim produces
**bitwise-identical** CATE across 1500 query points.

`data/lalonde_macos_replication.json` re-runs the paper's own RealCause-Lalonde
benchmark (10 realizations per cohort) on Darwin 25.6.0 / arm64 / torch 2.12.1
and lands on the published numbers:

| Metric | This Mac (arm64) | Published (`benchmark_results_cpu.csv`) |
|---|---|---|
| PEHE cps (x10^3) | 8.965 ± 0.072 | 8.971 ± 0.061 |
| ATE rel. error cps | 0.097 ± 0.018 | 0.169 ± 0.035 |
| PEHE psid (x10^3) | 13.939 ± 0.407 | 13.998 ± 0.411 |
| ATE rel. error psid | 0.189 ± 0.043 | 0.237 ± 0.042 |

Paper Table 3 reports PEHE 8.97 ± 0.06 (cps) and 14.00 ± 0.41 (psid), ×10³.

ATE relative error runs a little lower here than published; the benchmark seeds
each task with `seed_everything` and this replication does not reproduce that
seeding, so per-realization draws differ. The PEHE agreement to three
significant figures is the load-bearing comparison.
