# Causal Foundation Models — Benchmark Suite

Companion code for the survey on **causal foundation models** (prior-fitted networks that use in-context learning to estimate causal quantities on new datasets). In this repo, we compare three recent causal foundation models (CFMs) against six traditional metalearners on synthetic and real-world causal inference benchmarks.

## Models Included

### Foundation Models (In-Context Learning)
| Model | Paper | Code |
|---|---|---|
| **CausalPFN** | Balazadeh, Kamkari et al., *CausalPFN: Amortized Causal Effect Estimation via In-Context Learning*, NeurIPS 2025 | [vdblm/CausalPFN](https://github.com/vdblm/CausalPFN) |
| **Do-PFN** | Robertson, Reuter et al., *Do-PFN: In-Context Learning for Causal Effect Estimation*, NeurIPS 2025 | [jr2021/Do-PFN](https://github.com/jr2021/Do-PFN) |
| **CausalFM** | Ma, Frauen, et al., *Foundation Models for Causal Inference via Prior-Data Fitted Networks*, ICLR 2026 | [yccm/CausalFM-toolkit](https://github.com/yccm/CausalFM-toolkit) |

### Metalearners (from econml)
| Method | Description |
|---|---|
| **S-learner** | Single-model learner: trains one model on covariates + treatment |
| **T-learner** | Two-model learner: separate models for each binary treatment group |
| **X-learner** | Cross-fit learner: asymptotically efficient variant |
| **Debiased ML** | Neyman-orthogonal approach, robust to nuisance parameter estimation |
| **IPW** | Inverse Probability Weighting: based on propensity scores |
| **DR** | Doubly Robust: combines outcome and propensity modeling |

## Quick Start

### 1. Install

Ensure you have `python >=3.10,<3.13`.

```bash
# With uv (recommended):
uv sync

# Or with pip:
pip install -r requirements.txt
```

### 2. Causal Foundation Model Quickstart

```bash
jupyter notebook notebooks/Foundation_models_quickstart.ipynb
```

The fastest path to working with one causal foundation model. Simply install, load your data, split into context and query, then predict — done. No training required for each new inference dataset.

### 3. Foundation Models Sandbox

```bash
jupyter notebook notebooks/Foundation_models_sandbox.ipynb
```

A playground/sandbox notebook. Runs three causal foundation models (CausalPFN, Do-PFN, CausalFM) side-by-side on one dataset, each called through its own native API. Includes basic evaluation and plotting code so that you can quickly experiment and learn about each CFM's properties.

### 4. Lalonde Benchmark

```bash
jupyter notebook notebooks/Lalonde_benchmark.ipynb
```

Compare three causal foundation models against six causal metalearners on the real-world Lalonde benchmark, using this repo's `causal_bench` wrappers. Models are compared head-to-head on a lightweight but practical task.

### 5. RealCause Benchmark

```bash
jupyter notebook notebooks/RealCause_benchmark.ipynb
```

Same 9 models as the Lalonde benchmark above, but run on RealCause semi-synthetic Lalonde realizations instead of the real NBER data — this gives individual-level CATE ground truth (impossible on real data) and matches CausalPFN's own paper methodology. See [`docs/LALONDE_DATASET.md`](docs/LALONDE_DATASET.md) for why the two Lalonde benchmarks report different numbers.

### Running notebooks locally

Every notebook's Colab install cells (`%pip install ...`) silently no-op in this repo's local `uv`-managed venv (it has no `pip` module) — install what you need yourself first, with `uv pip install <pkg>` :

- **CausalPFN**: `uv pip install causalpfn`
- **Do-PFN**: `uv pip install networkx tqdm einops "torch<2.10"` — not on PyPI, notebooks `git clone` it automatically; `torch<2.10` is required (Do-PFN breaks on newer)
- **CausalFM**: `uv pip install einops "tabpfn==2.0.9" tensorboard` — also not on PyPI, cloned automatically
- **Metalearners**: `uv pip install econml causalml` — not `uv sync --extra metalearners`, which is broken on Python 3.10

Apple Silicon Macs: CausalPFN segfaults on both CPU and MPS and is skipped automatically; Do-PFN and CausalFM both run fine on CPU, just slower than on a GPU.

Hit something not covered here (a stale-import error after re-running a cell, a version-pin conflict, etc.)? See [`CLAUDE.md`](CLAUDE.md).
## Repository Structure

```
.
├── causal_bench/                           # Shared evaluation library
│   ├── __init__.py
│   ├── data_generators.py                  # 4 synthetic datasets (linear, nonlinear, IV, frontdoor)
│   ├── data_loader.py                      # Load Lalonde: real NBER data + RealCause semi-synthetic
│   ├── metrics.py                          # PEHE, ATE error, bias, coverage, etc.
│   ├── wrap_causalfm.py                    # CausalFM wrapper
│   ├── wrap_causalpfn.py                   # CausalPFN wrapper
│   ├── wrap_dopfn.py                       # Do-PFN wrapper
│   ├── wrap_foundation.py                  # Shared logic used by the 3 foundation-model wrappers
│   └── wrap_metalearners.py                # S/T/X-learner, Debiased ML, IPW, DR wrappers
├── notebooks/
│   ├── Foundation_models_quickstart.ipynb  # CausalPFN alone, end to end (hand-maintained)
│   ├── Foundation_models_sandbox.ipynb     # All 3 foundation models side by side (hand-maintained)
│   ├── Lalonde_benchmark.ipynb             # Foundation models vs. metalearners, real NBER data
│   └── RealCause_benchmark.ipynb           # Same 9 models, RealCause semi-synthetic realizations
├── scripts/
│   └── build_new_notebooks.py              # Regenerates Lalonde_benchmark.ipynb + RealCause_benchmark.ipynb
├── docs/
│   └── LALONDE_DATASET.md                  # Why the two Lalonde benchmarks report different numbers
├── requirements.txt                        # Dependencies (numpy, pandas, torch, econml, causalml, etc.)
├── pyproject.toml                          # uv configuration
├── CLAUDE.md                               # Development guide
└── README.md                               # This file
```

## Datasets

### Real-World

| Name | Source | Notes |
|---|---|---|
| Lalonde (`load_lalonde`) | Real NSW vs. PSID data | No ground-truth CATE, but a true experimental ATE is available (`ds.ate`) |
| RealCause Lalonde (`load_lalonde_realcause`) | PSID + CPS, 10 realizations each | Semi-synthetic realizations over real covariates — gives individual-level CATE ground truth; matches CausalPFN's paper methodology |

## Metrics

All models evaluated on:
- **PEHE**: `√E[(τ̂ - τ)²]` — precision in estimating heterogeneous effects
- **ATE error**: `|ATE_hat - ATE_true|` — absolute error on average effect
- **ATE relative error**: `|ATE_hat - ATE_true| / |ATE_true|`
- **Bias**: `mean(τ̂ - τ)` — systematic over/under-estimation
- **Coverage @95%**: Fraction of true τ inside model's 95% confidence interval (when available)
- **Runtime**: Seconds (fit + predict on test set)

## Usage Examples

### Run one model on one dataset (Python)

See [`Foundation_models_quickstart.ipynb`](notebooks/Foundation_models_quickstart.ipynb) for the full runnable notebook (data generation, install, reference output). The core call, straight from that notebook:

```python
from causalpfn import CATEEstimator, ATEEstimator

cate_estimator = CATEEstimator(device=device, verbose=False)
cate_estimator.fit(X_ctx, T_ctx, Y_ctx)
cate_hat = np.asarray(cate_estimator.estimate_cate(X_qry)).reshape(-1)

ate_estimator = ATEEstimator(device=device, verbose=False)
ate_estimator.fit(X_ctx, T_ctx, Y_ctx)
ate_hat = float(np.asarray(ate_estimator.estimate_ate()).reshape(-1)[0])
```

### Compare models programmatically

See [`Foundation_models_sandbox.ipynb`](notebooks/Foundation_models_sandbox.ipynb) which runs CausalPFN, Do-PFN, and CausalFM side by side on the same dataset, then plots predicted-vs-true CATE and a PEHE bar chart for them.

> **Be aware**: all three models are pretrained on standardized (mean 0, unit variance) synthetic data, so standardizing your own inputs before comparing them can noticeably change your results. The sandbox's simulated dataset is already roughly standardized by construction — if you swap in your own data here, scale it first. (This repo's `causal_bench` wrappers, used in `Lalonde_benchmark.ipynb`/`RealCause_benchmark.ipynb`, do this standardization for you automatically; calling each model's native API directly like this notebook does does not.)

## On Google Colab

Each notebook includes an "Open in Colab" badge. Click it to run directly on Colab (all installs happen automatically). Alternatively:

1. Open Colab: https://colab.research.google.com
2. File → Open notebook → GitHub
3. Paste this repo URL and select a notebook
4. Run all cells top-to-bottom

**Note**: Foundation models that require checkpoints (CausalFM) or external repos (Do-PFN) are installed on first use in the notebook.

## Citation

If you find this code and survey helpful, please cite the paper as follows

```bibtex
@article{stith2026cfm,
  title={Causal Foundation Models},
  author={Stith, Christopher and Rahmani, Hossein and Cresswell, Jesse C},
  journal={arXiv:2609.XXXXX},
  year={2026}
}
```

## License

This code is licensed under the MIT License, copyright by Layer 6 AI.
