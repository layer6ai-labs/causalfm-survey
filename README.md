<div align="center">

<a href="https://layer6.ai"><img src="assets/layer6.png" alt="Layer 6 AI" width="220"></a>

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/github/license/layer6ai-labs/cfms)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.03003-b31b1b.svg)](https://arxiv.org/abs/2609.03003)
[![Hugging Face Papers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Papers-yellow)](https://huggingface.co/papers/2609.03003)

# Causal Foundation Models

</div>

This repository is the easiest way to start using **Causal Foundational Models** (prior-fitted networks that use in-context learning to estimate causal quantities on new datasets). In this repo, we give a [quickstart](notebooks/Foundation_models_quickstart.ipynb) to get one Causal Foundational Model (CFM) running, a [sandbox](notebooks/Foundation_models_sandbox.ipynb) comparing known CFMs side by side, and a benchmark that runs known CFMs against meta learners on real-world semi-synthetic data ([benchmark script](scripts/run_benchmark.py)) visualized in the [results notebook](notebooks/Lalonde_benchmark_results.ipynb).

## Models Included

### Foundation Models (In-Context Learning)
| Model | Paper | Code |
|---|---|---|
| **CausalPFN** | Balazadeh, Kamkari et al., *CausalPFN: Amortized Causal Effect Estimation via In-Context Learning*, NeurIPS 2025 | [vdblm/CausalPFN](https://github.com/vdblm/CausalPFN) |
| **Do-PFN** | Robertson, Reuter et al., *Do-PFN: In-Context Learning for Causal Effect Estimation*, NeurIPS 2025 | [jr2021/Do-PFN](https://github.com/jr2021/Do-PFN) |
| **CausalFM** | Ma, Frauen, et al., *Foundation Models for Causal Inference via Prior-Data Fitted Networks*, ICLR 2026 | [yccm/CausalFM-toolkit](https://github.com/yccm/CausalFM-toolkit) |

### Metalearners (from EconML)
| Method | Description |
|---|---|
| **S-learner** | Single-model learner: trains one model on covariates + treatment |
| **T-learner** | Two-model learner: separate models for each binary treatment group |
| **X-learner** | Cross-fit learner: asymptotically efficient variant |
| **Debiased ML** | Neyman-orthogonal approach, robust to nuisance parameter estimation |
| **IPW** | Inverse Probability Weighting: based on propensity scores |
| **DR** | Doubly Robust: combines outcome and propensity modeling |

## Quick Start

Every notebook runs on **Google Colab with zero local setup** (click its
"Open in Colab" badge, see [On Google Colab](#on-google-colab)), or locally
with the steps below (see [Running notebooks locally](#running-notebooks-locally)
for per-model dependency installs).

### 1. Install

Ensure you have `python >=3.10,<3.13`.

```bash
# With uv (recommended): everything -- core, metalearners, CausalFM deps
uv sync --all-extras

# Or with pip:
pip install -r requirements.txt
```

### 2. Causal Foundation Model Quickstart

```bash
jupyter notebook notebooks/Foundation_models_quickstart.ipynb
```

The fastest path to working with one causal foundation model. Simply install, load your data, split into context and query, then predict. No training required for each new inference dataset.

### 3. Foundation Models Sandbox

```bash
jupyter notebook notebooks/Foundation_models_sandbox.ipynb
```

A playground/sandbox notebook. Runs three causal foundation models (CausalPFN, Do-PFN, CausalFM) side-by-side on one dataset, each called through its own native API. Includes basic evaluation and plotting code so that you can quickly experiment and learn about each CFM's properties.

### 4. RealCause Lalonde HPO Benchmark

```bash
python scripts/run_benchmark.py --smoke   # fast end-to-end check (no HPO, 1 realization/cohort)
python scripts/run_benchmark.py           # full run: 9 models x 2 cohorts x 10 realizations, HPO-tuned
```

The Lalonde benchmark: three foundation models (Causal PFN, Do-PFN, and CausalFM) against six metalearners (HPO-tuned), benchmarked on the RealCause semi-synthetic Lalonde dataset (see [`docs/LALONDE_DATASET.md`](docs/LALONDE_DATASET.md) for what that dataset is and why we use it). A full run is expensive (~105 CPU-hours of FLAML search); results are checkpointed to CSV, and progress is logged to both stdout and `logs/run_benchmark_<timestamp>.log`. Run `python scripts/run_benchmark.py --help` for all options (`--gpu`, `--n-realizations`, `--hpo-time-budget`, `--results-dir`).

### 5. Lalonde Benchmark Results

```bash
jupyter notebook notebooks/Lalonde_benchmark_results.ipynb
```

Loads the full production run's output (already checked into `data/benchmark_results_cpu.csv`, all 9 models, and `data/benchmark_results_gpu.csv`, the 3 foundation models, re-run on GPU), averages over the 10 realizations per cohort, and reproduces a summary table plus a rank-vs-runtime figure, CATE accuracy vs. compute cost, at a glance.

### Running notebooks locally

Every notebook's Colab install cells (`%pip install ...`) silently no-op in this repo's local `uv`-managed venv (it has no `pip` module). `uv sync --all-extras` installs everything below except Do-PFN's `networkx` (already pulled in by torch) and the two `git clone`s; to install a single group by hand instead:

- **CausalPFN**: already a core dependency (`uv sync`), pinned to an upstream commit because the 0.1.4 PyPI release predates its Apple Silicon fix. Standalone: `uv pip install "causalpfn @ git+https://github.com/vdblm/CausalPFN@896a2617adbf9bec1cb2ea0926ca4a28fc48990b"`
- **Do-PFN**: `uv pip install networkx tqdm einops "torch<2.10"`. Do-PFN is not on PyPI, notebooks `git clone` it automatically; `torch<2.10` is required (Do-PFN breaks on newer)
- **CausalFM**: `uv pip install einops "tabpfn==2.0.9" tensorboard`. CausalFM is also not on PyPI, cloned automatically
- **Metalearners**: `uv pip install econml causalml "FLAML[automl]==2.3.5"`.

Apple Silicon Macs: all three foundation models run on CPU, just slower than on a GPU. (CausalPFN's 0.1.4 PyPI release segfaults on Apple Silicon; the pinned commit fixes it.)

`scripts/run_benchmark.py` needs all four dependency groups above installed at once (it runs all 9 models), plus `git clone`s of Do-PFN and CausalFM-toolkit under `notebooks/` (same layout the sandbox notebook uses).

Hit something not covered here (a stale-import error after re-running a cell, a version-pin conflict, etc.)? See [`CLAUDE.md`](CLAUDE.md).

### On Google Colab

Each notebook includes an "Open in Colab" badge. Click it to run directly on Colab (all installs happen automatically). Alternatively:

1. Open Colab: https://colab.research.google.com
2. File → Open notebook → GitHub
3. Paste this repo URL and select a notebook
4. Run all cells top-to-bottom

**Note**: Foundation models that require checkpoints (CausalFM) or external repos (Do-PFN) are installed on first use in the notebook.

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
│   └── Lalonde_benchmark_results.ipynb     # Loads data/*.csv, builds the results table + figure
├── scripts/
│   └── run_benchmark.py                    # The Lalonde benchmark: 9 models, HPO-tuned, RealCause data
├── data/
│   ├── benchmark_results_cpu.csv           # Full run output: 9 models x 2 cohorts x 10 realizations
│   └── benchmark_results_gpu.csv           # Same, foundation models only, re-run on GPU
├── docs/
│   ├── LALONDE_DATASET.md                  # Which Lalonde version this repo benchmarks on, and why
│   └── WRAPPERS_GUIDE.md                   # causal_bench wrapper internals: HPO, standardization, gotchas
├── requirements.txt                        # Dependencies (numpy, pandas, torch, econml, causalml, etc.)
├── pyproject.toml                          # uv configuration
├── CLAUDE.md                               # Development guide
└── README.md                               # This file
```

## Datasets

### Real-World

| Name | Source | Notes |
|---|---|---|
| Lalonde (`load_lalonde`) | Real NSW vs. PSID data | No ground-truth CATE, but a true experimental ATE is available (`ds.ate`). ([Dehejia et al., 1999](https://www.jstor.org/stable/2669919)) |
| RealCause Lalonde (`load_lalonde_realcause`) | PSID + CPS, 10 realizations each | Semi-synthetic realizations over real covariates — gives individual-level CATE ground truth. ([Neal et al., 2020](https://arxiv.org/abs/2011.15007)) |

## Metrics

All models evaluated on:
- **PEHE**: `√E[(τ̂ - τ)²]` — precision in estimating heterogeneous effects
- **ATE error**: `|ATE_hat - ATE_true|` — absolute error on average effect
- **ATE relative error**: `|ATE_hat - ATE_true| / |ATE_true|`
- **Bias**: `mean(τ̂ - τ)` — systematic over/under-estimation
- **Coverage @95%**: Fraction of true τ inside model's 95% confidence interval (when available)
- **Runtime**: Seconds (fit + predict on test set)

<!-- ## Usage Examples

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
``` -->

## Citation

If you find this repository useful, please cite the paper as follows

```bibtex
@article{stith2026causalfoundationmodels,
      title={Causal Foundation Models}, 
      author={Christopher Stith and Hossein Rahmani and Jesse C. Cresswell},
      year={2026},
      journal={arXiv:2609.03003}
}
```

## License

This code is licensed under the MIT License, copyright by Layer 6 AI.
