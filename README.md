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

## Repository Structure

```
.
├── causal_bench/                           # Shared evaluation library
│   ├── __init__.py
│   ├── data_generators.py                  # 4 synthetic datasets (linear, nonlinear, IV, frontdoor)
│   ├── data_loader.py                      # Load Lalonde real-world benchmark
│   ├── metrics.py                          # PEHE, ATE error, bias, coverage, etc.
│   ├── wrap_causalfm.py                    # CausalFM wrapper
│   ├── wrap_causalpfn.py                   # CausalPFN wrapper
│   ├── wrap_dopfn.py                       # Do-PFN wrapper
│   └── wrap_metalearners.py                # S/T/X-learner, Debiased ML, IPW, DR wrappers
├── notebooks/
│   ├── Foundation_models_quickstart.ipynb  # CausalPFN alone, end to end (hand-maintained)
│   ├── Foundation_models_sandbox.ipynb     # All 3 foundation models side by side (hand-maintained)
│   └── Lalonde_benchmark.ipynb             # Foundation model vs. metalearners, via causal_bench
├── scripts/
│   └── build_new_notebooks.py              # Regenerates Lalonde_benchmark.ipynb
├── requirements.txt                        # Dependencies (numpy, pandas, torch, econml, causalml, etc.)
├── pyproject.toml                          # uv configuration
├── CLAUDE.md                               # Development guide
└── README.md                               # This file
```

## Datasets

### Synthetic (from `causal_bench.data_generators`)

| Name | Identification | CATE | Notes |
|---|---|---|---|
| `linear_confounded` | Backdoor | Homogeneous (constant) | Textbook setting: linear responses, observed confounders |
| `nonlinear_heterogeneous` | Backdoor | Heterogeneous `τ(x)=sin(x₀)+0.5x₁` | Matches CausalPFN's README example; tests nonlinearity |
| `iv_binary` | Instrumental Variable | Homogeneous | Hidden confounder + binary instrument; tests IV robustness |
| `frontdoor` | Front-Door Adjustment | Homogeneous | Hidden confounder + mediator; tests mediation |

### Real-World

| Name | Source | Notes |
|---|---|---|
| `lalonde_nsw_psid` | NSW vs. PSID | Standard causal ML benchmark; no ground truth CATE |

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

```python
from causal_bench import get_dataset, CausalPFNWrapper, evaluate_cate
import numpy as np

# Load synthetic dataset
ds = get_dataset("nonlinear_heterogeneous", n=2000, seed=0)
train_idx, test_idx = ds.train_test_split(0.7, seed=0)

# Create and fit model (use "mps" on Apple Silicon, "cuda" on GPU, "cpu" otherwise)
model = CausalPFNWrapper(device="cpu")
model.fit(ds.X[train_idx], ds.T[train_idx], ds.Y[train_idx])

# Predict and evaluate
tau_hat, lower, upper = model.predict(ds.X[test_idx])
results = evaluate_cate(tau_hat, ds.tau[test_idx], lower=lower, upper=upper)
print(f"PEHE: {results['pehe']:.4f}")
```

### Compare models programmatically

```python
from causal_bench import (
    get_dataset, evaluate_cate,
    CausalPFNWrapper, SLearnerWrapper, TLearnerWrapper
)

ds = get_dataset("nonlinear_heterogeneous", n=2000, seed=0)
train_idx, test_idx = ds.train_test_split(0.7, seed=0)

models = [
    CausalPFNWrapper(),
    SLearnerWrapper(),
    TLearnerWrapper(),
]

for model_cls in models:
    tau_hat, lower, upper, ate_hat, runtime = model_cls().run(
        ds.X[train_idx], ds.T[train_idx], ds.Y[train_idx],
        ds.X[test_idx]
    )
    result = evaluate_cate(tau_hat, ds.tau[test_idx], ate_hat=ate_hat,
                          ate_true=ds.ate, lower=lower, upper=upper, runtime_s=runtime)
    print(f"{model_cls.name}: PEHE={result['pehe']:.4f}, runtime={runtime:.2f}s")
```

## On Google Colab

Each notebook includes an "Open in Colab" badge. Click it to run directly on Colab (all installs happen automatically). Alternatively:

1. Open Colab: https://colab.research.google.com
2. File → Open notebook → GitHub
3. Paste this repo URL and select a notebook
4. Run all cells top-to-bottom

**Note**: Foundation models that require checkpoints (CausalFM) or external repos (Do-PFN) are installed on first use in the notebook.

## Important Notes

- **Synthetic datasets**: Ground-truth CATE available; full metrics (PEHE, bias, coverage) computed.
- **Lalonde**: Real-world data; no ground-truth CATE, but `ds.ate` is a true experimental ATE (~$1,794) from a separate randomized comparison, not a naive diff-in-means on the (confounded) data models are actually scored on — see `ds.ate_naive_observed` for that naive number. `Lalonde_benchmark.ipynb` uses `variant="nsw_psid_trimmed"` (common-support trimmed) by default rather than the raw pairing, which has too little covariate overlap for any method to get the sign of the effect right. See [`docs/LALONDE_DATASET.md`](docs/LALONDE_DATASET.md) for the full explanation.
- **Foundation models**: Fast (no per-dataset training). Do not re-train; just condition on data.
- **Metalearners**: Traditional ML methods trained from scratch on each dataset.
- **Missing dependencies**: Notebooks gracefully skip unavailable models (with warnings) rather than crashing.
- **CausalFM setup**: `CausalFM-toolkit` isn't on PyPI — the notebook clones it and adds it to `sys.path` automatically, but it also needs `einops`, `tabpfn==2.0.9`, and `tensorboard` (its own `requirements.txt` is a frozen dev snapshot with Linux/CUDA-only pins and shouldn't be installed directly). On Colab the notebook's `%pip install` cell handles this for you. Locally: `uv pip install einops "tabpfn==2.0.9" tensorboard` (see next note — don't use `uv sync --extra causalfm`).
- **Local venv (`uv`) has no `pip`**: notebook cells using `%pip install ...` only work on Colab (which ships `pip`). Locally, install missing packages with `uv pip install <pkg>` instead. Avoid `uv sync --extra <name>` for the `metalearners`/`causalfm` extras specifically — on Python 3.10 it resolves an incompatible `llvmlite` for `metalearners` and, for either extra, reconciles the whole env to just what's declared, uninstalling anything from extras you didn't also pass.

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
