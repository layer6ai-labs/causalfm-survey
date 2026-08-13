# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Benchmark companion code for a survey on **causal foundation models** — zero-shot, amortised CATE estimators trained via PFN-style meta-learning. Compare three foundation models against six traditional metalearners on synthetic and real-world data.

### Foundation Models

| Model | Install method |
|---|---|
| **CausalPFN** | `pip install causalpfn` (weights downloaded from HF Hub on first use) |
| **Do-PFN** | Clone `jr2021/Do-PFN`, add to `sys.path` — not its `requirements.txt` as-is, see caveats below |
| **CausalFM** | Clone `yccm/CausalFM-toolkit`, install requirements, provide a checkpoint path |

### Traditional Metalearners (from econml)

| Method | Type | Key feature |
|---|---|---|
| **S-learner** | Single-model | Trains one model on X⊕T |
| **T-learner** | Two-model | Separate models for T=0, T=1 |
| **X-learner** | Cross-fitting | Asymptotically efficient variant |
| **Debiased ML** | Neyman-orthogonal | Robust to nuisance parameter estimation |
| **IPW** | Inverse Probability Weighting | Based on propensity scores |
| **DR** | Doubly Robust | Combines outcome + propensity modeling |

## Commands

### Local setup (with uv)

```bash
uv sync                           # install all deps from pyproject.toml
jupyter notebook notebooks/Foundation_models_quickstart.ipynb   # start exploring
```

Or with pip:

```bash
pip install -r requirements.txt   # core deps
jupyter notebook notebooks/Foundation_models_quickstart.ipynb
```

There's no dedicated setup/data notebook — synthetic dataset generation is inline via `causal_bench.data_generators` (see the smoke-test below), used directly wherever a notebook needs a dataset.

### Running notebooks

Three notebooks, no numeric ordering — pick based on what you need:

**1. Foundation Models Quickstart** (fastest path to one working model):
```bash
jupyter notebook notebooks/Foundation_models_quickstart.ipynb
```
- Covers **CausalPFN** alone, end to end: install, fit, predict. Standalone — calls CausalPFN's native API directly, not this repo's wrapper classes, so any cell can be copied into another project as-is.
- Deliberately narrow (one model, one dataset) — see the sandbox notebook below for a three-model comparison.

**2. Lalonde Benchmark** (compare all foundation models vs. all metalearners):
```bash
jupyter notebook notebooks/Lalonde_benchmark.ipynb
```
- Runs all 3 foundation models (CausalPFN, Do-PFN, CausalFM) + all 6 metalearners on Lalonde, no selection needed
- Each model runs independently — an unavailable or erroring model is skipped/reported without blocking the rest
- Uses `causal_bench` wrappers (unlike the other two notebooks, which call each library's native API directly)
- Sections 1–7: real NBER Lalonde data (`load_lalonde`) — results table (ATE error, relative error, runtime) and bar chart
- Sections 8–10: a second, additional benchmark on RealCause semi-synthetic Lalonde realizations (`load_lalonde_realcause`), replicating CausalPFN's own paper methodology (PSID + CPS cohorts, 10 realizations each, PEHE + ATE relative error) so results are directly comparable to CausalPFN's Table 1 — see [`docs/LALONDE_DATASET.md`](docs/LALONDE_DATASET.md) for why these two benchmarks disagree and shouldn't be compared to each other directly

**3. Foundation Models Sandbox** (practitioner playground, no `causal_bench` wrappers):
```bash
jupyter notebook notebooks/Foundation_models_sandbox.ipynb
```
- **A sandbox/playground notebook, not a quickstart** — per advisor feedback, a real "quickstart" should cover a single model (CausalPFN) end to end; that's what notebook 1 above is for. This notebook instead runs all three foundation models side by side and is checked into git specifically to preserve the observations, caveats, and gotchas found while exploring them together.
- Standalone — calls each library's own native API directly (`CATEEstimator`/`ATEEstimator`, `DoPFNRegressor`, `StandardCATEModel`), not this repo's wrapper classes
- Simulates one self-contained example dataset inline (a confounded discount-email scenario with known heterogeneous CATE) instead of using `causal_bench.data_generators`
- One cell per model, each with markdown explaining install/setup and any library-specific gotchas

### Regenerate notebooks from scripts

If you modify the notebook generator scripts, regenerate:

```bash
python scripts/build_new_notebooks.py   # writes notebooks/Lalonde_benchmark.ipynb
```

Before publishing to Colab: update `REPO_SLUG` in `build_new_notebooks.py` with your GitHub `owner/repo`.

### Using on Google Colab

Each notebook has a "Open in Colab" badge. Click it to run on Colab directly (all installs happen automatically). Or:

1. Copy notebook URL to Colab
2. Colab will clone the repo and install dependencies on first run
3. Run all cells top-to-bottom

### VS Code + Jupyter

1. Install Jupyter extension in VS Code
2. Open a notebook file (`.ipynb`)
3. Click "Select Kernel" → choose your Python environment
4. Run cells individually or with "Run All"

### Quick smoke-test of `causal_bench`

```python
from causal_bench import get_dataset, evaluate_cate
ds = get_dataset("nonlinear_heterogeneous", n=200, seed=0)
train_idx, test_idx = ds.train_test_split(0.7, seed=0)
# feed ds.X, ds.T, ds.Y[train_idx] to any wrapper, evaluate on test_idx
```

## Architecture

### `causal_bench/` — shared library

**`data_generators.py`** — Synthetic dataset generators:
- `linear_confounded` — constant CATE, observed confounders, backdoor
- `nonlinear_heterogeneous` — heterogeneous CATE `τ(x)=sin(x₀)+0.5x₁`, backdoor
- `iv_binary` — hidden confounder + binary instrument
- `frontdoor` — hidden confounder + mediator

All generators register in `GENERATORS` and accept `(n, seed)` kwargs.

**`data_loader.py`** — Real-world dataset loading:
- `load_lalonde()` — downloads Lalonde benchmark directly from Dehejia's NBER page (no `causalml` dependency); see `docs/LALONDE_DATASET.md`
- `load_lalonde_realcause(cohort, n_realizations=10)` — downloads RealCause semi-synthetic Lalonde realizations (PSID or CPS) directly from `vdblm/CausalPFN`'s own repo, replicating CausalPFN's paper methodology (individual-level CATE ground truth, 10 realizations averaged); see `docs/LALONDE_DATASET.md`

**`metrics.py`** — Evaluation metrics:
- `evaluate_cate(tau_hat, tau_true, ...)` → dict with pehe, ate_error, bias, coverage_95, runtime_s

**`wrap_*.py`** files — Model wrappers:
- **Foundation models** (`wrap_causalpfn.py`, `wrap_dopfn.py`, `wrap_causalfm.py`)
- **Metalearners** (`wrap_metalearners.py`: S/T/X-learner, Debiased ML, IPW, DR)

All wrappers follow a common interface:
```python
class *Wrapper:
    name: str
    @classmethod
    def is_available() -> bool      # Check if library is installed
    def fit(X, T, Y) -> self        # Store data / load model
    def predict(X) -> (tau_hat, lower, upper)
    def run(X_train, T_train, Y_train, X_test) -> (tau_hat, lower, upper, ate_hat, runtime)
```

### `notebooks/`

**`Foundation_models_quickstart.ipynb`** — Fastest path to one working foundation model
- Covers **CausalPFN** alone, end to end: simulate data, install, fit, predict, check against ground truth
- Standalone — calls CausalPFN's native API directly, not this repo's wrapper classes
- Not built by `scripts/build_new_notebooks.py`; hand-maintained like the sandbox notebook below, since it doesn't share the Lalonde notebook's wrapper-based structure

**`Lalonde_benchmark.ipynb`** — Real-world comparison, uses `causal_bench`
- Automatically runs all 3 foundation models + all 6 metalearners on Lalonde, no selection needed
- Unavailable/erroring models are skipped/reported per-model, without blocking the rest
- Produces results table (ATE error, runtime) and bar charts
- Built by `scripts/build_new_notebooks.py` — edit the script, not the `.ipynb`, then regenerate

**`Foundation_models_sandbox.ipynb`** — Practitioner playground/sandbox, standalone
- **Not a quickstart** — per advisor feedback, a real quickstart should cover just CausalPFN end to end; that's `Foundation_models_quickstart.ipynb` above. This is a sandbox that runs all three foundation models side by side and is kept in git to preserve the exploratory observations and gotchas found along the way.
- Not built by `scripts/build_new_notebooks.py`; built by a one-off script (see repo history) since it doesn't share the Lalonde notebook's wrapper-based structure
- Does not import `causal_bench` — every foundation model call is that library's own native API, so a cell can be copy-pasted into another project as-is
- Data is a small inline simulation (not `causal_bench.data_generators`), chosen for a business narrative rather than an abstract `X0, X1, ...` matrix
- Has a "Reference output" section (real numbers + plot from a verified Colab GPU run, `SEED=42`, image at `notebooks/assets/reference_output_colab.png`) so practitioners can sanity-check their own run against a known-good one

### `scripts/`

**`build_new_notebooks.py`** — Generates `notebooks/Lalonde_benchmark.ipynb`
- Define notebook structure as Python code (using nbformat)
- Re-run after editing the script to regenerate the `.ipynb` file
- Does **not** build the quickstart or sandbox notebooks — those are hand-maintained (see above)

## Key caveats & usage notes

### Datasets

- **`iv_binary` / `frontdoor`**: Intentionally violate unconfoundedness-given-X. Methods that assume unconfoundedness will be biased (expected behavior, used to test robustness).
- **Lalonde**: Real-world data with no ground-truth *CATE*, but `load_lalonde()` does supply a true experimental *ATE* (`ds.ate`, ~$1,794) from a randomized comparison that's separate from the (X, T, Y) fed to models — see `causal_bench/data_loader.py` for why, and the wrapper-standardization note below for why that matters for scoring foundation models fairly on it. `Lalonde_benchmark.ipynb` defaults to `variant="nsw_psid_trimmed"` (common propensity-score-support trimming of the PSID controls) rather than the untrimmed `"nsw_psid"` — the untrimmed pairing has so little covariate overlap that most models get the *sign* of the ATE wrong, not just the magnitude, which is a real property of the data, not a model failure to fix. Full explanation: [`docs/LALONDE_DATASET.md`](docs/LALONDE_DATASET.md).
- **Lalonde numbers won't match CausalPFN's paper unless you use `load_lalonde_realcause()`**: CausalPFN's own reported Lalonde results are not computed on the real NBER data `load_lalonde()` loads — they use RealCause (Neal et al. 2020) semi-synthetic realizations (simulated potential outcomes over real covariates, giving individual-level CATE ground truth that real data can never supply), averaged over 10 realizations, for both PSID and CPS cohorts. `causal_bench/data_loader.py::load_lalonde_realcause(cohort, n_realizations=10)` downloads CausalPFN's own checked-in realization CSVs (`vdblm/CausalPFN` repo) and replicates their train/test split and CATE/ATE evaluation exactly; `Lalonde_benchmark.ipynb` sections 8–10 run all 9 models through it and report mean PEHE / mean ATE relative error ± SEM, directly comparable to the paper's Table 1. Full comparison of the two Lalonde benchmarks: [`docs/LALONDE_DATASET.md`](docs/LALONDE_DATASET.md#two-lalonde-benchmarks-in-this-repo-real-nber-data-vs-realcause-semi-synthetic).

### Models & dependencies

- **Foundation-model wrappers standardize X and Y internally** (`wrap_causalpfn.py`, `wrap_dopfn.py`, `wrap_causalfm.py`, all via `sklearn.preprocessing.StandardScaler`): all three models are pretrained on normalized synthetic priors, so raw real-world scales (Lalonde's `re74`/`re75`/`re78` are in thousands of dollars) put them badly out of distribution — verified directly: on Lalonde, CausalFM's ATE error dropped from being off by the full scale of the outcome to a small fraction of it once inputs were standardized. `fit()` fits scalers on the training data; `predict()`/`estimate_ate()` transform inputs and multiply CATE/ATE back by `Y`'s std to return original-scale values (the mean cancels out in a difference of group means, so no need to add it back). Do-PFN's treatment column is deliberately excluded from scaling — `predict_cid` sets it to exactly `0`/`1` internally, so it must stay untransformed. Metalearners (`wrap_metalearners.py`, tree/linear models via econml) are not touched by this — they're comparatively scale-robust and were already working correctly unscaled.
- **CausalPFN on Apple Silicon macOS**: segfaults (a hard process crash — not a catchable Python exception) on *both* `device="cpu"` and `device="mps"`. Verified directly, not inherited from the package's own docs: the installed `causalpfn` package has no `torch.compile` call anywhere in it, so this isn't a compiled-for-CUDA artifact; the likely cause is `F.scaled_dot_product_attention` (`models/transformer_layer.py`) hitting an unstable SDPA kernel on macOS's CPU/MPS backends — a known class of PyTorch bug, not a hard CUDA-only architectural requirement. Because it's a segfault, code must check the platform/device combo *before* calling into CausalPFN rather than wrapping the call in `try/except` — `causal_bench/wrap_causalpfn.py`'s `CausalPFNWrapper.is_available()` returns `False` unconditionally on `platform.system() == "Darwin" and platform.machine() == "arm64"`, and `notebooks/Foundation_models_sandbox.ipynb`'s CausalPFN cell applies the same guard directly (it doesn't use the wrapper). Untested but likely fine: Colab GPU (CUDA, best-supported) and Colab CPU (Linux x86_64, mature SDPA kernels) — the bug looks macOS-specific rather than universal to non-CUDA devices.
- **`econml` on Colab pulls a breaking pandas upgrade**: `econml`'s own metadata declares only `pandas>1.0` — no upper bound (verified via `importlib.metadata.requires("econml")`) — so `%pip install -q econml causalpfn` on a stock Colab runtime resolves to the newest pandas (3.x) rather than keeping Colab's preinstalled 2.2.2. That breaks Colab's own `google-colab`/`cudf-cu12`/`dask-cudf-cu12` packages, which all require `pandas<2.4`. `notebooks/Lalonde_benchmark.ipynb`'s install cell (the only one that installs `econml`) pins `"pandas<2.4"` alongside `econml causalpfn` to keep pip's resolver from upgrading it — this repo's own `pd.read_csv`/`pd.DataFrame`/`pd.concat` usage (`data_loader.py`, the results-table cells) is also untested against pandas 3.x, so this isn't a Colab-only cosmetic warning like the `huggingface-hub` one below; an unconstrained install can genuinely break the run.
- **CausalFM**: `CausalFMWrapper` requires a checkpoint path. `CausalFM-toolkit` is **not on PyPI** and is not `pip install`-able as a package — it must be `git clone`d and its root added to `sys.path` (both the `causalfm` package and its internal `src.tabpfn` module live there). It also needs `einops`, `tabpfn==2.0.9`, and `tensorboard` — packages the toolkit's own `requirements.txt` bundles inside a full frozen dev-env snapshot (including Linux/CUDA-only pins) that should **not** be installed as-is; install just those three instead (see the venv caveat below for how). The real pretrained checkpoint ships at `checkpoints/checkpoints_standard/best_model.pth` inside the toolkit repo (not `checkpoints/best_model.pth`, despite what the toolkit's own docs/README examples show). `StandardCATEModel.estimate_cate` expects `torch.Tensor` inputs with treatment/outcome shaped `[N, 1]`, not the 1-D numpy arrays used elsewhere in this repo's common wrapper interface — `wrap_causalfm.py` converts internally.
  - **Expected, harmless pip warning on Colab**: installing `tabpfn==2.0.9` (which itself declares `huggingface-hub<1,>=0.0.1` in its own metadata — verified via `importlib.metadata.requires("tabpfn")`) conflicts with Colab's preinstalled `gradio`/`transformers`, which require `huggingface-hub>=1.x`. pip reports this as a dependency-conflict warning but still installs everything requested (downgrading `huggingface-hub`) rather than failing. Since this repo never imports `gradio`/`transformers`, the warning is safe to ignore — verified end-to-end: CausalFM loads and produces correct results right after it.
- **`uv`-managed local venv has no `pip` module**: notebook cells that use `%pip install ...` (or `!pip install ...`) silently no-op locally with `No module named pip` — they only work on Colab, where `pip` is preinstalled. Locally, use `uv pip install <pkg>` instead of relying on the notebook's pip cells.
- **`uv sync --extra metalearners` is currently broken on Python 3.10** (the version this project targets): it resolves `llvmlite==0.36.0` via `numba`/`sparse`, which only supports Python `<3.10` and fails to build. The `metalearners` extra exists in `pyproject.toml` for documentation/reference, but installing it locally means `uv pip install econml causalml` (ad hoc — this resolves a different, Python-3.10-compatible `llvmlite`/`numba` pair, bypassing the broken lock resolution) rather than `uv sync --extra metalearners`. Likewise for `causalfm`: prefer `uv pip install einops "tabpfn==2.0.9" tensorboard` over `uv sync --extra causalfm`. Reserve `uv sync` (no extras, or `uv sync` alone) for the core deps only — running it with any extra reconciles the whole environment to exactly what's declared and will **uninstall** ad hoc-installed packages from extras you didn't pass.
- **Do-PFN**: Not on PyPI — `git clone https://github.com/jr2021/Do-PFN.git` and add its root to `sys.path`. Several gotchas verified directly by tracing the current `jr2021/Do-PFN` main branch source, none obvious from its README. `causal_bench/wrap_dopfn.py`'s `DoPFNWrapper` implements all of these (constructor takes `repo_dir: str = "Do-PFN"`); `notebooks/Foundation_models_sandbox.ipynb`'s Do-PFN cell applies the same fixes directly against the native API (it doesn't use the wrapper):
  - **Don't `pip install -r Do-PFN/requirements.txt` as-is** — like CausalFM-toolkit's, it's a frozen research/benchmark environment, not a minimal runtime spec, and pins `catboost==1.1.1`, which has no wheel for recent Python and makes the whole install fail. `DoPFNRegressor` itself never imports `catboost` (only a baseline-comparison script does); the actual runtime deps beyond `torch`/`numpy`/`scipy`/`pandas`/`scikit-learn` are just `networkx`, `tqdm`, `einops`.
  - **Correct import** is `from scripts.transformer_prediction_interface import DoPFNRegressor` — not `from dopfn import ...` or `from model.dopfn import ...` (there is no top-level `dopfn` package in the current repo at all).
  - **Treatment must be column 0** of the feature matrix passed to `fit`/`predict`, not appended at the end — `predict_cid` does `X[:, 0] = t` internally, so whatever's in that column at prediction time gets overwritten anyway.
  - **`DoPFNRegressor()` loads its checkpoint via paths relative to the Do-PFN repo root** (e.g. `artifacts/dopfn_config.pkl`), lazily on *both* construction and the first `fit()` call — so the working directory must be `Do-PFN/` for construction, `fit`, and `predict`/`predict_cate` alike (`os.chdir` there and restore it in a `finally`, since other cells rely on other relative paths from the notebook's own cwd).
  - Use the dedicated `predict_cate(X)` method (computes `do(T=1)` minus `do(T=0)` internally) rather than calling `predict_full` twice by hand — but it requires `X` to be a `torch.Tensor`; passing a plain numpy array fails inside `predict_common_setup`'s `X_eval.cpu().detach().numpy()` (which assumes tensor input unconditionally).
  - **Torch version sensitivity**: Do-PFN's own model code (`model/layer.py`) imports `Optional`, `Tensor`, `Module`, `Linear`, `Dropout`, `LayerNorm`, `MultiheadAttention` from `torch.nn.modules.transformer` — an unofficial re-export. Verified directly against PyTorch's own source history: `Optional` is present through the `v2.9.0` tag and gone as of `v2.10.0` (the other names survive). So this raises `ImportError: cannot import name 'Optional' from 'torch.nn.modules.transformer'` on torch ≥ 2.10 — reproduced both locally (torch 2.12.1) and on Colab (which currently ships torch ≥ 2.10 by default). An upstream compatibility gap in Do-PFN itself, not something to patch around here. **Verified fix**: `torch<2.10` — safe for the other two models too (CausalPFN only requires `torch>=2.0`, its own dev/test pin is `torch==2.3.1`; CausalFM's `tabpfn==2.0.9` requires `torch<3,>=2.1`).
    - `notebooks/Foundation_models_sandbox.ipynb` and `notebooks/Lalonde_benchmark.ipynb` both handle this with a dedicated "one-time environment check" cell that must run *first, before anything else*. It pins `torch==2.9.1` (last version verified compatible with Do-PFN) via a plain `pip install`, gated on `"google.colab" in sys.modules` — no restart-detection/auto-restart machinery at all. This deliberately relies on the check reading the installed version via `importlib.metadata.version("torch")`, which does *not* import torch, so on a fresh runtime nothing has loaded the incompatible version into memory yet before the fix lands — the freshly `pip install`ed one is just what any later `import torch` picks up. `Lalonde_benchmark.ipynb`'s version of this cell **must run before its own `causal_bench` import** — `causal_bench/wrap_dopfn.py` does a top-level `import torch`, so importing `causal_bench` first would pre-import torch and break this every time; that's why the cell ordering there is env-check → setup/`causal_bench` import, not the reverse. If torch was already imported earlier in the session (e.g. re-running after a previous full run), the `pip install` alone won't fix an already-loaded compiled extension — restart the runtime and re-run from the top; the notebooks don't attempt to auto-detect or auto-restart for this rarer case, by design (see next bullet for why).
    - **`Lalonde_benchmark.ipynb`'s setup cell separately handles a stale-`causal_bench`-import case** (re-running after a `git pull` picked up code changes) where a restart genuinely can't be avoided — unlike the torch check, there's no way to dodge it, since `causal_bench` is by definition already imported when this fires. That one still goes through `display(Javascript("google.colab.kernel.restart()"))` — the same API Colab's own "Runtime > Restart session" menu item uses, so the frontend shows a normal "Restarting..." indicator instead of the false "Your session crashed for an unknown reason" error a raw `os.kill(os.getpid(), 9)` triggers (verified directly — `SIGKILL` does restart the process fine, but Colab's frontend has no way to distinguish an unannounced kill from an actual crash). **This API itself isn't fully reliable either** — reported directly by a user: it sometimes doesn't fire the restart at all, silently — which is exactly why the torch check above was reworked to avoid needing any restart mechanism in the common case, rather than trying to make the restart itself more reliable. The cell has an explicit fallback message ("if nothing happens within a few seconds, use Runtime > Restart session yourself"), since there's no way to detect failure and retry from inside a kernel that may or may not still be alive. Either way, no restart mechanism can resume notebook execution on its own — re-run the cell once more afterward, then continue top to bottom.
    - **This cell cannot self-fix locally**: this repo's `uv`-managed venv has no `pip` module at all (`No module named pip`), so a `!pip install` line inside the notebook is a silent no-op there — the earlier version of this fix instruction was locally broken for exactly this reason. Locally the cell only *detects* the problem and prints the real fix to run in a terminal: `uv pip install "torch<2.10"`, then restart the notebook's kernel.
    - **Re-running the Do-PFN cell without restarting the kernel does NOT retry cleanly.** Reproduced directly: on a fresh process, `from scripts.transformer_prediction_interface import DoPFNRegressor` correctly raises the `Optional` `ImportError`. But if the *same* import is attempted again in the *same* kernel session (e.g. the user just re-runs the cell after it failed), Python leaves the `scripts` package half-cached in `sys.modules` from the first failed attempt, and the second attempt silently "succeeds" — handing back a broken `DoPFNRegressor` class — only to fail later, uncaught, inside `dopfn.fit()`/`predict_cate()` (deep in `TabPFNBaseModel.init_model_and_get_model_config()`). The notebook's Do-PFN cell wraps the *entire* import-through-predict flow in one `try/except` so this degrades gracefully instead of crashing with a raw traceback, but the underlying fix is still: run "§0", then actually restart the kernel/runtime, then re-run all cells from the top — not just re-run the Do-PFN cell in place.
- **Metalearners**: Require `econml` + `causalml` (the `metalearners` extra in `pyproject.toml` / `requirements.txt`).
- **Missing models**: Notebooks use `try/except` for installs; unavailable models are skipped with a warning.

### Running & testing

- No automated test suite; validation is via end-to-end notebook execution.
- To verify setup, run the smoke-test (see Commands section above).
- Notebooks save results to CSV + PNG for easy comparison.
