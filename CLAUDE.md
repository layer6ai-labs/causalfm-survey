# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Benchmark companion code for this project on **causal foundation models** — zero-shot, amortised CATE estimators trained via PFN-style meta-learning. Compare three foundation models against six traditional metalearners on synthetic and real-world data.

### Foundation Models

| Model | Install method |
|---|---|
| **CausalPFN** | Core dependency, `causalpfn>=0.1.5` from PyPI (0.1.5 is the first release with the Apple Silicon and `estimate_ate_CI` fixes); weights downloaded from HF Hub on first use |
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
uv sync --all-extras              # core + metalearners + causalfm deps
jupyter notebook notebooks/Foundation_models_quickstart.ipynb   # start exploring
```

Or with pip:

```bash
pip install -r requirements.txt   # core deps
jupyter notebook notebooks/Foundation_models_quickstart.ipynb
```

There's no dedicated setup/data notebook — synthetic dataset generation is inline via `causal_bench.data_generators` (see the smoke-test below), used directly wherever a notebook needs a dataset.

### Running notebooks

Three notebooks, no numeric ordering — pick based on what you need. (The Lalonde benchmark *run itself* used to be a notebook; it's now `scripts/run_benchmark.py` — see the next section. `Lalonde_benchmark_results.ipynb` below only reads that script's already-committed output, it doesn't run the benchmark.)

**1. Foundation Models Quickstart** (fastest path to one working model):
```bash
jupyter notebook notebooks/Foundation_models_quickstart.ipynb
```
- Covers **CausalPFN** alone, end to end: install, fit, predict. Standalone — calls CausalPFN's native API directly, not this repo's wrapper classes, so any cell can be copied into another project as-is.
- Deliberately narrow (one model, one dataset) — see the sandbox notebook below for a three-model comparison.

**2. Foundation Models Sandbox** (practitioner playground, no `causal_bench` wrappers):
```bash
jupyter notebook notebooks/Foundation_models_sandbox.ipynb
```
- **A sandbox/playground notebook, not a quickstart** — per advisor feedback, a real "quickstart" should cover a single model (CausalPFN) end to end; that's what notebook 1 above is for. This notebook instead runs all three foundation models side by side and is checked into git specifically to preserve the observations, caveats, and gotchas found while exploring them together.
- Standalone — calls each library's own native API directly (`CATEEstimator`/`ATEEstimator`, `DoPFNRegressor`, `StandardCATEModel`), not this repo's wrapper classes
- Simulates one self-contained example dataset inline (a confounded discount-email scenario with known heterogeneous CATE) instead of using `causal_bench.data_generators`
- One cell per model, each with markdown explaining install/setup and any library-specific gotchas

**3. Lalonde Benchmark Results** (loads and visualizes `scripts/run_benchmark.py`'s output):
```bash
jupyter notebook notebooks/Lalonde_benchmark_results.ipynb
```
- Reads the production run's already-committed results — `data/benchmark_results_cpu.csv` (all 9 models) and `data/benchmark_results_gpu.csv` (the 3 foundation models, re-run on GPU) — no benchmark run required to use this notebook
- Averages over the 10 realizations per cohort into a summary table (PEHE in units of $1,000 to match CausalPFN's Table 1 units, plus per-cohort and pooled PEHE/ATE-relative-error ranks), and a rank-vs-runtime figure styled after CausalPFN's own paper Figure 1 (foundation models vs. metalearners vs. Debiased ML called out separately, since it's the best-ranked non-foundation model)
- Deliberately does **not** normalize runtime "per 1,000 samples" the way the paper's figure axis is labeled: the metalearners' HPO time is a fixed 900s *budget* per sub-model regardless of dataset size (verified directly: CPS has ~6x PSID's sample count but near-identical metalearner runtimes), so normalizing by sample count would make PSID look artificially cheaper for a cost that's actually constant. Raw per-task CPU seconds is used instead.
- Uses `causal_bench` wrappers only insofar as the CSVs it reads were produced by them — the notebook itself is pure pandas/matplotlib over the CSVs, no model libraries imported

### Running the benchmark script

**`scripts/run_benchmark.py`** is the canonical Lalonde benchmark — compare all 3 foundation models vs. all 6 metalearners:
```bash
python scripts/run_benchmark.py --smoke   # fast: IPW only, no HPO, 1 realization/cohort
python scripts/run_benchmark.py           # full: 9 models x 2 cohorts x 10 realizations, HPO-tuned
```
- Runs all 3 foundation models + all 6 metalearners (with FLAML-tuned nuisance models, `hpo=True` unless `--smoke`) on RealCause semi-synthetic Lalonde realizations (`load_lalonde_realcause`), replicating CausalPFN's arXiv v1 "first-10" protocol so results are directly comparable to their Table 1 — see [`docs/LALONDE_DATASET.md`](docs/LALONDE_DATASET.md)
- Written to survive a genuinely expensive full run (production default: ~105 CPU-hours of FLAML search) rather than a quick one: per-task CSV checkpointing after every (model, cohort, realization) via an atomic temp-file-then-`os.replace`, fail-fast with a full traceback logged to both stdout and `logs/run_benchmark_<timestamp>.log`. Flags: `--cpu`/`--gpu` (device), `--smoke`, `--n-realizations`, `--hpo-time-budget`, `--results-dir` — see `python scripts/run_benchmark.py --help`.
- Uses `causal_bench` wrappers (unlike the quickstart/sandbox notebooks, which call each library's native API directly), including each wrapper's `estimate_ate()` method (fit fresh on the full realization for the ATE metric, separate from the train/test CATE fit)
- Not a notebook — deliberately a plain script, since a genuinely expensive unattended run (hours) benefits from a real process you can `nohup`/background and tail a log file for, rather than a kernel you have to keep a browser tab open against

### Notebook history

This repo used to have three separate Lalonde-family notebooks: `Lalonde_benchmark.ipynb`
(real NBER data — no CATE ground truth possible, only ATE), `RealCause_benchmark.ipynb`
(RealCause semi-synthetic, no HPO), and `RealCause_with_hpo_benchmark.ipynb` (RealCause
semi-synthetic, with HPO). All three have since been deleted: this project is about CATE
accuracy, real data can never supply that ground truth no matter the methodology (see
`docs/LALONDE_DATASET.md`), and once a proper HPO-enabled RealCause notebook existed there
was no reason to keep a non-HPO duplicate of it around either. `causal_bench.load_lalonde()`
itself is unchanged if you ever want to reproduce the real-data-only comparison.
`scripts/build_new_notebooks.py`, which used to generate the two first-deleted notebooks from
a shared Python builder, has also been deleted.

`RealCause_with_hpo_benchmark.ipynb` itself was later replaced by **`scripts/run_benchmark.py`**
(a plain Python script, not a notebook) — a teammate's rewrite, merged into this repo — since a
genuinely expensive unattended run (hours of FLAML search) is a better fit for a backgroundable
process with a log file than a notebook kernel. The quickstart and sandbox notebooks are
unaffected and remain hand-maintained, as they always were.

Once a teammate's full production run of `scripts/run_benchmark.py` (~105 CPU-hours) finished
and its output CSVs were checked into `data/`, **`Lalonde_benchmark_results.ipynb`** was added
to load and visualize those already-computed results — a table and a paper-style rank-vs-runtime
figure — without re-running anything. It's a notebook (not a script) because, unlike the
benchmark run itself, this is fast, interactive, and meant to be tweaked cell-by-cell.

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
- **Foundation models** (`wrap_causalpfn.py`, `wrap_dopfn.py`, `wrap_causalfm.py`), sharing a common `_StandardizedFoundationWrapper` base class in `wrap_foundation.py` for input validation, standardization, and the `run()` contract
- **Metalearners** (`wrap_metalearners.py`: S/T/X-learner, Debiased ML, IPW, DR), with optional FLAML-based HPO (`hpo=True`)
- Full detail on both, including the HPO/standardization story: [`docs/WRAPPERS_GUIDE.md`](docs/WRAPPERS_GUIDE.md)

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
- Hand-maintained, like every notebook in this repo now (see "Notebook history" above)

**`Foundation_models_sandbox.ipynb`** — Practitioner playground/sandbox, standalone
- **Not a quickstart** — per advisor feedback, a real quickstart should cover just CausalPFN end to end; that's `Foundation_models_quickstart.ipynb` above. This is a sandbox that runs all three foundation models side by side and is kept in git to preserve the exploratory observations and gotchas found along the way.
- Does not import `causal_bench` — every foundation model call is that library's own native API, so a cell can be copy-pasted into another project as-is
- Data is a small inline simulation (not `causal_bench.data_generators`), chosen for a business narrative rather than an abstract `X0, X1, ...` matrix
- Has a "Reference output" section (real numbers + plot from a verified Colab GPU run, `SEED=42`, image at `assets/reference_output_sandbox.png`) so practitioners can sanity-check their own run against a known-good one
- Committed with each code cell's own execution output cleared (`jupyter nbconvert --clear-output --inplace`) — those were transient artifacts of whichever local/Colab run last executed the notebook, not curated content; the static "Reference output" section above is unaffected since it lives in a separate markdown cell + image file, not a cell output

**`Lalonde_benchmark_results.ipynb`** — Loads and visualizes `scripts/run_benchmark.py`'s output, standalone
- Reads `data/benchmark_results_cpu.csv` / `data/benchmark_results_gpu.csv` directly with pandas — doesn't import `causal_bench` or re-run anything
- Builds a summary table (PEHE ÷ 1,000 to match CausalPFN's Table 1 units; PEHE and ATE-relative-error, each with a per-cohort and pooled rank) and a rank-vs-runtime figure styled after CausalPFN's own paper Figure 1
- Colored by group in the figure: Foundation Models, Debiased ML (called out on its own — the best-ranked non-foundation model), and the remaining metalearners
- Committed **with** its cell outputs (unlike the sandbox notebook above) — the table and figures are the point, so a viewer should see them without needing to run anything locally
- Hand-maintained, like every notebook in this repo now (see "Notebook history" above)

### `scripts/`

**`run_benchmark.py`** — The canonical Lalonde benchmark, uses `causal_bench` (see "Running the benchmark script" above)
- Runs all 3 foundation models + all 6 metalearners (FLAML-tuned via `hpo=True` unless `--smoke`) on RealCause semi-synthetic realizations, replicating CausalPFN's arXiv v1 protocol exactly (see `docs/LALONDE_DATASET.md`)
- Structured for a genuinely expensive unattended run rather than a quick one: builds the model registry from `--cpu`/`--gpu`/`--smoke`/etc. CLI flags; loads RealCause realizations; runs every (model, cohort, realization) task, appending to `records` and rewriting the results CSV atomically (temp file + `os.replace`) after each one so an interrupted run keeps everything completed so far; logs progress to stdout and `logs/run_benchmark_<timestamp>.log`; prints a summary table at the end (PEHE divided by 1,000 to match the paper's units)
- Supersedes `RealCause_with_hpo_benchmark.ipynb` (and, transitively, `RealCause_benchmark.ipynb`/`Lalonde_benchmark.ipynb`), all now deleted — see "Notebook history" above

### `data/`

**`benchmark_results_cpu.csv`** / **`benchmark_results_gpu.csv`** — Checked-in output of a teammate's full production `scripts/run_benchmark.py` run (~105 CPU-hours): CPU has all 9 models (2 cohorts x 10 realizations each, 180 rows); GPU has just the 3 foundation models re-run for a realistic-deployment-speed comparison (60 rows). Read directly by `notebooks/Lalonde_benchmark_results.ipynb` — see above.

## Key caveats & usage notes

### Datasets

- **`iv_binary` / `frontdoor`**: Intentionally violate unconfoundedness-given-X. Methods that assume unconfoundedness will be biased (expected behavior, used to test robustness).
- **Lalonde**: Real-world data with no ground-truth *CATE*, but `load_lalonde()` does supply a true experimental *ATE* (`ds.ate`, ~$1,794) from a randomized comparison that's separate from the (X, T, Y) fed to models — see `causal_bench/data_loader.py` for why, and the wrapper-standardization note below for why that matters for scoring foundation models fairly on it. Pass `variant="nsw_psid_trimmed"` (common propensity-score-support trimming of the PSID controls) rather than using the untrimmed `"nsw_psid"` default — the untrimmed pairing has so little covariate overlap that most models get the *sign* of the ATE wrong, not just the magnitude, which is a real property of the data, not a model failure to fix. No notebook currently exercises `load_lalonde()` directly (see "Notebook history" above) — this repo's actual benchmark uses `load_lalonde_realcause()` instead, since CATE (not just ATE) is what this benchmark measures. Full explanation: [`docs/LALONDE_DATASET.md`](docs/LALONDE_DATASET.md).
- **Lalonde numbers won't match CausalPFN's paper unless you use `load_lalonde_realcause()`**: CausalPFN's own reported Lalonde results are not computed on the real NBER data `load_lalonde()` loads — they use RealCause (Neal et al. 2020) semi-synthetic realizations (simulated potential outcomes over real covariates, giving individual-level CATE ground truth that real data can never supply), averaged over 10 realizations, for both PSID and CPS cohorts. `causal_bench/data_loader.py::load_lalonde_realcause(cohort, n_realizations=10)` downloads CausalPFN's own checked-in realization CSVs (`vdblm/CausalPFN` repo) and replicates their train/test split and CATE/ATE evaluation exactly; `scripts/run_benchmark.py` runs all 9 models through it and reports mean PEHE / mean ATE relative error ± SEM, directly comparable to the paper's Table 1. Full explanation of why this repo benchmarks on RealCause rather than the real data: [`docs/LALONDE_DATASET.md`](docs/LALONDE_DATASET.md).
- **RealCause has two easy-to-miss gotchas when comparing to the paper**: (1) the default `n_realizations=10` matches CausalPFN's **arXiv v1** protocol (first 10 of the 100 available realizations); the current **arXiv v2** averages all 100 and reports different numbers — this repo intentionally targets v1, since v1's numbers are what's checked into the realization CSVs it downloads. (2) The paper's Table 1 PEHE numbers are in **units of $1,000** — `evaluate_cate`'s PEHE is raw dollars, so divide by 1,000 before comparing (`scripts/run_benchmark.py`'s summary table already does this). Skipping either of these makes the numbers look wildly wrong (PEHE off by ~1000x) when the pipeline is actually working correctly. Full explanation: [`docs/LALONDE_DATASET.md`](docs/LALONDE_DATASET.md#two-things-to-know-before-comparing-to-the-papers-numbers).

### Models & dependencies

- **Foundation-model wrappers standardize X and Y internally** (`wrap_causalpfn.py`, `wrap_dopfn.py`, `wrap_causalfm.py`, all via `sklearn.preprocessing.StandardScaler`): all three models are pretrained on normalized synthetic priors, so raw real-world scales (Lalonde's `re74`/`re75`/`re78` are in thousands of dollars) put them badly out of distribution — verified directly: on Lalonde, CausalFM's ATE error dropped from being off by the full scale of the outcome to a small fraction of it once inputs were standardized. `fit()` fits scalers on the training data; `predict()`/`estimate_ate()` transform inputs and multiply CATE/ATE back by `Y`'s std to return original-scale values (the mean cancels out in a difference of group means, so no need to add it back). Do-PFN's treatment column is deliberately excluded from scaling — `predict_cid` sets it to exactly `0`/`1` internally, so it must stay untransformed. The EconML-based wrappers in `wrap_metalearners.py` also standardize X and Y and unscale effects, matching CausalPFN's reference `EconMLBaseline` preprocessing exactly. `IPWWrapper` intentionally remains on raw X and Y because the reference `IPWBaseline` is separate from `EconMLBaseline` and uses raw inputs with Hájek-normalized weighted group means.
- **CausalPFN credible intervals**: `causalpfn` 0.1.4 (the PyPI release) has a broken `CATEEstimator.estimate_ate_CI` (`KeyError: 'ate'` on every call); fixed upstream in vdblm/CausalPFN#14, first released in 0.1.5 (the minimum this repo pins). `CausalPFNWrapper.estimate_att_ci()` calls the public method; `predict(alpha=...)` uses `estimate_cate_CI` (whose bounds upstream #15 squeezed to shape `(n_queries,)`). A `KeyError` from it means a stale 0.1.4 install was picked up instead of `>=0.1.5`. Consequence worth knowing: `coverage_95` is blank for CausalPFN in `data/benchmark_results_*.csv` because the wrapper returned `None` intervals; passing `alpha=` to the constructor now fills that column.
- **CausalPFN on Apple Silicon macOS**: works natively with `causalpfn>=0.1.5`. The 0.1.4 PyPI release segfaults there (on both `device="cpu"` and `"mps"`): it imports `faiss` before `torch`, and `faiss-cpu` and `torch` each bundle their own `libomp.dylib`, which corrupt each other on arm64. This repo used to carry a NumPy faiss shim (`causal_bench/faiss_shim.py`/`macos_compat.py`, plus inline copies in the notebooks); upstream vdblm/CausalPFN#14 removed faiss entirely, so the shim is gone too. If CausalPFN segfaults on a Mac, the environment has the old 0.1.4 release — re-run `uv sync --all-extras`. Numbers are unchanged vs. the committed CSVs: `CausalPFNWrapper(cap_num_neighbours=True)` (the default, used by `scripts/run_benchmark.py`) still caps neighbours to the smaller arm, exactly as the production run did.
- **`econml` on Colab pulls a breaking pandas upgrade**: `econml`'s own metadata declares only `pandas>1.0` — no upper bound (verified via `importlib.metadata.requires("econml")`) — so an unconstrained `pip install econml` on a stock Colab runtime resolves to the newest pandas (3.x) rather than keeping Colab's preinstalled 2.2.2. That breaks Colab's own `google-colab`/`cudf-cu12`/`dask-cudf-cu12` packages, which all require `pandas<2.4`. The now-deleted `RealCause_with_hpo_benchmark.ipynb` used to sidestep this with an exact pandas pin (`2.3.3`) verified in a dedicated §1 cell; `scripts/run_benchmark.py` doesn't run on Colab at all (it's a local/server CLI script, not installed via notebook `%pip` cells) so that specific failure mode doesn't apply — but this repo's own `pd.read_csv`/`pd.DataFrame`/`pd.concat` usage (`data_loader.py`, the script's summary table) is still untested against pandas 3.x, so pin pandas below `2.4` (`requirements.txt`/`pyproject.toml` currently leave it unpinned) if you hit an unrelated pandas-version issue while running the script locally.
- **CausalFM**: `CausalFMWrapper` requires a checkpoint path. `CausalFM-toolkit` is **not on PyPI** and is not `pip install`-able as a package — it must be `git clone`d and its root added to `sys.path` (both the `causalfm` package and its internal `src.tabpfn` module live there). It also needs `einops`, `tabpfn==2.0.9`, and `tensorboard` — packages the toolkit's own `requirements.txt` bundles inside a full frozen dev-env snapshot (including Linux/CUDA-only pins) that should **not** be installed as-is; install just those three instead (see the venv caveat below for how). The real pretrained checkpoint ships at `checkpoints/checkpoints_standard/best_model.pth` inside the toolkit repo (not `checkpoints/best_model.pth`, despite what the toolkit's own docs/README examples show). `StandardCATEModel.estimate_cate` expects `torch.Tensor` inputs with treatment/outcome shaped `[N, 1]`, not the 1-D numpy arrays used elsewhere in this repo's common wrapper interface — `wrap_causalfm.py` converts internally.
  - **Expected, harmless pip warning on Colab**: installing `tabpfn==2.0.9` (which itself declares `huggingface-hub<1,>=0.0.1` in its own metadata — verified via `importlib.metadata.requires("tabpfn")`) conflicts with Colab's preinstalled `gradio`/`transformers`, which require `huggingface-hub>=1.x`. pip reports this as a dependency-conflict warning but still installs everything requested (downgrading `huggingface-hub`) rather than failing. Since this repo never imports `gradio`/`transformers`, the warning is safe to ignore — verified end-to-end: CausalFM loads and produces correct results right after it.
- **`uv`-managed local venv has no `pip` module**: notebook cells that use `%pip install ...` (or `!pip install ...`) silently no-op locally with `No module named pip` — they only work on Colab, where `pip` is preinstalled. Locally, use `uv pip install <pkg>` instead of relying on the notebook's pip cells.
- **`uv sync` needs `--all-extras` to keep the extras installed**: `uv sync` reconciles the venv to exactly what's declared, so plain `uv sync` **uninstalls** the metalearner/CausalFM packages; use `uv sync --all-extras`. It used to fail on Python 3.10 because `shap`/`sparse` (via `econml`/`causalml`) put no lower bound on `numba`, so the lock resolved `numba 0.53.1`/`llvmlite 0.36.0`, which predate 3.10 and fail to build; `[tool.uv] constraint-dependencies` in `pyproject.toml` now sets floors (`numba>=0.59`, `llvmlite>=0.42`).
- **Do-PFN**: Not on PyPI — `git clone https://github.com/jr2021/Do-PFN.git` and add its root to `sys.path`. Several gotchas verified directly by tracing the current `jr2021/Do-PFN` main branch source, none obvious from its README. `causal_bench/wrap_dopfn.py`'s `DoPFNWrapper` implements all of these (constructor takes `repo_dir: str = "Do-PFN"`); `notebooks/Foundation_models_sandbox.ipynb`'s Do-PFN cell applies the same fixes directly against the native API (it doesn't use the wrapper):
  - **Don't `pip install -r Do-PFN/requirements.txt` as-is** — like CausalFM-toolkit's, it's a frozen research/benchmark environment, not a minimal runtime spec, and pins `catboost==1.1.1`, which has no wheel for recent Python and makes the whole install fail. `DoPFNRegressor` itself never imports `catboost` (only a baseline-comparison script does); the actual runtime deps beyond `torch`/`numpy`/`scipy`/`pandas`/`scikit-learn` are just `networkx`, `tqdm`, `einops`.
  - **Correct import** is `from scripts.transformer_prediction_interface import DoPFNRegressor` — not `from dopfn import ...` or `from model.dopfn import ...` (there is no top-level `dopfn` package in the current repo at all).
  - **Treatment must be column 0** of the feature matrix passed to `fit`/`predict`, not appended at the end — `predict_cid` does `X[:, 0] = t` internally, so whatever's in that column at prediction time gets overwritten anyway.
  - **`DoPFNRegressor()` loads its checkpoint via paths relative to the Do-PFN repo root** (e.g. `artifacts/dopfn_config.pkl`), lazily on *both* construction and the first `fit()` call — so the working directory must be `Do-PFN/` for construction, `fit`, and `predict`/`predict_cate` alike (`os.chdir` there and restore it in a `finally`, since other cells rely on other relative paths from the notebook's own cwd).
  - Use the dedicated `predict_cate(X)` method (computes `do(T=1)` minus `do(T=0)` internally) rather than calling `predict_full` twice by hand — but it requires `X` to be a `torch.Tensor`; passing a plain numpy array fails inside `predict_common_setup`'s `X_eval.cpu().detach().numpy()` (which assumes tensor input unconditionally).
  - **Torch version sensitivity**: Do-PFN's own model code (`model/layer.py`) imports `Optional`, `Tensor`, `Module`, `Linear`, `Dropout`, `LayerNorm`, `MultiheadAttention` from `torch.nn.modules.transformer` — an unofficial re-export. Verified directly against PyTorch's own source history: `Optional` is present through the `v2.9.0` tag and gone as of `v2.10.0` (the other names survive). So this raises `ImportError: cannot import name 'Optional' from 'torch.nn.modules.transformer'` on torch ≥ 2.10 — reproduced both locally (torch 2.12.1) and on Colab (which currently ships torch ≥ 2.10 by default). An upstream compatibility gap in Do-PFN itself, not something to patch around here. **Verified fix**: `torch<2.10` — safe for the other two models too (CausalPFN only requires `torch>=2.0`, its own dev/test pin is `torch==2.3.1`; CausalFM's `tabpfn==2.0.9` requires `torch<3,>=2.1`).
    - **Expected, harmless pip warning on Colab**: pinning `torch==2.9.1` this way conflicts with Colab's preinstalled `torchvision` build, which is tied to a specific torch version (e.g. `torchvision 0.26.0+cu128 requires torch==2.11.0`). pip reports this as a dependency-conflict warning but still installs the pinned torch rather than failing. None of the three foundation models here import `torchvision` (they're tabular/transformer models, not vision models), so a version-mismatched `torchvision` sitting unused in the environment doesn't affect anything this notebook actually runs — same shape of warning, and same "safe to ignore" verdict, as the `huggingface-hub` one above.
    - `notebooks/Foundation_models_sandbox.ipynb` handles this with a dedicated "one-time environment check" cell that must run *first, before anything else*. It pins `torch==2.9.1` (last version verified compatible with Do-PFN) via a plain `pip install`, gated on `"google.colab" in sys.modules` — no restart-detection/auto-restart machinery at all. This deliberately relies on the check reading the installed version via `importlib.metadata.version("torch")`, which does *not* import torch, so on a fresh runtime nothing has loaded the incompatible version into memory yet before the fix lands — the freshly `pip install`ed one is just what any later `import torch` picks up. If torch was already imported earlier in the session (e.g. re-running after a previous full run), the `pip install` alone won't fix an already-loaded compiled extension — restart the runtime and re-run from the top; the cell doesn't attempt to auto-detect or auto-restart for this rarer case, by design. **This cell cannot self-fix locally**: this repo's `uv`-managed venv has no `pip` module at all (`No module named pip`), so a `!pip install` line inside the notebook is a silent no-op there. Locally the cell only *detects* the problem and prints the real fix to run in a terminal: `uv pip install "torch<2.10"`, then restart the notebook's kernel. `scripts/run_benchmark.py` doesn't need any of this Colab-specific dance — it's a local/server script, and `torch<2.10` is already a core dependency in `pyproject.toml`/`requirements.txt` (not a notebook-cell check), so a correctly-installed environment (`uv sync` or `pip install -r requirements.txt`) already satisfies it before the script ever runs.
    - **Historical note — Colab's kernel-restart API**: the now-deleted `Lalonde_benchmark.ipynb`/`RealCause_benchmark.ipynb` used to separately handle a stale-`causal_bench`-import case (re-running after a `git pull` picked up code changes) by calling `display(Javascript("google.colab.kernel.restart()"))` — the same API Colab's own "Runtime > Restart session" menu item uses, so the frontend shows a normal "Restarting..." indicator instead of the false "Your session crashed for an unknown reason" error a raw `os.kill(os.getpid(), 9)` triggers. Worth remembering if a future notebook needs this again: verified directly that the API itself isn't fully reliable — it can silently fail to fire the restart at all — which is part of why the current notebook doesn't rely on it (the sandbox's torch check avoids needing a restart in the common case). Not applicable to `scripts/run_benchmark.py` at all — there's no kernel to restart in a plain script; a stale import just means re-running the process.
    - **Re-running the Do-PFN cell without restarting the kernel does NOT retry cleanly.** Reproduced directly: on a fresh process, `from scripts.transformer_prediction_interface import DoPFNRegressor` correctly raises the `Optional` `ImportError`. But if the *same* import is attempted again in the *same* kernel session (e.g. the user just re-runs the cell after it failed), Python leaves the `scripts` package half-cached in `sys.modules` from the first failed attempt, and the second attempt silently "succeeds" — handing back a broken `DoPFNRegressor` class — only to fail later, uncaught, inside `dopfn.fit()`/`predict_cate()` (deep in `TabPFNBaseModel.init_model_and_get_model_config()`). The notebook's Do-PFN cell wraps the *entire* import-through-predict flow in one `try/except` so this degrades gracefully instead of crashing with a raw traceback, but the underlying fix is still: run "§0", then actually restart the kernel/runtime, then re-run all cells from the top — not just re-run the Do-PFN cell in place.
- **Metalearners**: Require `econml` + `causalml`; `hpo=True` also uses CausalPFN's pinned `FLAML[automl]==2.3.5` search (the `metalearners` extra in `pyproject.toml` / `requirements.txt`).
- **Missing models**: Notebooks use `try/except` for installs; unavailable models are skipped with a warning.

### Running & testing

- No automated test suite; validation is via end-to-end notebook execution.
- To verify setup, run the smoke-test (see Commands section above).
- Notebooks save results to CSV + PNG for easy comparison.
