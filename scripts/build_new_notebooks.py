"""
Builds notebooks using nbformat.
Run: python3 scripts/build_new_notebooks.py

Generates:
- Lalonde_benchmark.ipynb (all 6 metalearners + all 3 foundation models on real Lalonde data)
- RealCause_benchmark.ipynb (same 9 models on RealCause semi-synthetic Lalonde realizations,
  replicating CausalPFN's own paper methodology -- see docs/LALONDE_DATASET.md)
"""
import nbformat as nbf
import os

OUT_DIR = "notebooks"
os.makedirs(OUT_DIR, exist_ok=True)

REPO_SLUG = "layer6ai-labs/causalfm-survey"


def colab_badge(notebook_path):
    url = f"https://colab.research.google.com/github/{REPO_SLUG}/blob/main/{notebook_path}"
    return f'[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})'


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


def save(nb, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)


def env_and_setup_cells():
    """Sections '1. One-time environment check' + '2. Setup': torch-version guard for
    Do-PFN, causal_bench import/clone, econml+causalpfn install, Do-PFN/CausalFM clone
    and extra deps. Identical prerequisites for every notebook that uses causal_bench's
    foundation-model wrappers, so shared here rather than duplicated per notebook."""
    return [
        md("""## 1. One-time environment check — needed for Do-PFN

Do-PFN's model code depends on an internal PyTorch name removed in
`torch>=2.10`. **This must run first, before any other cell** — in
particular before `causal_bench` gets imported below, since it transitively
imports `torch` itself (`wrap_dopfn.py`), which would defeat the whole
point of this cell if it ran second.

- **On Colab**: pins `torch==2.9.1` (verified compatible with Do-PFN) via a
  plain `pip install` -- **can take up to a minute or so** (torch is a large
  package; network speed is the bottleneck, not this notebook). As long as
  this is the first cell you run, **no restart needed** -- Colab ships torch
  preinstalled but never auto-imports it, so nothing has loaded the
  incompatible version into memory yet; the freshly installed one is just
  what later cells pick up. (If you'd already run other cells before this
  one, torch may already be in memory -- restart the runtime and re-run from
  the top.)
- **Locally (this repo's `uv` venv)**: `pip` isn't available inside the
  notebook, so this only detects the problem. Fix in a terminal:
  `uv pip install "torch==2.9.1"`, then restart the kernel.
- Not planning to run Do-PFN? Skip — CausalPFN and CausalFM work fine on any
  recent torch."""),

        code("""import sys, os, subprocess, importlib.metadata

IN_COLAB = "google.colab" in sys.modules
TORCH_PIN = "2.9.1"  # last version verified compatible with Do-PFN (torch>=2.10 breaks it)

def _restart_colab_kernel():
    # google.colab.kernel.restart() is Colab's own restart API -- the same
    # one "Runtime > Restart session" uses. A raw os.kill(pid, SIGKILL) also
    # restarts the process, but Colab's frontend doesn't recognize it as an
    # intentional restart and reports "session crashed" instead (verified
    # directly). Either way execution can't resume on its own afterward, and
    # this API itself isn't always reliable (also verified directly). Used
    # below only for the (rarer) stale-causal_bench-import case -- the torch
    # check itself avoids needing a restart at all, see next cell.
    from IPython.display import Javascript, display
    display(Javascript("google.colab.kernel.restart()"))

def _torch_needs_downgrade():
    try:
        v = importlib.metadata.version("torch")  # reads metadata, doesn't import torch
    except importlib.metadata.PackageNotFoundError:
        return False  # not installed yet -- nothing to fix here
    major, minor = (int(p) for p in v.split("+")[0].split(".")[:2])
    return (major, minor) >= (2, 10)

if not _torch_needs_downgrade():
    print("OK -- torch version is compatible with Do-PFN (or not installed yet).")
elif IN_COLAB:
    print(f"torch >= 2.10 detected -- installing torch=={TORCH_PIN} (can take a minute)...")
    get_ipython().system(f"pip install -q torch=={TORCH_PIN}")
    print("OK -- done. If this was the first cell you ran, no restart needed -- "
          "continue to the next cell.")
else:
    print("torch >= 2.10 detected -- Do-PFN will fail to import.")
    print('Fix, in a terminal (not this notebook -- local uv venv has no pip):')
    print(f'    uv pip install "torch=={TORCH_PIN}"')
    print("then restart this notebook's kernel and re-run from the top.")"""),

        md("""## 2. Setup

If this is a fresh runtime, just run this cell normally. If you're
re-running the notebook in a runtime that already ran it before (e.g. after
a repo update), this cell detects that `causal_bench` was already imported
earlier in the session and restarts the kernel the same way the cell above
does -- Python caches imported modules in memory, so a `git pull` alone
can't make an already-running session pick up code changes; only a restart
can."""),

        code("""# Python caches imported modules -- if causal_bench is already loaded, no
# amount of re-cloning/pulling below will change what's in memory. Restart
# is the only fix (Colab: automatic; locally: this cell raises instead).
if "causal_bench" in sys.modules:
    msg = ("causal_bench was already imported earlier in this session -- a repo "
           "update (e.g. git pull below) can't refresh an already-loaded module, "
           "so continuing would risk confusing errors (like an AttributeError on "
           "a field that exists in the current code but not in memory).")
    if IN_COLAB:
        print(msg + " Restarting the session now -- if nothing happens within "
              "a few seconds (this isn't always reliable), use Runtime > "
              "Restart session yourself. Either way, re-run this cell once "
              "it reconnects, then continue from the top.")
        _restart_colab_kernel()
    else:
        raise RuntimeError(msg + " Restart the kernel (Kernel/Restart), then "
                            "run all cells again from the top.")

# ── FOR COLAB ONLY: set your GitHub token if the repo is private ──────────────
# Create one at: github.com/settings/tokens  (scope: repo → read)
# Leave as "" if the repo is public.
GITHUB_TOKEN = ""
# ──────────────────────────────────────────────────────────────────────────────

REPO_SLUG = "layer6ai-labs/causalfm-survey"
REPO_DIR  = "causalfm-survey"

if IN_COLAB:
    if not os.path.exists(REPO_DIR):
        if GITHUB_TOKEN:
            clone_url = f"https://{GITHUB_TOKEN}@github.com/{REPO_SLUG}.git"
        else:
            clone_url = f"https://github.com/{REPO_SLUG}.git"
        result = subprocess.run(["git", "clone", clone_url], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "git clone failed — the repo is likely private.\\n"
                "Fix: set GITHUB_TOKEN above (github.com/settings/tokens, scope: repo→read).\\n"
                f"Error: {result.stderr.strip()}"
            )
    else:
        # Shouldn't normally get here -- the causal_bench check above already
        # forces a restart before a stale clone could be reused. Pull anyway
        # in case the clone exists but causal_bench was never imported yet.
        result = subprocess.run(["git", "-C", REPO_DIR, "pull"], capture_output=True, text=True)
        print(result.stdout.strip() or result.stderr.strip())
    sys.path.insert(0, REPO_DIR)
else:
    sys.path.insert(0, os.path.abspath(".."))

import causal_bench
print("causal_bench imported from:", causal_bench.__file__)"""),

        code("""# "pandas<2.4" pin: econml has no pandas upper bound, so pip's resolver
# otherwise grabs the newest pandas (3.x) -- which conflicts with Colab's
# preinstalled google-colab/cudf-cu12/dask-cudf-cu12 (all require pandas<2.4).
%pip install -q econml causalpfn "pandas<2.4"
import numpy as np, pandas as pd, time, warnings
warnings.filterwarnings('ignore')

import torch
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")"""),

        md("""### Do-PFN and CausalFM setup

Neither is on PyPI, so clone if missing and install just the extra deps each
actually needs — **not** their bundled `requirements.txt` files, which are
frozen dev/CUDA snapshots that fail to install as-is (`catboost==1.1.1` has
no wheel for recent Python; CausalFM's snapshot has Linux/CUDA-only pins).
CausalFM additionally needs its pretrained checkpoint, which ships inside the
cloned repo at `checkpoints/checkpoints_standard/best_model.pth`."""),

        code("""DOPFN_DIR = "Do-PFN"
CAUSALFM_DIR = "CausalFM-toolkit"

if not os.path.exists(DOPFN_DIR):
    print(f"Cloning Do-PFN...")
    subprocess.run(["git", "clone", "https://github.com/jr2021/Do-PFN.git"], check=True)
sys.path.insert(0, os.path.abspath(DOPFN_DIR))

if not os.path.exists(CAUSALFM_DIR):
    print(f"Cloning CausalFM-toolkit...")
    subprocess.run(["git", "clone", "https://github.com/yccm/CausalFM-toolkit.git"], check=True)
sys.path.insert(0, os.path.abspath(CAUSALFM_DIR))

if IN_COLAB:
    get_ipython().system('pip install -q networkx tqdm einops "tabpfn==2.0.9" tensorboard')
# Locally: uv pip install networkx tqdm einops "tabpfn==2.0.9" tensorboard"""),
    ]


# ============================================================================
# Lalonde Benchmark (All Models, real NBER data)
# ============================================================================
nb = nbf.v4.new_notebook()
nb.cells = [
    md(f"""# Lalonde Benchmark: Foundation Models vs. Metalearners

{colab_badge('notebooks/Lalonde_benchmark.ipynb')}

**Compare all foundation models against all traditional metalearners on the Lalonde dataset.**

This notebook runs:
- 3 foundation models (CausalPFN, Do-PFN, CausalFM)
- 6 metalearners (S-learner, T-learner, X-learner, Debiased ML, IPW, DR)

on the Lalonde real-world causal inference benchmark and produces a comparison table. Each
model runs independently and failures don't block the rest — unavailable or erroring
models are reported and skipped.

This uses the real NBER Lalonde data, which only has a true experimental **ATE** to check
against — no individual-level CATE ground truth is possible on real data. For a companion
benchmark that replicates CausalPFN's own paper methodology (RealCause semi-synthetic
realizations, individual-level CATE ground truth, directly comparable to their Table 1), see
`RealCause_benchmark.ipynb`. See `docs/LALONDE_DATASET.md` for why the two aren't directly
comparable to each other."""),

    *env_and_setup_cells(),

    md("""## 3. Load Lalonde Dataset

Uses `variant="nsw_psid_trimmed"`: NSW-treated vs. PSID-controls restricted
to common propensity-score support (see `docs/LALONDE_DATASET.md`). The
untrimmed `"nsw_psid"` pairing has almost no covariate overlap between
groups at all, which defeats every method here by construction, not just
the weaker ones -- trimming gives a real, if still hard, estimation task."""),

    code("""from causal_bench import load_lalonde, evaluate_cate

print("Loading Lalonde dataset (common-support trimmed)...")
ds = load_lalonde("nsw_psid_trimmed")
print(f"  n={len(ds.Y)}, X.shape={ds.X.shape}  "
      f"({ds.meta['n_dropped_by_trimming']} PSID-control units dropped for lacking overlap)")
print(f"  True experimental ATE (NSW-treated vs. NSW-control, scored against): {ds.ate:.3f}")
print(f"  Naive observed diff on X/T/Y itself (still confounded, just less "
      f"extremely so post-trimming): {ds.ate_naive_observed:.3f}")

train_idx, test_idx = ds.train_test_split(0.7, seed=0)
X_train, X_test = ds.X[train_idx], ds.X[test_idx]
T_train, Y_train = ds.T[train_idx], ds.Y[train_idx]

print(f"  train: n={len(train_idx)}, test: n={len(test_idx)}")"""),

    md("## 4. Foundation Model Availability"),

    code("""from causal_bench import CausalPFNWrapper, DoPFNWrapper, CausalFMWrapper

FOUNDATION_MODELS = {
    "CausalPFN": CausalPFNWrapper,
    "Do-PFN":    DoPFNWrapper,
    "CausalFM":  CausalFMWrapper,
}

print("Foundation model availability:")
for name, cls in FOUNDATION_MODELS.items():
    print(f"  {'✓' if cls.is_available() else '✗'}  {name}")
print(f"\\ndevice: {device}")"""),

    md("## 5. Run All Models"),

    code("""from causal_bench import (
    SLearnerWrapper, TLearnerWrapper, XLearnerWrapper,
    DebiasedMLWrapper, IPWWrapper, DRWrapper,
)

METALEARNERS = {
    "S-learner":          SLearnerWrapper,
    "T-learner":          TLearnerWrapper,
    "X-learner":          XLearnerWrapper,
    "Debiased ML":        DebiasedMLWrapper,
    "IPW":                IPWWrapper,
    "DR (Doubly Robust)": DRWrapper,
}

results = []

# ── Run all metalearners ──────────────────────────────────────────────────────
print("=" * 70)
print("METALEARNERS")
print("=" * 70)
for name, model_cls in METALEARNERS.items():
    if not model_cls.is_available():
        print(f"  {name:25s}: SKIPPED (not installed)")
        continue
    try:
        t0 = time.time()
        model = model_cls()
        model.fit(X_train, T_train, Y_train)
        tau_hat, _, _ = model.predict(X_test)
        runtime = time.time() - t0
        ate_hat = float(np.mean(tau_hat))
        results.append({
            "model": name,
            "ate_hat": ate_hat, "ate_true": ds.ate,
            "ate_abs_error": abs(ate_hat - ds.ate),
            "ate_rel_error": abs(ate_hat - ds.ate) / (abs(ds.ate) + 1e-8),
            "runtime_s": runtime,
        })
        print(f"  {name:25s}: ATE_error={abs(ate_hat-ds.ate):.1f}  runtime={runtime:.2f}s")
    except Exception as e:
        print(f"  {name:25s}: ERROR: {e}")

# ── Run all foundation models ─────────────────────────────────────────────────
print("\\n" + "=" * 70)
print("FOUNDATION MODELS")
print("=" * 70)
for fm_name, model_cls in FOUNDATION_MODELS.items():
    if not model_cls.is_available():
        print(f"  {fm_name:25s}: SKIPPED (not available in this environment)")
        continue
    try:
        t0 = time.time()

        if fm_name == "CausalFM":
            checkpoint_path = "CausalFM-toolkit/checkpoints/checkpoints_standard/best_model.pth"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            model = model_cls(checkpoint_path=checkpoint_path, device=device)
        else:
            model = model_cls(device=device)

        model.fit(X_train, T_train, Y_train)
        tau_hat, _, _ = model.predict(X_test)
        runtime = time.time() - t0
        ate_hat = float(np.mean(tau_hat))

        results.append({
            "model": fm_name + " (Foundation)",
            "ate_hat": ate_hat, "ate_true": ds.ate,
            "ate_abs_error": abs(ate_hat - ds.ate),
            "ate_rel_error": abs(ate_hat - ds.ate) / (abs(ds.ate) + 1e-8),
            "runtime_s": runtime,
        })
        print(f"  {fm_name:25s}: ATE_error={abs(ate_hat-ds.ate):.1f}  runtime={runtime:.2f}s")
    except Exception as e:
        print(f"  {fm_name:25s}: ERROR: {e}")

print("\\n" + "=" * 70)"""),

    md("## 6. Results Table"),

    code("""df = pd.DataFrame(results)
df_sorted = df.sort_values("ate_abs_error")

print("\\nResults (sorted by ATE absolute error):")
print(df_sorted[["model", "ate_hat", "ate_true", "ate_abs_error", "ate_rel_error", "runtime_s"]].to_string(index=False))

df_sorted.to_csv("lalonde_benchmark.csv", index=False)
print("\\nSaved to lalonde_benchmark.csv")"""),

    md("## 7. Visualization"),

    code("""import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FOUNDATION_COLOR, METALEARNER_COLOR = '#1f77b4', '#ff7f0e'
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ATE error comparison
ax = axes[0]
df_plot = df_sorted.copy()
colors = [FOUNDATION_COLOR if 'Foundation' in name else METALEARNER_COLOR for name in df_plot['model']]
ax.barh(range(len(df_plot)), df_plot['ate_abs_error'], color=colors)
ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot['model'])
ax.set_xlabel('Absolute ATE Error')
ax.set_title('ATE Error: Foundation vs. Metalearners')
ax.axvline(ds.ate, color='red', linestyle='--', alpha=0.5, label='True ATE')

# Runtime comparison
ax = axes[1]
ax.barh(range(len(df_plot)), df_plot['runtime_s'], color=colors)
ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot['model'])
ax.set_xlabel('Runtime (seconds)')
ax.set_title('Runtime: Fit + Predict on Test Set')

# Explicit legend handles -- a bare `plt.legend([...])` call grabs whatever
# single bar-container artist happens to exist on the current axes and
# mislabels it (verified directly: it picked up the first bar's actual
# color, not a color matching the given label).
legend_handles = [
    mpatches.Patch(color=FOUNDATION_COLOR, label='Foundation'),
    mpatches.Patch(color=METALEARNER_COLOR, label='Metalearner'),
]
axes[1].legend(handles=legend_handles, loc='lower right')
plt.tight_layout()
plt.savefig("lalonde_benchmark.png", dpi=150)
plt.show()
print("Saved to lalonde_benchmark.png")"""),

    md("""## Reference output — verified successful run (Colab, GPU runtime)

If your numbers look roughly like this, your setup is correct. The
train/test split is seeded (`seed=0`), so the split itself is identical
every run — differences beyond what's shown here come from the usual
sources of nondeterminism in the models (GPU non-determinism, library
version drift, CPU vs. GPU execution), not from your setup being wrong.

**Which Lalonde data this is**: `load_lalonde("nsw_psid_trimmed")` — NSW's
185 randomly-assigned job-training participants (`treat=1`) as the treated
group, compared against a *subset* of 2,490 PSID survey respondents
(`treat=0`) as the control group. The full PSID sample is restricted to
those with propensity scores overlapping the treated group (common-support
trimming — see `docs/LALONDE_DATASET.md`), since the untrimmed pairing has
almost no covariate overlap at all (PSID respondents are on average 9 years
older, mostly married vs. mostly not, and earn ~9x more pre-treatment) and
defeats every method here by construction rather than by genuine
difficulty. `ds.ate` (the row all models are scored against) is **not**
computed from this treated/control pairing at all — it's the true
experimental ATE (**$1,794.34**) from a separate randomized comparison
(NSW-treated vs. NSW's own experimental control group), the standard
literature benchmark for this program.

| Model | ATE_hat | ATE_true | ATE_abs_error | ATE_rel_error | Runtime (s) |
|---|---|---|---|---|---|
| S-learner | $150 | $1,794 | $1,645 | 0.92 | 0.57 |
| Do-PFN (Foundation) | $4,222 | $1,794 | $2,428 | 1.35 | 4.97 |
| CausalFM (Foundation) | -$926 | $1,794 | $2,720 | 1.52 | 0.32 |
| CausalPFN (Foundation) | -$2,553 | $1,794 | $4,347 | 2.42 | 5.51 |
| Debiased ML | -$3,079 | $1,794 | $4,873 | 2.72 | 0.88 |
| T-learner | -$3,774 | $1,794 | $5,568 | 3.10 | 0.57 |
| IPW | -$3,774 | $1,794 | $5,568 | 3.10 | 0.70 |
| DR (Doubly Robust) | -$4,014 | $1,794 | $5,808 | 3.24 | 0.68 |
| X-learner | -$4,176 | $1,794 | $5,970 | 3.33 | 0.97 |

![Reference plot: ATE error and runtime, foundation models vs. metalearners](assets/lalonde_reference_output_colab.png)

**S-learner is most accurate here**, and the only model within the same
order of magnitude as the true effect. All three foundation models cluster
in the middle of the pack — **Do-PFN** and **CausalFM** actually beat most
metalearners, while **CausalPFN** lands in between. Every metalearner
except S-learner underestimates by a similar amount (roughly -$3,800 to
-$4,200) — consistent with them all reacting to the same remaining
confounding in the trimmed sample, rather than each failing independently.
Runtime-wise, CausalPFN and Do-PFN are markedly slower than everything else
here (GPU inference + framework overhead on a dataset this small dwarfs the
actual computation), while CausalFM and every metalearner finish in under a
second."""),

    md("""## Interpretation

**ATE Error** (lower is better):
- Measures how well each model estimates the average treatment effect
- Foundation models learn from large training priors; metalearners fit to this specific data

**Runtime** (lower is better):
- Foundation models: fast (forward pass only, no retraining)
- Metalearners: slower (train separate models for each group)

**Real data, real ground truth for ATE — but not for CATE**: `ds.ate` is the true experimental
ATE from the randomized NSW-treated vs. NSW-control comparison (not computed from the
confounded X/T/Y models actually see), so ATE error here is a genuine accuracy measure, not
just distance from a naive number. Individual-level CATE still has no ground truth on real
data — only ATE is checkable.

**Selection bias is still real, even after trimming**: the naive diff-in-means on the X/T/Y
models see (`ds.ate_naive_observed`, printed above) remains far from the true ATE even on
the common-support-trimmed sample — trimming removes PSID-control units with *no* comparable
treated unit at all (an impossible-by-construction problem), it doesn't eliminate confounding
among the units that remain. Recovering the true effect is still a real test of confounder
adjustment; see `docs/LALONDE_DATASET.md` for the untrimmed comparison, where the covariate
overlap is so poor that practically every method here gets the *sign* of the effect wrong,
not just the magnitude."""),
]

save(nb, "Lalonde_benchmark.ipynb")


# ============================================================================
# RealCause Lalonde Benchmark (All Models, matches CausalPFN's own methodology)
# ============================================================================
nb = nbf.v4.new_notebook()
nb.cells = [
    md(f"""# RealCause Lalonde Benchmark: Foundation Models vs. Metalearners

{colab_badge('notebooks/RealCause_benchmark.ipynb')}

**Replicates CausalPFN's own Lalonde evaluation methodology**, so results here are directly
comparable to their paper's Table 1 (arXiv:2506.07918; repo: `github.com/vdblm/CausalPFN`).

`Lalonde_benchmark.ipynb` scores models on the real NBER Lalonde data, which only has a true
experimental **ATE** to check against — individual-level CATE ground truth is structurally
impossible on real data (you only ever observe one of `y0`/`y1` per unit). CausalPFN's own
paper doesn't score Lalonde on real data at all: it uses **RealCause** (Neal, Huang &
Raghupathi 2020, arXiv:2011.15007), which fits a generative model to the real Lalonde
covariates/treatment/outcome distribution, then *simulates* new potential outcomes `y0`/`y1`
from it. The real covariates and treatment assignment are preserved, but because both
potential outcomes are now simulated, every unit has a known individual treatment effect
(`ite = y1 - y0`) — which is what makes PEHE computable at all here.

This notebook runs:
- 3 foundation models (CausalPFN, Do-PFN, CausalFM)
- 6 metalearners (S-learner, T-learner, X-learner, Debiased ML, IPW, DR)

on 10 RealCause realizations each of the PSID and CPS cohorts, and reports mean PEHE / mean ATE
relative error ± SEM per model/cohort. Each model runs independently and failures don't block
the rest — unavailable or erroring models are reported and skipped. See
`docs/LALONDE_DATASET.md` for the full write-up of why this and `Lalonde_benchmark.ipynb`
disagree and shouldn't be compared to each other directly."""),

    *env_and_setup_cells(),

    md("## 3. Model Registry"),

    code("""from causal_bench import CausalPFNWrapper, DoPFNWrapper, CausalFMWrapper
from causal_bench import (
    SLearnerWrapper, TLearnerWrapper, XLearnerWrapper,
    DebiasedMLWrapper, IPWWrapper, DRWrapper,
)

FOUNDATION_MODELS = {
    "CausalPFN": CausalPFNWrapper,
    "Do-PFN":    DoPFNWrapper,
    "CausalFM":  CausalFMWrapper,
}
METALEARNERS = {
    "S-learner":          SLearnerWrapper,
    "T-learner":          TLearnerWrapper,
    "X-learner":          XLearnerWrapper,
    "Debiased ML":        DebiasedMLWrapper,
    "IPW":                IPWWrapper,
    "DR (Doubly Robust)": DRWrapper,
}
ALL_MODELS = {**METALEARNERS, **{f"{name} (Foundation)": cls for name, cls in FOUNDATION_MODELS.items()}}

print("Foundation model availability:")
for name, cls in FOUNDATION_MODELS.items():
    print(f"  {'✓' if cls.is_available() else '✗'}  {name}")
print(f"\\ndevice: {device}")"""),

    md("""## 4. Load RealCause Lalonde Realizations

**What "realizations" means**: RealCause fits one generative model per cohort (PSID or CPS) to
the real covariates/treatment/outcome distribution. Each "realization" is one independent
*sample* drawn from that fitted model — same real covariates and treatment assignment every
time, but a different simulated draw of potential outcomes `y0`/`y1` (and therefore a different
individual treatment effect `ite = y1 - y0`) each time. 100 such draws are pre-computed and
checked into CausalPFN's own repo as flat CSVs; the paper uses "the first 10" and reports
mean ± SEM across them — same reason IHDP ships 100 fixed realizations: it turns one noisy
point estimate into a distribution, so a lucky/unlucky draw doesn't dominate the number.

This downloads those same CSVs directly from `vdblm/CausalPFN` (cached locally after first run)
and reproduces CausalPFN's own evaluation loop exactly (verified directly against
`benchmarks/realcause.py` and `notebooks/causal_effect_full.ipynb` in their repo): per
realization, fit each model on a 90% train split and score PEHE on the held-out 10% test split
against the true `ite`; separately, fit a *fresh* instance of each model on the *full*
realization data (matching their own separate CATE/ATE fits) and score ATE relative error
against `ite.mean()` over all units. See `docs/LALONDE_DATASET.md` for the full writeup.

**Reference numbers from the paper's Table 1** (CausalPFN only, for sanity-checking your own
run against): Lalonde PSID — mean PEHE 13.98 ± 0.43, mean ATE relative error 0.20 ± 0.03.
Lalonde CPS — mean PEHE 8.83 ± 0.04, mean ATE relative error 0.08 ± 0.02.

**Heavier than a typical run**: CPS has ~16,000 rows per realization (PSID ~2,700), and the
default runs 2 cohorts × 10 realizations × 2 fits (CATE + ATE) per model — expect this to take
a while, especially for the foundation models. Best run on Colab with a GPU runtime. Lower
`REALCAUSE_N_REALIZATIONS` (e.g. 1-2) for a quick smoke test before committing to a full run."""),

    code("""from causal_bench import load_lalonde_realcause, evaluate_cate

REALCAUSE_N_REALIZATIONS = 10  # matches the paper's "first 10 realizations"
REALCAUSE_COHORTS = ["psid", "cps"]

realcause_data = {}
for cohort in REALCAUSE_COHORTS:
    print(f"Loading RealCause Lalonde {cohort.upper()} ({REALCAUSE_N_REALIZATIONS} realizations)...")
    reals = load_lalonde_realcause(cohort, n_realizations=REALCAUSE_N_REALIZATIONS)
    realcause_data[cohort] = reals
    print(f"  n_samples/realization={reals[0].meta['n_samples']}  "
          f"(train={reals[0].meta['n_train']}, test={reals[0].meta['n_test']})")"""),

    md("## 5. Run All Models on Every RealCause Realization"),

    code("""def make_model(model_name, model_cls):
    if model_name == "CausalFM":
        checkpoint_path = "CausalFM-toolkit/checkpoints/checkpoints_standard/best_model.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return model_cls(checkpoint_path=checkpoint_path, device=device)
    if model_name in FOUNDATION_MODELS:
        return model_cls(device=device)
    return model_cls()


realcause_results = []
print("=" * 70)
print("REALCAUSE-BASED BENCHMARK")
print("=" * 70)
for display_name, model_cls in ALL_MODELS.items():
    base_name = display_name.replace(" (Foundation)", "")
    if not model_cls.is_available():
        print(f"  {display_name:25s}: SKIPPED (not available in this environment)")
        continue

    for cohort in REALCAUSE_COHORTS:
        n_ok, n_failed = 0, 0
        for r in realcause_data[cohort]:
            try:
                t0 = time.time()
                cate_model = make_model(base_name, model_cls)
                cate_model.fit(r.X_train, r.T_train, r.Y_train)
                tau_hat_test, _, _ = cate_model.predict(r.X_test)
                cate_runtime = time.time() - t0

                t0 = time.time()
                ate_model = make_model(base_name, model_cls)
                ate_model.fit(r.X_full, r.T_full, r.Y_full)
                if hasattr(ate_model, "estimate_ate"):
                    ate_hat = ate_model.estimate_ate(r.X_full, r.T_full, r.Y_full)
                else:
                    tau_hat_full, _, _ = ate_model.predict(r.X_full)
                    ate_hat = float(np.mean(tau_hat_full))
                ate_runtime = time.time() - t0

                m = evaluate_cate(
                    tau_hat_test, r.tau_true_test,
                    ate_hat=ate_hat, ate_true=r.ate_true,
                    runtime_s=cate_runtime + ate_runtime,
                )
                m.update(model=display_name, cohort=cohort, realization=r.realization)
                realcause_results.append(m)
                n_ok += 1
            except Exception as e:
                n_failed += 1
                if n_failed == 1:
                    print(f"  {display_name:25s} [{cohort}] realization {r.realization}: ERROR: {e}")
        print(f"  {display_name:25s} [{cohort}]: {n_ok}/{len(realcause_data[cohort])} realizations OK"
              + (f", {n_failed} failed" if n_failed else ""))
print("\\n" + "=" * 70)"""),

    md("## 6. RealCause Results Table"),

    code("""realcause_df = pd.DataFrame(realcause_results)
realcause_df.to_csv("lalonde_realcause_benchmark.csv", index=False)

summary = (
    realcause_df.groupby(["model", "cohort"])[["pehe", "ate_rel_error"]]
    .agg(["mean", "sem"])
)

table_rows = []
for (model_name, cohort), row in summary.iterrows():
    table_rows.append({
        "model": model_name,
        "cohort": cohort,
        "pehe": f"{row[('pehe', 'mean')]:.2f} \\u00b1 {row[('pehe', 'sem')]:.2f}",
        "ate_rel_error": f"{row[('ate_rel_error', 'mean')]:.2f} \\u00b1 {row[('ate_rel_error', 'sem')]:.2f}",
    })
realcause_summary_df = pd.DataFrame(table_rows).sort_values(["cohort", "model"])

print("RealCause-based Lalonde results (mean \\u00b1 SEM across "
      f"{REALCAUSE_N_REALIZATIONS} realizations) -- compare directly to the paper's Table 1:")
print(realcause_summary_df.to_string(index=False))
print("\\nSaved per-realization rows to lalonde_realcause_benchmark.csv")"""),
]

save(nb, "RealCause_benchmark.ipynb")

print("\nAll notebooks built successfully!")
print(f"Generated: {os.path.join(OUT_DIR, 'Lalonde_benchmark.ipynb')}")
