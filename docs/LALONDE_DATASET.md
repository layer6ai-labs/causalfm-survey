# The Lalonde Dataset: What It Is, and Why We Fixed How We Score It

## What it is

The Lalonde dataset comes from Robert LaLonde's 1986 study of the National
Supported Work (NSW) program, a job-training intervention. Dehejia & Wahba
(1999, 2002) re-released the data in the form this repo uses, hosted at
Dehejia's NBER page. It's a standard benchmark in causal inference precisely
because it contains **two different ways to estimate the same treatment
effect**, one trustworthy and one not:

- **NSW-treated vs. NSW-control** — a genuine **randomized experiment**.
  Because assignment to treatment was random, a simple difference in mean
  `re78` (1978 earnings) between these two groups is an *unbiased* estimate
  of the true average treatment effect. No adjustment needed.
- **NSW-treated vs. PSID-controls** — an **observational** substitute.
  LaLonde's point (and the reason this dataset is famous) was to ask: if you
  only had non-experimental controls, could standard econometric methods
  still recover the right answer? PSID controls are a very different
  population from NSW participants (older, more work history, higher
  earnings generally), so a naive comparison is badly confounded by
  selection bias. This pairing (or a trimmed subset of it — see below) is
  the actual estimation task fed to every model in `Lalonde_benchmark.ipynb`
  as `(X, T, Y)`.

Features: `age`, `educ`, `black`, `hisp`, `married`, `nodegree`, `re74`,
`re75`. Treatment: `treat`. Outcome: `re78`.

## The numbers, verified directly

| Comparison | ATE |
|---|---|
| True experimental ATE (NSW-treated vs. NSW-control) | $1,794.34 |
| Naive observed diff (NSW-treated vs. PSID-controls) | -$15,204.78 |

The true effect is a modest positive number — the job-training program
helped, a bit. The naive comparison says the opposite and by a huge margin,
purely because PSID controls earn far more than NSW participants for reasons
that have nothing to do with the program. That gap **is** the selection
bias this dataset exists to test whether a causal method can correct for.

## What was wrong before, and the fix

`causal_bench/data_loader.py` originally computed `ds.ate` as the naive
diff-in-means on the *same* NSW-treated/PSID-controls data fed to every
model — i.e., it scored models against the confounded number, not the true
one. That's backwards: a method that stayed close to `-$15,204` wasn't
doing well, it was failing to adjust for confounding at all, and a method
that moved far away from it — toward the true small positive effect — was
being penalized for doing its job correctly.

`load_lalonde()` now loads the NSW-treated vs. NSW-control randomized
comparison separately and uses *that* diff-in-means as `ds.ate` — the true
benchmark to score against. The old naive number is preserved as
`ds.ate_naive_observed`, printed in the notebook for context, so the scale
of the selection-bias problem stays visible.

**Individual-level CATE still has no ground truth on this dataset** — only
the population-level ATE is checkable this way. That was true before and
remains true.

## Why this flipped the benchmark results

Scored against the corrected true ATE, DR and Debiased ML — which had
looked best under the old (wrong) scoring, because they landed close to the
naive `-$15,204` — turned out to be the *least* accurate models in the
table. They weren't correcting for the selection bias; they were mostly
reproducing it. CausalFM, previously worst, became the most accurate model
once scored correctly (and once its inputs were standardized — see
CLAUDE.md's "Models & dependencies" caveats for that separate fix). This
matches the point LaLonde's original paper was making: this exact
comparison is hard, and many standard adjustment methods fail to recover
the true effect from it.

## The overlap problem, and trimming (`variant="nsw_psid_trimmed"`)

Even scored correctly, the untrimmed comparison has a second problem: NSW-treated
and PSID-controls barely overlap on covariates *at all* — verified directly:

| | NSW-treated | PSID-controls |
|---|---|---|
| age | 25.8 | 34.9 |
| married | 19% | 87% |
| nodegree | 71% | 31% |
| re74 (1974 earnings) | $2,096 | $19,429 |
| re75 (1975 earnings) | $1,532 | $19,063 |

This is severe enough that practically every method in `Lalonde_benchmark.ipynb`
gets the *sign* of the effect wrong on the untrimmed data, not just the
magnitude — not because the methods are bad, but because most PSID-control
units have no comparable treated unit to be compared against, so there's no
covariate-space region where "controlling for X" can actually work.

`load_lalonde(variant="nsw_psid_trimmed")` — what `Lalonde_benchmark.ipynb`
uses by default — restricts PSID-controls to **common propensity-score
support**: fit `p(treat=1|X)` on the pooled sample (`causal_bench/data_loader.py`,
`_trim_to_common_support`), keep only control units whose score falls
within the range actually observed among treated units. Verified effect:

| Metric | Value |
|---|---|
| PSID-control units dropped | 1,392 of 2,490 (56%) |
| Naive observed diff, trimmed | -$5,896.60 |
| Naive observed diff, untrimmed (for comparison) | -$15,204.78 |

Trimming doesn't eliminate confounding among the units that remain — the
naive diff-in-means is still far from the true $1,794 — but it removes the
impossible-by-construction part of the problem. Verified effect on actual
model accuracy (absolute ATE error, lower is better):

| Model | Untrimmed error | Trimmed error | Notes |
|---|---|---|---|
| S-learner | $2,561 | **$1,645** | |
| Do-PFN | $7,010 | **$2,428** | now sign-correct: +$4,222 |
| CausalFM | $2,092 | $2,720 | |
| Debiased ML | $12,278 | **$4,265** | |
| T-learner | $15,262 | **$5,568** | |
| IPW | $15,262 | **$5,568** | identical to T-learner here |
| DR | $16,205 | **$5,836** | |
| X-learner | $11,305 | **$6,015** | |

Every metalearner improves substantially, Do-PFN becomes sign-correct for
the first time, and S-learner becomes the most accurate model overall.
CausalFM is the one exception (slightly worse trimmed) — plausibly because
it's zero-shot on a smaller, differently-shaped context (896 fewer training
units) rather than being fit fresh to it like the metalearners are.

The untrimmed `variant="nsw_psid"` (the default if you don't pass `variant=`)
is still available if you want the original, maximally-hard comparison.

## Verified reference run (Colab, GPU) — all 9 models

The table above only had 6 metalearners + Do-PFN + CausalFM (CausalPFN can't
run locally on Apple Silicon — see CLAUDE.md). Here's a full run of all 9
models on `nsw_psid_trimmed`, from Colab with a GPU runtime, matching the
"Reference output" section in `Lalonde_benchmark.ipynb` itself:

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

![Reference plot: ATE error and runtime, foundation models vs. metalearners](../notebooks/assets/lalonde_reference_output_colab.png)

S-learner is the most accurate model in this run, and the only one within
the same order of magnitude as the true effect. Do-PFN and CausalFM both
beat most metalearners; CausalPFN lands in between. Every metalearner
except S-learner underestimates by a similar amount (roughly -$3,800 to
-$4,200) — consistent with all of them reacting to the same remaining
confounding in the trimmed sample rather than failing independently.

## Practical takeaway

If you extend this benchmark to other real-world datasets: check whether
your "ground truth" was actually measured independently (e.g., from a
randomized experiment, like here) or whether it's silently derived from the
same confounded data your models are being scored on. The latter isn't a
ground truth at all — it's the bias you're trying to measure.

## Two Lalonde benchmarks in this repo: real NBER data vs. RealCause semi-synthetic

Everything above describes `load_lalonde()` — the real NBER Lalonde data, sections
3–7 of `Lalonde_benchmark.ipynb`. If you compare its numbers directly against
CausalPFN's own published Lalonde results (paper: arXiv:2506.07918; repo:
`github.com/vdblm/CausalPFN`), **they won't match, and that's expected** — the two
setups are testing genuinely different things. Sections 8–10 of the notebook add a
second Lalonde benchmark, `load_lalonde_realcause()`, that replicates CausalPFN's
actual methodology so its numbers *are* directly comparable to their Table 1.

### Why the two disagree

CausalPFN's paper does not score Lalonde on the real NBER data at all. It uses
**RealCause** (Neal, Huang & Raghupathi, 2020, arXiv:2011.15007): a method that fits
a generative model to a real dataset's covariate/treatment/outcome distribution, then
*simulates* new potential outcomes `y0`/`y1` from that fitted model. The real
covariates and treatment assignment are preserved, but the outcomes — and therefore
the individual treatment effect `ite = y1 - y0` — are simulated, not observed. This
is what makes individual-level CATE ground truth possible at all, which is
structurally impossible on real data (see "What was wrong before, and the fix"
above — real data only ever gives you one of `y0`/`y1` per unit, never both).

| | `load_lalonde()` (sections 3–7) | `load_lalonde_realcause()` (sections 8–10) |
|---|---|---|
| Data | Real NBER data: NSW-treated vs. PSID-controls | RealCause semi-synthetic realizations of real Lalonde covariates (PSID and CPS cohorts) |
| Outcomes | Real, observed `re78` | Simulated `y0`/`y1`, drawn from a generative model fit to the real data |
| Ground truth | Population-level ATE only, from a *separate* randomized experiment (NSW vs. NSW-control) | Known **per-unit** ITE for every unit (the `ite` column) |
| CATE ground truth | None — impossible on real data | Yes — this is what makes PEHE computable at all |
| Repetition | Single fixed dataset | Averaged over 10 repeated realizations (of 100 available) |
| Reported metric | ATE abs/rel error only | Mean PEHE ± SEM and mean ATE relative error ± SEM, across realizations |
| Difficulty | Real, severe selection bias + poor covariate overlap (see above) | Whatever bias/overlap the fitted generative model reproduces — a different, and on this data an easier, problem |

Neither is "more correct" than the other — they're standard, complementary ways to
evaluate causal estimators on real-world-shaped data. The real-data benchmark tests
whether a method can recover a *known-true, independently-measured* population ATE
under genuine, unmodified confounding. The RealCause benchmark tests whether a method
can recover *individual-level* effects when the ground truth is simulated but the
covariate/treatment structure is real — the standard way papers like CausalPFN's
report CATE accuracy at all, since real data can never supply that ground truth.

### What "realizations" means

RealCause fits one generative model per cohort (PSID or CPS) to the real
covariate/treatment/outcome distribution. Each "realization" is one independent
*sample* drawn from that fitted model — the same real covariates and treatment
assignment every time, but a different simulated draw of `y0`/`y1` (and therefore a
different `ite`) each time. CausalPFN's repo ships 100 such pre-computed realizations
per cohort as flat CSVs (`benchmarks/realcause_datasets/lalonde_{cohort}_sample{i}.csv`,
`i=0..99`) — this is the same role IHDP's 100 fixed realizations play in that
benchmark: it turns one noisy point estimate into a distribution over draws, so a
single lucky or unlucky simulated dataset doesn't dominate the reported number. The
paper's Table 1 uses "the first 10 realizations" of each cohort, averaged as
mean ± SEM; `load_lalonde_realcause(..., n_realizations=10)` (the default) matches
this exactly.

### Where the data comes from, and how it's scored

`causal_bench/data_loader.py::load_lalonde_realcause()` downloads (and locally
caches, like `load_lalonde()` does) the CSVs directly from `vdblm/CausalPFN`'s own
repo — the literal artifacts their reported numbers were computed from — rather than
re-running RealCause's own generative-model-fitting pipeline from scratch. Train/test
split and ATE ground truth replicate CausalPFN's own
`benchmarks/realcause.py::RealCauseDataset._get_data` exactly, verified directly
against that source:

- Per realization: `np.random.default_rng(seed + i)` (default `seed=42`), permute
  all rows, first 90% → train (fit the CATE model on this), held-out 10% → test
  (score PEHE only on these, against their true `ite`).
- `ate_true` is `ite.mean()` over **all** rows (train+test combined) — CausalPFN's
  own evaluation notebook (`notebooks/causal_effect_full.ipynb`) fits a *separate*
  model instance on the full realization data for the ATE metric, rather than
  reusing the CATE model's train-only fit. `Lalonde_benchmark.ipynb`'s section 9
  does the same: two fresh model instances per realization per method, one fit on
  the 90% train split (for PEHE), one fit on the full data (for ATE).

### Reference numbers (CausalPFN, from the paper's Table 1)

| Cohort | Mean PEHE | Mean ATE relative error |
|---|---|---|
| Lalonde PSID | 13.98 ± 0.43 | 0.20 ± 0.03 |
| Lalonde CPS | 8.83 ± 0.04 | 0.08 ± 0.02 |

If your own CausalPFN run through `Lalonde_benchmark.ipynb`'s section 10 lands in
this range, your RealCause setup is working correctly — the same sanity-check role
the "Verified reference run" table above plays for the real-data benchmark.
