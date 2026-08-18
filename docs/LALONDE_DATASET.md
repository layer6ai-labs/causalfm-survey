# The Lalonde Dataset: Which Version We Use, and Why

## Summary

This repo's benchmark, `RealCause_with_hpo_benchmark.ipynb`, evaluates models on
**RealCause** — a semi-synthetic version of the classic Lalonde dataset that keeps
the real covariates and treatment assignment but *simulates* the outcomes with a
fitted generative model, so that a true, individual-level CATE exists to score
against. That's the whole reason we use it: the real Lalonde data can supply a
true population-level *ATE*, but it can never supply an individual-level CATE (
the actual quantity this survey is about ) because in reality you only ever
observe one of a unit's two potential outcomes, never both.

The rest of this page explains what the real Lalonde data is and why it isn't
enough on its own (§1), then how RealCause manufactures a CATE ground truth
instead — who built it, how the generative model works, and how this repo uses
it (§2).

## §1. The real Lalonde data, and why it's not enough on its own

The Lalonde dataset comes from Robert LaLonde's 1986 study of the National
Supported Work (NSW) program, a job-training intervention, re-released by
Dehejia & Wahba (1999, 2002) in the form used here. It contains two ways to
estimate the same treatment effect:

- **NSW-treated vs. NSW-control** — a genuine randomized experiment. Because
  assignment was random, a simple difference in mean 1978 earnings (`re78`)
  between these two groups is an unbiased estimate of the true average
  treatment effect.
- **NSW-treated vs. PSID/CPS-controls** — an observational substitute. These
  comparison groups are drawn from national surveys rather than the same
  randomized study, so they're a much larger and more diverse population,
  and comparisons against them are meaningfully confounded by selection bias
  (job-training participants and survey respondents differ systematically —
  in age, prior earnings, education, and more). This is the classic
  causal-inference test case: can a method recover the *true* effect using
  only the observational comparison?

Features: `age`, `educ`, `black`, `hisp`, `married`, `nodegree`, `re74`,
`re75`. Treatment: `treat`. Outcome: `re78`. `causal_bench.load_lalonde()`
loads this data (with an optional `variant="nsw_psid_trimmed"` that restricts
the PSID controls to common propensity-score support, since the untrimmed
comparison has very little covariate overlap with the treated group).

**Why this alone isn't enough for this survey**: even in the best case ( a
randomized experiment like NSW's, which most real-world datasets don't even
have) real data only ever tells you the *population-level* effect. It can
never tell you the *individual-level* effect, because every unit receives
exactly one treatment and shows exactly one outcome. A unit's outcome under
the treatment it *didn't* receive is simply never observed, for anyone, ever
— the fundamental problem of causal inference. Since this survey is about how
well causal foundation models estimate *individual-level* (CATE) effects, not
just population averages, real data structurally cannot serve as its main
benchmark. That's what RealCause is for.

## §2. RealCause: the data this benchmark actually uses

### Who built it, and the core idea

**RealCause** (Neal, Huang & Raghupathi, 2020, arXiv:2011.15007) solves the
problem §1 ends on. Instead of trying to observe both of a unit's potential
outcomes — which reality never allows — it **learns a model of the
outcome-generating process** that can be queried for both:

1. Take the real Lalonde covariates `X` and real treatment assignment `T` —
   unchanged from the real data described in §1.
2. Fit a generative model (separately per cohort — PSID or CPS) to the joint
   distribution of `(X, T, Y)` observed in that real data: learn, from real
   examples, what outcome a unit with these covariates tends to produce
   under a given treatment.
3. Query that fitted model twice for the **same** unit — once conditioned on
   `T=0`, once on `T=1`. A real-world unit can only ever receive one
   treatment, but a fitted model has no such restriction: it will generate a
   draw under either condition on request. This gives simulated potential
   outcomes `y0` and `y1` for *every* unit, and therefore a known individual
   treatment effect `ite = y1 - y0` for every unit — exactly the quantity
   real data can never supply.

So a RealCause "realization" of Lalonde has the real covariates and real
treatment assignment, but outcomes drawn from a fitted generative model
rather than observed in reality — real `X` and `T`, simulated `Y`. We use
this as the benchmark's primary data specifically because it's the only
version of this dataset where an individual-level ground truth exists to
check a model's CATE estimates against at all.

### What "realizations" means

RealCause fits one generative model per cohort (PSID or CPS). Each
"realization" is one independent sample drawn from that fitted model — the
same real covariates and treatment assignment every time, but a different
simulated draw of `y0`/`y1` (and therefore a different `ite`) each time.
CausalPFN's repo ships 100 such pre-computed realizations per cohort
(the same role IHDP's 100 fixed realizations play in that benchmark): it
turns one noisy point estimate into a distribution over draws, so a single
lucky or unlucky simulated dataset doesn't dominate the reported number.

### Where the data comes from, and how it's scored

`causal_bench.load_lalonde_realcause()` downloads (and locally caches) these
realizations directly from `vdblm/CausalPFN`'s own repo and replicates their scoring
methodology:

- Per realization: permute all rows, first 90% → train (fit the CATE model
  on this), held-out 10% → test (score PEHE on these, against their true
  `ite`).
- The ATE metric uses a separate model instance fit on the *full* realization
  data, scored against `ite.mean()` over all units — not reused from the
  CATE model's train-only fit.

### Two things to know before comparing to the paper's numbers

- **Realization count**: CausalPFN's paper has two arXiv revisions with
  different RealCause numbers. **v1** averages the first 10 of the 100
  available realizations per cohort — what `load_lalonde_realcause`'s
  default (`n_realizations=10`) reproduces, and what this repo targets. The
  current **v2** averages all 100 and reports different numbers. If you've
  compared against v2, that's why the numbers won't match.
- **Units**: the paper reports PEHE in units of $1,000. `evaluate_cate`
  computes PEHE in raw dollars; `RealCause_with_hpo_benchmark.ipynb`'s
  results table divides by 1,000 before displaying it, to match.

### Reference numbers (CausalPFN, from the paper's Table 1)

| Cohort | Mean PEHE (×10³) | Mean ATE relative error |
|---|---|---|
| Lalonde PSID | 14.40 ± 0.2 | 0.22 ± 0.02 |
| Lalonde CPS | 8.96 ± 0.02 | 0.13 ± 0.01 |

If your own CausalPFN run through `RealCause_with_hpo_benchmark.ipynb` lands
in this range, your setup is working correctly.
