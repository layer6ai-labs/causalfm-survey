# Wrappers Guide

`causal_bench`'s `wrap_*.py` files adapt all of foundational models and metalearners (each
with its own install method, constructor signature, and fit/predict
API) into one shared interface:

```python
model.fit(X_train, T_train, Y_train)
tau_hat, lower, upper = model.predict(X_test)
tau_hat, lower, upper, ate_hat, runtime = model.run(X_train, T_train, Y_train, X_test)
model.is_available()  # classmethod: is the underlying library/checkpoint usable right now?
```

Without wrappers, the same nine models require nine different calling
conventions — e.g. econml's metalearners take `.fit(Y, T, X=X)` (note the
order) and return effects via `.effect(X)`, not `.fit(X, T, Y)` /
`.predict(X)`; CausalFM's native call returns a dict (`result["cate"]`);
Do-PFN needs treatment concatenated into the feature matrix in a specific
column and expects `torch.Tensor`, not numpy. The wrapper is what lets
benchmark code loop over all nine identically instead of branching per model.

The three foundation-model wrappers (`wrap_causalpfn.py`, `wrap_dopfn.py`,
`wrap_causalfm.py`) share a common base class, `_StandardizedFoundationWrapper`
in `wrap_foundation.py` — it centralizes input validation, `X`/`Y`
standardization (see below), and the `run()` contract, so each model-specific
wrapper only implements `fit`/`predict` against its own library.

## Model-by-model notes

| Model | Underlying lib | Wrapper-specific gotcha |
|---|---|---|
| **CausalPFN** | `causalpfn` (PyPI) | `is_available()` returns `False` on Apple Silicon macOS — the library segfaults there, not a normal exception `fit()` could catch. Constructor also takes `cap_num_neighbours` (default `True`) — caps CausalPFN's neighbor count to the smaller treatment arm to avoid FAISS `-1` sentinels; set `False` for exact upstream behavior |
| **Do-PFN** | `jr2021/Do-PFN` (git clone) | Constructor takes `repo_dir` (default `"Do-PFN"`); needs `torch<2.10`; wrapper handles the treatment-in-column-0 requirement and the repo-relative checkpoint path internally. `is_available(repo_dir=...)` and `clear_model_cache()` (clears Do-PFN's process-wide checkpoint cache) are classmethods |
| **CausalFM** | `yccm/CausalFM-toolkit` (git clone) | Constructor **requires** `checkpoint_path` (no default) — real path is `checkpoints/checkpoints_standard/best_model.pth` inside the clone, not `checkpoints/best_model.pth`. Also takes `query_batch_size` (default `None` = native unbatched) as an OOM fallback for large query sets. `is_available(checkpoint_path=...)` and `clear_model_cache()` are classmethods |
| **S/T/X-learner, Debiased ML, DR** | `econml.metalearners` / `econml.dml` / `econml.dr.ForestDRLearner` | Wrapper reorders args to `.fit(X, T, Y)` and exposes `.predict(X)` in place of `.effect(X)`; default base model is `RandomForestRegressor`, override via the `model=` constructor arg |
| **IPW** | `sklearn` directly (no dedicated econml class) | Manual propensity + Hájek-normalized weighted outcome-mean implementation. `supports_cate = False` (a class attribute) — IPW only ever produces a single scalar ATE, not a per-unit CATE, so `predict()` returns that same value broadcast across every row rather than a real CATE estimate |

## Hyperparameter optimization (all six metalearners, including IPW)

Every metalearner wrapper (S/T/X-learner, Debiased ML, DR, **and IPW** )
accepts `hpo=True` (default `False`) and an optional
`hpo_config=HPOConfig(time_budget=900.0, cv=3, verbose=0, early_stop=True)`.
When enabled, nuisance models (outcome regressors, propensity classifiers) are
selected via a FLAML `AutoML` search instead of a fixed `RandomForestRegressor`
— the same search CausalPFN's own paper uses for its EconML baselines. IPW
gets this too even though it has no CATE output: both its propensity model
and its per-arm outcome regressors go through the same `_regressor`/
`_propensity` helpers as the other five. This needs `FLAML[automl]==2.3.5`
installed (see `CLAUDE.md`); `hpo=False` has no such dependency.
`RealCause_with_hpo_benchmark.ipynb` runs the full benchmark with `hpo=True`,
since a fixed random forest isn't a fair comparison against a tuned one when
reproducing a paper's numbers.

For exact, verified example code calling each **foundation** model's native
API directly (no wrapper) — including every install/environment gotcha
found while getting them running — see
`notebooks/Foundation_models_sandbox.ipynb`. That notebook is the source of
truth for raw-API usage;

## Standardization

All three foundation-model wrappers fit a `StandardScaler` on `X` (and `Y`)
inside `fit()` and inverse-transform predictions back to the original
scale — the underlying models are pretrained on normalized synthetic data,
so raw real-world scales (e.g. Lalonde's dollar-denominated features) put
them out of distribution.

The `econml`-based metalearner wrappers (S/T/X-learner, Debiased ML, DR) do
the same — matching CausalPFN's own reference `EconMLBaseline` preprocessing
— so they're just as scale-robust when handed raw real-world data.
**`IPWWrapper` is the one exception**: it stays on raw `X`/`Y` deliberately,
since the reference `IPWBaseline` it matches is a separate baseline that
uses raw inputs with Hájek-normalized weighted group means, not the
`EconMLBaseline` preprocessing the other five follow.

See the "Foundation-model wrappers standardize..." note in `CLAUDE.md` for
the full explanation. This is invisible to callers either way: `fit`/`predict`
still take/return values in your original units regardless of which wrappers
scale internally.

## With wrapper vs. without

Use the wrapper (via `causal_bench`) for: comparing/looping over multiple
models, benchmarking, notebooks. Use each library's native API directly
(see the sandbox notebook) for: deep-diving one model, using
library-specific features the common interface doesn't expose (e.g.
CausalPFN's calibrated quantiles), or copy-pasting a single model into
another project.
