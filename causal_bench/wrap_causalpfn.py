"""
Thin wrapper around CausalPFN (https://github.com/vdblm/CausalPFN) exposing
a common `.fit(X, T, Y)` / `.predict(X)` interface for the benchmark.

CausalPFN: Balazadeh et al., "CausalPFN: Amortized Causal Effect Estimation
via In-Context Learning", arXiv:2506.07918.

Install:
    pip install "git+https://github.com/vdblm/CausalPFN"

The first call downloads pretrained weights from the Hugging Face Hub
(~ a few hundred MB), so an internet connection is required on first run.
"""

from __future__ import annotations
from typing import Optional
import numpy as np

from .wrap_foundation import Prediction, _StandardizedFoundationWrapper


class CausalPFNWrapper(_StandardizedFoundationWrapper):
    name = "CausalPFN"

    def __init__(
        self,
        device: str = "cpu",
        verbose: bool = False,
        max_context_length: int = 4096,
        max_query_length: int = 4096,
        num_neighbours: int = 1024,
        cap_num_neighbours: bool = True,
        alpha: Optional[float] = None,
        interval_n_samples: int = 10_000,
    ):
        for name, value in (
            ("max_context_length", max_context_length),
            ("max_query_length", max_query_length),
            ("num_neighbours", num_neighbours),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if max_context_length < 2:
            raise ValueError("max_context_length must be at least 2")
        if not isinstance(cap_num_neighbours, (bool, np.bool_)):
            raise TypeError("cap_num_neighbours must be a boolean")

        super().__init__()
        self.device = device
        self.verbose = verbose
        self.max_context_length = int(max_context_length)
        self.max_query_length = int(max_query_length)
        self.num_neighbours = int(num_neighbours)
        self.cap_num_neighbours = bool(cap_num_neighbours)
        if alpha is not None and not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        self.alpha = None if alpha is None else float(alpha)
        if not isinstance(interval_n_samples, (int, np.integer)) or interval_n_samples <= 0:
            raise ValueError("interval_n_samples must be a positive integer")
        self.interval_n_samples = int(interval_n_samples)
        self._effective_num_neighbours = None
        self._cate_estimator = None
        self._ate_estimator = None
        self._X_train = None
        self._T_train = None
        self._Y_train = None

    def _reset_fit_state(self) -> None:
        super()._reset_fit_state()
        self._effective_num_neighbours = None
        self._cate_estimator = None
        self._ate_estimator = None
        self._X_train = None
        self._T_train = None
        self._Y_train = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from causalpfn import CATEEstimator, ATEEstimator  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Standardize and retain the context used by the requested estimator.

        CATE and ATE estimators are initialized lazily because each loads the
        same checkpoint and trains its own weak learner. Most benchmark paths
        request only one of them, so eagerly fitting both doubles that work.

        CausalPFN is pretrained on normalized synthetic priors, and raw
        real-world scales (e.g. Lalonde's dollar-denominated features/outcome)
        are out of that distribution. Only a linear rescaling, so CATE/ATE are
        converted back to the original outcome scale in `predict`/`estimate_ate`.

        By default this wrapper caps k to the *smaller* arm and half the
        context limit, which is how ``data/benchmark_results_*.csv`` were
        produced. Upstream caps each arm separately instead; set
        ``cap_num_neighbours=False`` to get exactly that behavior.
        """
        self._reset_fit_state()
        X_arr, T_arr, Y_arr = self._validate_fit_data(X, T, Y)

        n_control = int(np.count_nonzero(T_arr == 0.0))
        n_treated = int(np.count_nonzero(T_arr == 1.0))
        min_arm_size = min(n_control, n_treated)

        max_neighbours_for_context = self.max_context_length // 2
        if self.cap_num_neighbours:
            self._effective_num_neighbours = min(
                self.num_neighbours, min_arm_size, max_neighbours_for_context
            )
        else:
            self._effective_num_neighbours = self.num_neighbours

        self._X_train, self._Y_train = self._fit_scalers(X_arr, Y_arr)
        self._T_train = T_arr
        return self

    def _ensure_cate_estimator(self):
        if self._X_train is None:
            raise RuntimeError("fit() must be called before predict()")
        if self._cate_estimator is None:
            from causalpfn import CATEEstimator

            estimator = CATEEstimator(
                device=self.device,
                verbose=self.verbose,
                max_context_length=self.max_context_length,
                max_query_length=self.max_query_length,
                num_neighbours=self._effective_num_neighbours,
            )
            estimator.fit(self._X_train, self._T_train, self._Y_train)
            self._cate_estimator = estimator
        return self._cate_estimator

    def _ensure_ate_estimator(self):
        if self._X_train is None:
            raise RuntimeError("fit() must be called before estimate_ate()")
        if self._ate_estimator is None:
            from causalpfn import ATEEstimator

            estimator = ATEEstimator(
                device=self.device,
                verbose=self.verbose,
                max_context_length=self.max_context_length,
                max_query_length=self.max_query_length,
                num_neighbours=self._effective_num_neighbours,
            )
            estimator.fit(self._X_train, self._T_train, self._Y_train)
            self._ate_estimator = estimator
        return self._ate_estimator

    def predict(self, X: np.ndarray) -> Prediction:
        """Return CATE predictions, with credible intervals when asked for.

        Intervals are off by default because they cost an extra forward pass
        over ``2 * len(X)`` query rows drawing ``interval_n_samples`` posterior
        draws each. Pass ``alpha`` to the constructor to turn them on; without
        them ``coverage_95`` cannot be computed for this model, which is why
        that column is blank for CausalPFN in ``data/benchmark_results_*.csv``.
        """
        X_s = self._transform_x(X)
        estimator = self._ensure_cate_estimator()
        tau_hat_s = np.asarray(estimator.estimate_cate(X_s)).reshape(-1)
        tau_hat = self._unscale_effect(tau_hat_s)
        if self.alpha is None:
            return tau_hat, None, None

        bounds = estimator.estimate_cate_CI(
            X_s, alpha=self.alpha, n_samples=self.interval_n_samples
        )
        lower = self._unscale_effect(np.asarray(bounds["lower_bound"]).reshape(-1))
        upper = self._unscale_effect(np.asarray(bounds["upper_bound"]).reshape(-1))
        return tau_hat, lower, upper

    def estimate_att_ci(
        self, X: np.ndarray, alpha: float = 0.05, n_samples: Optional[int] = None
    ) -> tuple[float, float, float]:
        """Point estimate and credible interval for the mean effect over ``X``.

        Query ``X`` at the treated rows and this is the ATT; query it at
        everything and it is the ATE. The interval is the posterior over the
        *mean* effect, which is narrower than averaging per-unit intervals and
        is the quantity a "how big could this be?" question wants.
        """
        X_s = self._transform_x(X)
        out = self._ensure_cate_estimator().estimate_ate_CI(
            X_s, alpha=alpha, n_samples=n_samples or self.interval_n_samples
        )
        point = float(np.asarray(out["ate"]).reshape(-1)[0])
        lo = float(np.asarray(out["lower_bound"]).reshape(-1)[0])
        hi = float(np.asarray(out["upper_bound"]).reshape(-1)[0])
        return (
            float(self._unscale_effect(point)),
            float(self._unscale_effect(lo)),
            float(self._unscale_effect(hi)),
        )

    def estimate_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
        ate_hat_s = float(np.asarray(self._ensure_ate_estimator().estimate_ate()).reshape(-1)[0])
        return float(self._unscale_effect(ate_hat_s))

    def _estimate_ate_for_run(self, X_train, T_train, Y_train, tau_hat) -> float:
        return self.estimate_ate(X_train, T_train, Y_train)
