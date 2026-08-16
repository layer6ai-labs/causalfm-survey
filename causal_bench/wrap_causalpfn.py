"""
Thin wrapper around CausalPFN (https://github.com/vdblm/CausalPFN) exposing
a common `.fit(X, T, Y)` / `.predict(X)` interface for the benchmark.

CausalPFN: Balazadeh et al., "CausalPFN: Amortized Causal Effect Estimation
via In-Context Learning", arXiv:2506.07918.

Install:
    pip install causalpfn

The first call downloads pretrained weights from the Hugging Face Hub
(~ a few hundred MB), so an internet connection is required on first run.
"""

from __future__ import annotations
import platform
import warnings
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
        # CausalPFN segfaults on Apple Silicon macOS -- a hard process crash,
        # not a catchable exception -- on both CPU and MPS (likely an
        # unstable scaled_dot_product_attention kernel, not a CUDA
        # requirement). Report unavailable here rather than let `fit()`
        # crash the interpreter. Fine on Colab (CPU or GPU).
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return False
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

        CausalPFN 0.1.4 asks FAISS for ``num_neighbours`` from each arm even
        when an arm is smaller, in which case FAISS returns -1 sentinels. By
        default this wrapper caps k to the smaller arm and half the context
        limit; set ``cap_num_neighbours=False`` only for exact upstream behavior.
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
            if (
                self.num_neighbours > min_arm_size
                or 2 * self.num_neighbours > self.max_context_length
            ):
                warnings.warn(
                    "cap_num_neighbours=False preserves upstream behavior, but the "
                    "requested k exceeds an arm size or half the context limit; "
                    "FAISS can return -1 sentinel neighbors or CausalPFN can exceed "
                    "max_context_length.",
                    RuntimeWarning,
                    stacklevel=2,
                )

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
        """Return CATE predictions and optional interval placeholders."""
        X_s = self._transform_x(X)
        tau_hat_s = np.asarray(self._ensure_cate_estimator().estimate_cate(X_s)).reshape(-1)
        return self._unscale_effect(tau_hat_s), None, None

    def estimate_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
        ate_hat_s = float(np.asarray(self._ensure_ate_estimator().estimate_ate()).reshape(-1)[0])
        return float(self._unscale_effect(ate_hat_s))

    def _estimate_ate_for_run(self, X_train, T_train, Y_train, tau_hat) -> float:
        return self.estimate_ate(X_train, T_train, Y_train)
