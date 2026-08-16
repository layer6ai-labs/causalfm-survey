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
import numpy as np

from .wrap_foundation import Prediction, _StandardizedFoundationWrapper


class CausalPFNWrapper(_StandardizedFoundationWrapper):
    name = "CausalPFN"

    def __init__(self, device: str = "cpu", verbose: bool = False):
        super().__init__()
        self.device = device
        self.verbose = verbose
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
        """
        self._X_train, self._Y_train = self._fit_scalers(X, Y)
        self._T_train = np.asarray(T, dtype=np.float32).reshape(-1)
        self._cate_estimator = None
        self._ate_estimator = None
        return self

    def _ensure_cate_estimator(self):
        if self._cate_estimator is None:
            from causalpfn import CATEEstimator

            self._cate_estimator = CATEEstimator(device=self.device, verbose=self.verbose)
            self._cate_estimator.fit(self._X_train, self._T_train, self._Y_train)
        return self._cate_estimator

    def _ensure_ate_estimator(self):
        if self._ate_estimator is None:
            from causalpfn import ATEEstimator

            self._ate_estimator = ATEEstimator(device=self.device, verbose=self.verbose)
            self._ate_estimator.fit(self._X_train, self._T_train, self._Y_train)
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
