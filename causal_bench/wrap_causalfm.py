"""
Thin wrapper around CausalFM-toolkit
(https://github.com/yccm/CausalFM-toolkit) exposing a common
`.fit(X, T, Y)` / `.predict(X)` interface for the benchmark.

CausalFM: Ma, Frauen, Javurek & Feuerriegel, "Foundation Models for Causal
Inference via Prior-Data Fitted Networks", arXiv:2506.10914 (ICLR 2026).

Install:
    git clone https://github.com/yccm/CausalFM-toolkit.git
    cd CausalFM-toolkit && pip install -r requirements.txt

Per the toolkit's quick-start example:

    from causalfm.data import StandardCATEGenerator
    from causalfm.models import StandardCATEModel
    from causalfm.evaluation import compute_pehe

    model = StandardCATEModel.from_pretrained("checkpoints/checkpoints_standard/best_model.pth")
    result = model.estimate_cate(x_train, a_train, y_train, x_test)
    cate = result['cate']

This wrapper uses `StandardCATEModel` for the standard CATE setting (our
`linear_confounded` / `nonlinear_heterogeneous` datasets). A pretrained
checkpoint path must be supplied (see notebook setup); if unavailable,
`is_available()` returns False and the benchmark skips CausalFM.
"""

from __future__ import annotations

from functools import lru_cache
import os

import numpy as np
import torch
from typing import Optional

from .wrap_foundation import Prediction, _StandardizedFoundationWrapper


# StandardCATEModel inference is eval/no-grad and does not retain a training
# context by default, so sequential wrappers can share immutable weights.
@lru_cache(maxsize=4)
def _load_causalfm_model(checkpoint_path: str, device: str):
    from causalfm.models import StandardCATEModel

    return StandardCATEModel.from_pretrained(checkpoint_path, device=device)


class CausalFMWrapper(_StandardizedFoundationWrapper):
    name = "CausalFM"

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    @classmethod
    def is_available(cls, checkpoint_path: Optional[str] = None) -> bool:
        try:
            from causalfm.models import StandardCATEModel  # noqa: F401
        except Exception:
            return False
        if checkpoint_path is not None:
            return os.path.exists(checkpoint_path)
        return True

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """CausalFM is amortized (zero-shot): 'fit' loads the pretrained
        model and stores the training context (X, T, Y), which is passed
        to `estimate_cate` together with the test points -- this mirrors
        the toolkit's `model.estimate_cate(x_train, a_train, y_train,
        x_test)` signature. X and Y are standardized here -- CausalFM is
        pretrained on normalized synthetic priors, and raw real-world
        scales (e.g. Lalonde's dollar-denominated features/outcome) are out
        of that distribution; `predict` converts CATE back to the original
        outcome scale by multiplying by Y's std (the mean cancels in a
        difference of group means)."""
        checkpoint_path = os.path.realpath(self.checkpoint_path)
        self._model = _load_causalfm_model(checkpoint_path, self.device)
        X_s, Y_s = self._fit_scalers(X, Y)
        tensor_device = self._model.device
        self._X_train = torch.as_tensor(X_s, dtype=torch.float32, device=tensor_device)
        self._T_train = torch.as_tensor(T, dtype=torch.float32, device=tensor_device).reshape(-1, 1)
        self._Y_train = torch.as_tensor(Y_s, dtype=torch.float32, device=tensor_device).reshape(
            -1, 1
        )
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        # The toolkit's PerFeatureTransformerCATE expects torch.Tensor inputs,
        # with treatment/outcome shaped [N, 1] (not the 1-D arrays our common
        # wrapper interface uses) -- see `_pack_eval_io` in
        # src/tabpfn/model/causalFM.py.
        X_test_t = torch.as_tensor(
            self._transform_x(X), dtype=torch.float32, device=self._model.device
        )
        result = self._model.estimate_cate(self._X_train, self._T_train, self._Y_train, X_test_t)

        tau_hat = self._unscale_effect(result["cate"].detach().cpu().numpy().reshape(-1))
        lower = upper = None
        # Optional calibrated uncertainty intervals, if the toolkit
        # returns them (key names per docs: 'cate_lower'/'cate_upper'
        # or 'ci_lower'/'ci_upper')
        for lk, uk in (("cate_lower", "cate_upper"), ("ci_lower", "ci_upper")):
            if lk in result and uk in result:
                lower = self._unscale_effect(result[lk].detach().cpu().numpy().reshape(-1))
                upper = self._unscale_effect(result[uk].detach().cpu().numpy().reshape(-1))
                break
        return tau_hat, lower, upper
