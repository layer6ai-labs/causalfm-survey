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

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        query_batch_size: Optional[int] = None,
    ):
        """Configure CausalFM inference.

        Query batching is mathematically invariant and can lower peak memory,
        but the toolkit recomputes the complete training context for every
        batch. ``None`` preserves native one-shot speed; use a positive value
        such as 1024 only as an OOM fallback.
        """
        if query_batch_size is not None and (
            isinstance(query_batch_size, (bool, np.bool_))
            or not isinstance(query_batch_size, (int, np.integer))
            or query_batch_size <= 0
        ):
            raise ValueError(
                f"query_batch_size must be a positive integer or None, got {query_batch_size!r}"
            )
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.query_batch_size = None if query_batch_size is None else int(query_batch_size)
        self._model = None
        self._X_train = None
        self._T_train = None
        self._Y_train = None

    def _reset_fit_state(self) -> None:
        super()._reset_fit_state()
        self._model = None
        self._X_train = None
        self._T_train = None
        self._Y_train = None

    @classmethod
    def clear_model_cache(cls) -> int:
        """Clear cached CausalFM checkpoints and return the entry count.

        Discard fitted wrapper references before calling this method; a live
        wrapper still owns its model independently of the global cache.
        """
        count = _load_causalfm_model.cache_info().currsize
        _load_causalfm_model.cache_clear()
        return count

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
        self._reset_fit_state()
        try:
            X_arr, T_arr, Y_arr = self._validate_fit_data(X, T, Y)
            checkpoint_path = os.path.realpath(self.checkpoint_path)
            model = _load_causalfm_model(checkpoint_path, self.device)
            X_s, Y_s = self._fit_scalers(X_arr, Y_arr)
            tensor_device = model.device
            X_train = torch.as_tensor(X_s, dtype=torch.float32, device=tensor_device)
            T_train = torch.as_tensor(T_arr, dtype=torch.float32, device=tensor_device).reshape(
                -1, 1
            )
            Y_train = torch.as_tensor(Y_s, dtype=torch.float32, device=tensor_device).reshape(-1, 1)
        except Exception:
            self._reset_fit_state()
            raise

        self._model = model
        self._X_train = X_train
        self._T_train = T_train
        self._Y_train = Y_train
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        if self._model is None:
            raise RuntimeError("fit() must be called before predict()")

        # The toolkit's PerFeatureTransformerCATE expects torch.Tensor inputs,
        # with treatment/outcome shaped [N, 1] (not the 1-D arrays our common
        # wrapper interface uses) -- see `_pack_eval_io` in
        # src/tabpfn/model/causalFM.py.
        # PerFeatureEncoderLayer restricts every query's item-attention keys and
        # values to rows before single_eval_pos (the training context). Feature
        # attention and MLPs are row-local, so query batching is invariant.
        X_s = self._transform_x(X)
        n_query = len(X_s)
        if n_query == 0:
            return np.empty(0, dtype=np.float32), None, None
        X_test_t = torch.as_tensor(X_s, dtype=torch.float32, device=self._model.device)

        batch_size = n_query if self.query_batch_size is None else self.query_batch_size
        cate_chunks = []
        lower_chunks = []
        upper_chunks = []
        interval_keys = None

        for start in range(0, n_query, batch_size):
            stop = min(start + batch_size, n_query)
            result = self._model.estimate_cate(
                self._X_train,
                self._T_train,
                self._Y_train,
                X_test_t[start:stop],
            )
            cate_chunks.append(result["cate"].detach().cpu().numpy().reshape(-1))

            batch_interval_keys = None
            for lk, uk in (("cate_lower", "cate_upper"), ("ci_lower", "ci_upper")):
                if lk in result and uk in result:
                    batch_interval_keys = (lk, uk)
                    break
            if start == 0:
                interval_keys = batch_interval_keys
            elif batch_interval_keys != interval_keys:
                raise RuntimeError(
                    "CausalFM returned inconsistent interval keys across query batches"
                )

            if interval_keys is not None:
                lk, uk = interval_keys
                lower_chunks.append(result[lk].detach().cpu().numpy().reshape(-1))
                upper_chunks.append(result[uk].detach().cpu().numpy().reshape(-1))

        tau_hat = self._unscale_effect(np.concatenate(cate_chunks))
        lower = upper = None
        if interval_keys is not None:
            lower = self._unscale_effect(np.concatenate(lower_chunks))
            upper = self._unscale_effect(np.concatenate(upper_chunks))
        return tau_hat, lower, upper

    def estimate_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> float:
        """Estimate ATE using CausalFM's full-context effect distribution.

        The upstream toolkit only exposes ``StandardCATEModel.estimate_cate``;
        it does not ship a separate ATE model. This wrapper-level ATE API is
        intended to be called after a fresh fit on the full realization, so it
        is procedurally independent of held-out CATE evaluation.
        """
        tau_hat, _, _ = self.predict(X)
        if tau_hat.size == 0:
            raise ValueError("ATE estimation requires at least one row")
        return float(np.mean(tau_hat))
