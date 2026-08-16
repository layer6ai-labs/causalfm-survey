"""
Thin wrapper around Do-PFN (https://github.com/jr2021/Do-PFN) exposing a
common `.fit(X, T, Y)` / `.predict(X)` interface for the benchmark.

Do-PFN: Robertson & Reuter et al., "Do-PFN: In-Context Learning for Causal
Effect Estimation", arXiv:2506.06039.

Do-PFN is NOT distributed on PyPI -- it must be cloned from GitHub:

    git clone https://github.com/jr2021/Do-PFN.git

Don't `pip install -r Do-PFN/requirements.txt` as-is -- it's a frozen
research/benchmark environment (pins `catboost==1.1.1`, which has no wheel
for recent Python, purely for baseline comparisons `DoPFNRegressor` never
imports). The actual runtime deps beyond `torch`/`numpy`/`scipy`/`pandas`/
`scikit-learn` are just `networkx`, `tqdm`, `einops`.

This wrapper assumes the cloned `Do-PFN` directory lives at `repo_dir`
(default `"Do-PFN"`, relative to the process's cwd). Verified directly
against the current `jr2021/Do-PFN` main branch source (its README has
drifted from the code):

    from scripts.transformer_prediction_interface import DoPFNRegressor  # not `dopfn`

    dopfn = DoPFNRegressor()
    dopfn.fit(X_full_train, Y_train)          # X_full: treatment in COLUMN 0
    tau_hat = dopfn.predict_cate(X_full_test)  # torch.Tensor input required

`DoPFNRegressor()` reads its config via a path relative to the Do-PFN
repository, and `fit()` initializes the checkpoint. Do-PFN itself caches the
loaded model in `TabPFNBaseModel.models_in_memory`; this wrapper preserves
that cache and only centralizes the required temporary `chdir`.

Requires `torch<2.10`: Do-PFN's own `model/layer.py` imports `Optional` from
`torch.nn.modules.transformer`, an unofficial re-export PyTorch removed in
2.10. Raises `ImportError: cannot import name 'Optional' from
'torch.nn.modules.transformer'` on newer torch -- not something this wrapper
can work around.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys

import numpy as np
import torch

from .wrap_foundation import Prediction, _StandardizedFoundationWrapper


def _add_repo_to_path(repo_dir: str) -> str:
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    return repo_dir


@contextmanager
def _working_directory(path: str):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class DoPFNWrapper(_StandardizedFoundationWrapper):
    name = "Do-PFN"

    def __init__(self, repo_dir: str = "Do-PFN", device: str = "cpu"):
        """
        repo_dir: path to the cloned Do-PFN repo root (checkpoint paths are
            relative to it, so this wrapper `chdir`s there for
            construction/fit/predict).
        device: accepted for interface consistency with the other wrappers;
            DoPFNRegressor takes no device argument of its own.
        """
        super().__init__()
        self.repo_dir = os.path.abspath(repo_dir)
        self.device = device
        self._model = None

    @classmethod
    def is_available(cls, repo_dir: str = "Do-PFN") -> bool:
        if not os.path.isdir(repo_dir):
            return False
        _add_repo_to_path(repo_dir)
        try:
            from scripts.transformer_prediction_interface import DoPFNRegressor  # noqa: F401

            return True
        except Exception:
            return False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        """Builds the combined design matrix [T, X] -- treatment in column
        0, since `predict_cid` overwrites `X[:, 0]` internally regardless of
        what's there at prediction time. Standardizes X and Y (not T --
        `predict_cid` sets that column to exactly 0/1, which must stay
        untransformed) since Do-PFN is pretrained on normalized synthetic
        priors; converted back to the original outcome scale in `predict`
        by multiplying by Y's std (the mean cancels in a difference of
        group means)."""
        _add_repo_to_path(self.repo_dir)
        from scripts.transformer_prediction_interface import DoPFNRegressor

        T = np.asarray(T, dtype=np.float32).reshape(-1, 1)
        X_s, Y_s = self._fit_scalers(X, Y)
        X_full = np.concatenate([T, X_s], axis=1)

        with _working_directory(self.repo_dir):
            self._model = DoPFNRegressor()
            self._model.show_progress = False
            self._model.fit(X_full, Y_s)
        return self

    def predict(self, X: np.ndarray) -> Prediction:
        """Return CATE predictions from Do-PFN's dedicated method."""
        X_s = self._transform_x(X)
        # Column 0 is a placeholder -- predict_cate overwrites it internally.
        X_full = np.concatenate([np.zeros((len(X_s), 1), dtype=np.float32), X_s], axis=1)

        with _working_directory(self.repo_dir):
            tau_hat_s = np.asarray(self._model.predict_cate(torch.as_tensor(X_full))).reshape(-1)
        return self._unscale_effect(tau_hat_s), None, None
