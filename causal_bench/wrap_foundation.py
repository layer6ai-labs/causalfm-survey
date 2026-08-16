"""Shared mechanics for the foundation-model benchmark wrappers."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


Prediction = tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]


class _StandardizedFoundationWrapper:
    """Centralize scaling and the common benchmark ``run`` contract."""

    def __init__(self):
        self._x_scaler: Optional[StandardScaler] = None
        self._y_scaler: Optional[StandardScaler] = None

    def _fit_scalers(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32).reshape(-1, 1)
        self._x_scaler = StandardScaler().fit(X)
        self._y_scaler = StandardScaler().fit(Y)
        X_scaled = self._x_scaler.transform(X).astype(np.float32)
        Y_scaled = self._y_scaler.transform(Y).reshape(-1).astype(np.float32)
        return X_scaled, Y_scaled

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        if self._x_scaler is None:
            raise RuntimeError("fit() must be called before predict()")
        return self._x_scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)

    def _unscale_effect(self, effect: np.ndarray) -> np.ndarray:
        if self._y_scaler is None:
            raise RuntimeError("fit() must be called before predicting an effect")
        return np.asarray(effect) * self._y_scaler.scale_[0]

    def _estimate_ate_for_run(
        self,
        X_train: np.ndarray,
        T_train: np.ndarray,
        Y_train: np.ndarray,
        tau_hat: np.ndarray,
    ) -> float:
        return float(np.mean(tau_hat))

    def run(self, X_train, T_train, Y_train, X_test):
        t0 = time.time()
        self.fit(X_train, T_train, Y_train)
        tau_hat, lower, upper = self.predict(X_test)
        ate_hat = self._estimate_ate_for_run(X_train, T_train, Y_train, tau_hat)
        return tau_hat, lower, upper, ate_hat, time.time() - t0
