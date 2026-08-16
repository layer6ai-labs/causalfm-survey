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

    def _reset_fit_state(self) -> None:
        self._x_scaler = None
        self._y_scaler = None

    @staticmethod
    def _validate_fit_data(
        X: np.ndarray, T: np.ndarray, Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X_arr.shape}")
        if X_arr.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if X_arr.shape[1] == 0:
            raise ValueError("X must contain at least one feature")

        def as_vector(values: np.ndarray, name: str) -> np.ndarray:
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            elif arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D or [N, 1], got shape {arr.shape}")
            if len(arr) != len(X_arr):
                raise ValueError(
                    f"X and {name} have inconsistent lengths: {len(X_arr)} != {len(arr)}"
                )
            return arr

        T_arr = as_vector(T, "T")
        Y_arr = as_vector(Y, "Y")

        for name, arr in (("X", X_arr), ("T", T_arr), ("Y", Y_arr)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name} must contain only finite values")

        if not np.all((T_arr == 0.0) | (T_arr == 1.0)):
            raise ValueError("T must be binary and encoded as 0/1")
        if np.unique(T_arr).size != 2:
            raise ValueError("T must contain at least one sample in each treatment arm")
        return X_arr, T_arr, Y_arr

    def _fit_scalers(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float32)
        Y_arr = np.asarray(Y, dtype=np.float32).reshape(-1, 1)
        x_scaler = StandardScaler().fit(X_arr)
        y_scaler = StandardScaler().fit(Y_arr)
        X_scaled = x_scaler.transform(X_arr).astype(np.float32)
        Y_scaled = y_scaler.transform(Y_arr).reshape(-1).astype(np.float32)
        self._x_scaler = x_scaler
        self._y_scaler = y_scaler
        return X_scaled, Y_scaled

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        if self._x_scaler is None:
            raise RuntimeError("fit() must be called before predict()")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2-D array, got shape {X_arr.shape}")
        if X_arr.shape[1] != self._x_scaler.n_features_in_:
            raise ValueError(
                f"X has {X_arr.shape[1]} features; expected {self._x_scaler.n_features_in_}"
            )
        if not np.isfinite(X_arr).all():
            raise ValueError("X must contain only finite values")
        if X_arr.shape[0] == 0:
            return X_arr
        return self._x_scaler.transform(X_arr).astype(np.float32)

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
