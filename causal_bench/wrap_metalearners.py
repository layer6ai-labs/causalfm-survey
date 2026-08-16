"""Wrappers for traditional causal estimators, with optional HPO."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


REGRESSION_SPACE = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_features": ["sqrt", 0.5, 1.0],
}
PROPENSITY_SPACE = {
    "C": np.logspace(-3, 3, 20),
    "class_weight": [None, "balanced"],
}


@dataclass(frozen=True)
class HPOConfig:
    n_iter: int = 20
    cv: int = 3
    n_jobs: int = -1
    random_state: int = 0


def _arrays(X, T, Y):
    X = np.asarray(X, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32).reshape(-1)
    Y = np.asarray(Y, dtype=np.float32).reshape(-1)
    return X, T, Y


def _tune(estimator, X, y, space, config, scoring, stratified=False):
    from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold

    limit = np.unique(y, return_counts=True)[1].min() if stratified else len(y)
    n_splits = min(config.cv, int(limit))
    if n_splits < 2:
        raise ValueError("HPO requires at least two samples per CV split")
    splitter = StratifiedKFold if stratified else KFold
    cv = splitter(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    search = RandomizedSearchCV(
        estimator,
        space,
        n_iter=config.n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


class _BaseWrapper:
    def __init__(self, model=None, device="cpu", hpo=False, hpo_config=None):
        self.model = model
        self.device = device
        self.hpo = hpo
        self.hpo_config = hpo_config or HPOConfig()
        self.best_params_ = {}
        self._learner = None

    def _regressor(self, X, y):
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestRegressor

        model = clone(self.model) if self.model is not None else RandomForestRegressor(
            n_estimators=100, random_state=self.hpo_config.random_state
        )
        if not self.hpo:
            return model, None
        return _tune(
            model,
            X,
            y,
            REGRESSION_SPACE,
            self.hpo_config,
            "neg_root_mean_squared_error",
        )

    def _propensity(self, X, T):
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=2000, random_state=self.hpo_config.random_state)
        if not self.hpo:
            model.fit(X, T)
            return model, None
        return _tune(
            model,
            X,
            T,
            PROPENSITY_SPACE,
            self.hpo_config,
            "neg_log_loss",
            stratified=True,
        )

    def predict(self, X):
        tau = self._learner.effect(np.asarray(X, dtype=np.float32)).reshape(-1)
        return tau, None, None

    def run(self, X_train, T_train, Y_train, X_test):
        start = time.time()
        self.fit(X_train, T_train, Y_train)
        tau, lower, upper = self.predict(X_test)
        return tau, lower, upper, float(tau.mean()), time.time() - start


class SLearnerWrapper(_BaseWrapper):
    name = "S-learner"

    @classmethod
    def is_available(cls):
        try:
            from econml.metalearners import SLearner  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X, T, Y):
        from econml.metalearners import SLearner

        X, T, Y = _arrays(X, T, Y)
        model, params = self._regressor(np.column_stack([X, T]), Y)
        if params:
            self.best_params_["outcome"] = params
        self._learner = SLearner(overall_model=model).fit(Y, T, X=X)
        return self


class TLearnerWrapper(_BaseWrapper):
    name = "T-learner"

    @classmethod
    def is_available(cls):
        try:
            from econml.metalearners import TLearner  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X, T, Y):
        from econml.metalearners import TLearner

        X, T, Y = _arrays(X, T, Y)
        models = []
        for treatment in (0, 1):
            mask = T == treatment
            model, params = self._regressor(X[mask], Y[mask])
            models.append(model)
            if params:
                self.best_params_[f"outcome_{treatment}"] = params
        self._learner = TLearner(models=models).fit(Y, T, X=X)
        return self


class XLearnerWrapper(_BaseWrapper):
    name = "X-learner"

    @classmethod
    def is_available(cls):
        try:
            from econml.metalearners import XLearner  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X, T, Y):
        from econml.metalearners import XLearner

        X, T, Y = _arrays(X, T, Y)
        models = []
        for treatment in (0, 1):
            mask = T == treatment
            model, params = self._regressor(X[mask], Y[mask])
            models.append(model)
            if params:
                self.best_params_[f"outcome_{treatment}"] = params
        propensity, params = self._propensity(X, T)
        if params:
            self.best_params_["propensity"] = params
        self._learner = XLearner(models=models, propensity_model=propensity).fit(Y, T, X=X)
        return self


class DebiasedMLWrapper(_BaseWrapper):
    name = "Debiased ML"

    @classmethod
    def is_available(cls):
        try:
            from econml.dml import DML  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X, T, Y):
        from econml.dml import DML
        from sklearn.linear_model import LinearRegression

        X, T, Y = _arrays(X, T, Y)
        model_y, params_y = self._regressor(X, Y)
        model_t, params_t = self._regressor(X, T)
        if params_y:
            self.best_params_.update(outcome=params_y, treatment=params_t)
        self._learner = DML(
            model_y=model_y,
            model_t=model_t,
            model_final=LinearRegression(),
        ).fit(Y, T, X=X)
        return self


class IPWWrapper(_BaseWrapper):
    name = "IPW"

    def __init__(self, device="cpu", hpo=False, hpo_config=None):
        super().__init__(device=device, hpo=hpo, hpo_config=hpo_config)

    @classmethod
    def is_available(cls):
        return True

    def fit(self, X, T, Y):
        X, T, Y = _arrays(X, T, Y)
        self._propensity_model, params = self._propensity(X, T)
        if params:
            self.best_params_["propensity"] = params
        for treatment in (0, 1):
            mask = T == treatment
            model, params = self._regressor(X[mask], Y[mask])
            model.fit(X[mask], Y[mask])
            setattr(self, f"_outcome_model_{treatment}", model)
            if params:
                self.best_params_[f"outcome_{treatment}"] = params
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        tau = self._outcome_model_1.predict(X) - self._outcome_model_0.predict(X)
        return tau.reshape(-1), None, None


class DRWrapper(_BaseWrapper):
    name = "DR (Doubly Robust)"

    def __init__(self, device="cpu", hpo=False, hpo_config=None):
        super().__init__(device=device, hpo=hpo, hpo_config=hpo_config)

    @classmethod
    def is_available(cls):
        return True

    def fit(self, X, T, Y):
        X, T, Y = _arrays(X, T, Y)
        self._propensity_model, params = self._propensity(X, T)
        if params:
            self.best_params_["propensity"] = params
        propensity = self._propensity_model.predict_proba(X)[:, 1].clip(1e-3, 1 - 1e-3)
        for treatment in (0, 1):
            mask = T == treatment
            model, params = self._regressor(X[mask], Y[mask])
            probability = propensity if treatment else 1 - propensity
            weights = (mask / probability).astype(np.float64)
            model.fit(X, Y, sample_weight=weights)
            setattr(self, f"_outcome_model_{treatment}", model)
            if params:
                self.best_params_[f"outcome_{treatment}"] = params
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        tau = self._outcome_model_1.predict(X) - self._outcome_model_0.predict(X)
        return tau.reshape(-1), None, None


METALEARNER_WRAPPERS = {
    "S-learner": SLearnerWrapper,
    "T-learner": TLearnerWrapper,
    "X-learner": XLearnerWrapper,
    "Debiased ML": DebiasedMLWrapper,
    "IPW": IPWWrapper,
    "DR (Doubly Robust)": DRWrapper,
}
