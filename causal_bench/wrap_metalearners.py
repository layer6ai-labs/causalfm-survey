"""Wrappers for traditional causal estimators, with optional HPO."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


REGRESSION_ESTIMATORS = (
    "lgbm",
    "xgboost",
    "xgb_limitdepth",
    "rf",
    "kneighbor",
    "extra_tree",
)
PROPENSITY_ESTIMATORS = REGRESSION_ESTIMATORS + ("lrl1", "lrl2")


@dataclass(frozen=True)
class HPOConfig:
    """CausalPFN's FLAML configuration for EconML nuisance models.

    ``time_budget`` is the budget for *each* AutoML nuisance-model search,
    not for an entire causal estimator fit. This matches the reference
    CausalPFN benchmark implementation.
    """

    time_budget: float = 900.0
    cv: int = 3
    verbose: int = 0
    early_stop: bool = True


def _arrays(X, T, Y):
    X = np.asarray(X, dtype=np.float32)
    T = np.asarray(T, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array; got shape {X.shape}")
    if T.ndim > 2 or (T.ndim == 2 and T.shape[1] != 1):
        raise ValueError(f"T must be one-dimensional; got shape {T.shape}")
    if Y.ndim > 2 or (Y.ndim == 2 and Y.shape[1] != 1):
        raise ValueError(f"Y must be one-dimensional; got shape {Y.shape}")

    T = T.reshape(-1)
    Y = Y.reshape(-1)
    if not (len(X) == len(T) == len(Y)):
        raise ValueError(
            "X, T, and Y must have the same number of rows; "
            f"got {len(X)}, {len(T)}, and {len(Y)}"
        )
    if len(X) == 0:
        raise ValueError("X, T, and Y must not be empty")
    if not (np.isfinite(X).all() and np.isfinite(T).all() and np.isfinite(Y).all()):
        raise ValueError("X, T, and Y must contain only finite values")
    treatments = np.unique(T)
    if not np.array_equal(treatments, np.array([0.0, 1.0], dtype=np.float32)):
        raise ValueError(
            "These wrappers require both binary treatment arms encoded as 0 and 1; "
            f"got {treatments.tolist()}"
        )
    return X, T, Y


def _tune_with_flaml(X, y, task, estimator_list, config):
    """Run the same FLAML search used by CausalPFN's EconML baselines."""
    try:
        from flaml import AutoML
    except ImportError as exc:
        raise ImportError(
            "hpo=True requires FLAML. Install it with "
            "`uv pip install 'FLAML[automl]==2.3.5'` or "
            "`pip install 'FLAML[automl]==2.3.5'`."
        ) from exc

    from sklearn.base import clone

    if not np.isfinite(config.time_budget) or config.time_budget <= 0:
        raise ValueError("HPO time_budget must be a positive number of seconds")
    if not isinstance(config.cv, (int, np.integer)) or config.cv < 2:
        raise ValueError("HPO cv must be an integer of at least 2")

    y = np.asarray(y).reshape(-1)
    split_limit = len(y)
    if task == "classification":
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("Classification HPO requires at least two classes")
        split_limit = int(counts.min())
    n_splits = min(config.cv, split_limit)
    if n_splits < 2:
        raise ValueError("HPO requires at least two observations in every CV population")

    automl = AutoML()
    automl.fit(
        X_train=X,
        y_train=y,
        time_budget=config.time_budget,
        task=task,
        eval_method="cv",
        n_splits=n_splits,
        verbose=config.verbose,
        estimator_list=list(estimator_list),
        early_stop=config.early_stop,
    )
    if automl.model is None or not hasattr(automl.model, "estimator"):
        raise RuntimeError("FLAML completed without producing a cloneable best estimator")

    params = {
        "estimator": automl.best_estimator,
        "config": dict(automl.best_config),
        "n_splits": n_splits,
    }
    # CausalPFN clones FLAML's selected estimator so EconML performs the final
    # fit itself instead of reusing the model fitted during the search.
    return clone(automl.model.estimator), params


class _BaseWrapper:
    def __init__(self, model=None, device="cpu", hpo=False, hpo_config=None):
        self.model = model
        self.device = device
        self.hpo = hpo
        self.hpo_config = hpo_config or HPOConfig()
        self.best_params_ = {}
        self._learner = None
        self._x_scaler = None
        self._y_scaler = None

    def _prepare_fit_data(self, X, T, Y, *, scale=True):
        """Validate data and reproduce the reference benchmark preprocessing."""
        self.best_params_.clear()
        self._learner = None
        self._x_scaler = None
        self._y_scaler = None
        self.__dict__.pop("ate_", None)
        self.__dict__.pop("_propensity_model", None)
        X, T, Y = _arrays(X, T, Y)

        if not scale:
            return X, T, Y

        from sklearn.preprocessing import StandardScaler

        self._x_scaler = StandardScaler().fit(X)
        self._y_scaler = StandardScaler().fit(Y.reshape(-1, 1))
        X = self._x_scaler.transform(X).astype(np.float32, copy=False)
        Y = self._y_scaler.transform(Y.reshape(-1, 1)).reshape(-1).astype(
            np.float32,
            copy=False,
        )
        return X, T, Y

    def _transform_x(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array; got shape {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if self._x_scaler is not None:
            X = self._x_scaler.transform(X).astype(np.float32, copy=False)
        return X

    def _unscale_effect(self, effect):
        effect = np.asarray(effect)
        if self._y_scaler is not None:
            effect = effect * self._y_scaler.scale_[0]
        return effect

    def _regressor(self, X, y):
        from sklearn.base import clone
        from sklearn.ensemble import RandomForestRegressor

        if self.hpo:
            return _tune_with_flaml(
                X,
                y,
                "regression",
                REGRESSION_ESTIMATORS,
                self.hpo_config,
            )
        model = clone(self.model) if self.model is not None else RandomForestRegressor(
            n_estimators=100,
            random_state=0,
        )
        return model, None

    def _propensity(self, X, T):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if self.hpo:
            return _tune_with_flaml(
                X,
                T,
                "classification",
                PROPENSITY_ESTIMATORS,
                self.hpo_config,
            )
        return (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "logistic",
                        LogisticRegression(max_iter=2000, random_state=0),
                    ),
                ]
            ),
            None,
        )

    def predict(self, X):
        if self._learner is None:
            raise RuntimeError("fit must be called before predict")
        tau = self._learner.effect(self._transform_x(X)).reshape(-1)
        tau = self._unscale_effect(tau).astype(np.float32, copy=False)
        return tau, None, None

    def estimate_ate(self, X, T, Y):
        """Use EconML's population-level ATE API on the requested covariates."""
        if self._learner is None:
            raise RuntimeError("fit must be called before estimate_ate")
        ate = self._learner.ate(X=self._transform_x(X), T0=0, T1=1)
        ate = np.asarray(self._unscale_effect(ate), dtype=float).reshape(-1)
        if ate.size != 1 or not np.isfinite(ate[0]):
            raise RuntimeError(f"Expected one finite ATE value; got shape {ate.shape}")
        return float(ate[0])

    def run(self, X_train, T_train, Y_train, X_test):
        start = time.time()
        self.fit(X_train, T_train, Y_train)
        tau, lower, upper = self.predict(X_test)
        ate = self.estimate_ate(X_train, T_train, Y_train)
        return tau, lower, upper, ate, time.time() - start


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

        X, T, Y = self._prepare_fit_data(X, T, Y)
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

        X, T, Y = self._prepare_fit_data(X, T, Y)
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

        X, T, Y = self._prepare_fit_data(X, T, Y)
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

        X, T, Y = self._prepare_fit_data(X, T, Y)
        model_y, params_y = self._regressor(X, Y)
        model_t, params_t = self._propensity(X, T)
        if params_y:
            self.best_params_["outcome"] = params_y
        if params_t:
            self.best_params_["treatment"] = params_t
        self._learner = DML(
            model_y=model_y,
            model_t=model_t,
            model_final=LinearRegression(fit_intercept=False),
            discrete_treatment=True,
        ).fit(Y, T, X=X)
        return self


class IPWWrapper(_BaseWrapper):
    name = "IPW"
    supports_cate = False

    def __init__(self, device="cpu", hpo=False, hpo_config=None):
        super().__init__(device=device, hpo=hpo, hpo_config=hpo_config)

    @classmethod
    def is_available(cls):
        return True

    def fit(self, X, T, Y):
        # The paper's IPW baseline is separate from EconMLBaseline and uses
        # raw X and Y, while the EconML wrappers above are standardized.
        X, T, Y = self._prepare_fit_data(X, T, Y, scale=False)
        self._propensity_model, params = self._propensity(X, T)
        self._propensity_model.fit(X, T)
        if params:
            self.best_params_["propensity"] = params
        propensity = self._propensity_model.predict_proba(X)[:, 1].clip(1e-3, 1 - 1e-3)
        treated = T == 1
        treated_mean = np.average(Y[treated], weights=1.0 / propensity[treated])
        control_mean = np.average(Y[~treated], weights=1.0 / (1.0 - propensity[~treated]))
        self.ate_ = float(treated_mean - control_mean)
        return self

    def predict(self, X):
        if not hasattr(self, "ate_"):
            raise RuntimeError("fit must be called before predict")
        n_rows = self._transform_x(X).shape[0]
        return np.full(n_rows, self.ate_, dtype=np.float32), None, None

    def estimate_ate(self, X, T, Y):
        if not hasattr(self, "ate_"):
            raise RuntimeError("fit must be called before estimate_ate")
        return float(self.ate_)


class DRWrapper(_BaseWrapper):
    name = "DR (Doubly Robust)"

    def __init__(self, device="cpu", hpo=False, hpo_config=None):
        super().__init__(device=device, hpo=hpo, hpo_config=hpo_config)

    @classmethod
    def is_available(cls):
        try:
            from econml.dr import ForestDRLearner  # noqa: F401
            return True
        except Exception:
            return False

    def fit(self, X, T, Y):
        from econml.dr import ForestDRLearner
        from sklearn.preprocessing import PolynomialFeatures

        X, T, Y = self._prepare_fit_data(X, T, Y)
        model_regression, params_regression = self._regressor(X, Y)
        model_propensity, params_propensity = self._propensity(X, T)
        if params_regression:
            self.best_params_["outcome"] = params_regression
        if params_propensity:
            self.best_params_["propensity"] = params_propensity
        self._learner = ForestDRLearner(
            model_regression=model_regression,
            model_propensity=model_propensity,
            n_estimators=1000,
            cv=5,
            featurizer=PolynomialFeatures(degree=3),
        ).fit(Y, T, X=X)
        return self


METALEARNER_WRAPPERS = {
    "S-learner": SLearnerWrapper,
    "T-learner": TLearnerWrapper,
    "X-learner": XLearnerWrapper,
    "Debiased ML": DebiasedMLWrapper,
    "IPW": IPWWrapper,
    "DR (Doubly Robust)": DRWrapper,
}
