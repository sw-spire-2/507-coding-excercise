import time
import numpy as np
from typing import Tuple, Dict, Optional, Iterable, Any
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from modules.evaluation import evaluate_fix_split_train


def _get_scaler(X: np.ndarray, y: np.ndarray) -> Tuple[StandardScaler, StandardScaler]:
    X_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y.reshape(-1, 1))

    return X_scaler, y_scaler


def _normalize(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray, X_scaler: StandardScaler, y_scaler: StandardScaler) -> Tuple[np.ndarray, ...]:
    X_train_scaled = X_scaler.transform(X_train)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()

    X_val_scaled = X_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    X_test_scaled = X_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled


def train_krr_cv(X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, cv_grid: Optional[Dict[str, Iterable]] = None,
                 n_cv: int = 3, n_jobs: int = -1, refit: bool = False) -> Tuple[GridSearchCV, float]:
    if cv_grid is None:
        cv_grid = {
            "kernel": ["rbf"],
            "alpha": np.logspace(-7, 0, 8),
            "gamma": np.logspace(-2, 2, 9),
        }

    model = GridSearchCV(
        KernelRidge(),
        param_grid=cv_grid,
        scoring="neg_mean_squared_error",
        cv=n_cv,
        n_jobs=n_jobs,
        refit=refit,
    )

    t0 = time.time()
    model.fit(X_train_scaled, y_train_scaled)
    cv_time = time.time() - t0

    return model, cv_time


def refit_model(best_params: Dict[str, Any], X_train_scaled: np.ndarray, y_train_scaled: np.ndarray) -> Tuple[KernelRidge, float]:
    model = KernelRidge(**best_params)
    t0 = time.time()
    model.fit(X_train_scaled, y_train_scaled)
    refit_time = time.time() - t0

    return model, refit_time


def time_predict(model, X: np.ndarray) -> Tuple[np.ndarray, float]:
    t0 = time.time()
    yhat = model.predict(X)
    pred_time = time.time() - t0

    return yhat, pred_time


def train_krr_fix_split(X: np.ndarray, y: np.ndarray, dict_train_i: Dict[int, np.ndarray], val_i: np.ndarray, test_i: np.ndarray,
                        cv_grid: Optional[Dict[str, Iterable]] = None, n_cv: int = 3, inverse_transform: bool = True) \
        -> Dict[int, Dict[str, Any]]:
    summary: Dict[int, Dict[str, Any]] = {}

    X_val, y_val = X[val_i], y[val_i]
    X_test, y_test = X[test_i], y[test_i]

    for n, train_i in dict_train_i.items():
        X_train, y_train = X[train_i], y[train_i]
        X_scaler, y_scaler = _get_scaler(X_train, y_train)

        X_train_scaled, y_train_scaled, _, _, X_test_scaled, y_test_scaled = _normalize(X_train, y_train, X_val, y_val, X_test, y_test, X_scaler, y_scaler)

        search, cv_time = train_krr_cv(X_train_scaled, y_train_scaled, cv_grid, n_cv, -1, False)
        best_params = search.best_params_
        cv_best_mse = -float(search.best_score_)

        best_model, refit_time = refit_model(best_params, X_train_scaled, y_train_scaled)

        y_hat_scaled, predict_time = time_predict(best_model, X_test_scaled)

        evaluation_summary = evaluate_fix_split_train(y_test, y_hat_scaled, y_scaler, inverse_transform)

        summary[n] = {
            "model": best_model,
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
            "best_params": best_params,
            "cv_best_mse": cv_best_mse,
            "cv_time_sec": float(cv_time),
            "refit_time_sec": float(refit_time),
            "total_fit_time_sec": float(cv_time + refit_time),
            "predict_time_sec": float(predict_time),
            "test_RMSE": evaluation_summary["RMSE"],
            "test_r2": evaluation_summary["r2"],
            "inverse_transformed": evaluation_summary["inverse_transformed"],
            "train_indices": train_i,
            "val_indices": val_i,
            "test_indices": test_i,
        }

    return summary
