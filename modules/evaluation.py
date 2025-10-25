import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from typing import Optional

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay, classification_report
)

def _scale(X, y, X_scaler=None, y_scaler=None):
    X_scaled = X_scaler.transform(X) if X_scaler is not None else X
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel() if y_scaler is not None else y
    return X_scaled, y_scaled


def calculate_lml(model) -> float:
    return float(model.log_marginal_likelihood(model.kernel_.theta))


def calculate_bic(model, size) -> float:
    lml = calculate_lml(model)
    n_param = model.kernel_.theta.size
    return float(n_param * np.log(size) - 2.0 * lml)


def calculate_rmse(model, X, y, X_scaler: Optional[object] = None, y_scaler: Optional[object] = None, inverse_transform: bool = False) -> float:
    X_scaled, y_scaled = _scale(X, y, X_scaler, y_scaler)
    y_hat_scaled = model.predict(X_scaled)

    if inverse_transform:
        y_hat = y_scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(y, y_hat))
    else:
        rmse = np.sqrt(mean_squared_error(y_scaled, y_hat_scaled))

    return float(rmse)


def calculate_r2(model, X, y, X_scaler: Optional[object] = None, y_scaler: Optional[object] = None, inverse_transform: bool = False) -> float:
    X_scaled, y_scaled = _scale(X, y, X_scaler, y_scaler)
    y_hat_scaled = model.predict(X_scaled)

    if inverse_transform:
        y_hat = y_scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).ravel()
        r2 = r2_score(y, y_hat)
    else:
        r2 = r2_score(y_scaled, y_hat_scaled)

    return float(r2)


def evaluate_gp_models(gp_models, size_train: int, X: np.ndarray, y: np.ndarray, X_scaler, y_scaler, inverse_transform: bool = False) -> pd.DataFrame:
    """
    :param gp_models: dict[str, sklearn.gaussian_process.GaussianProcessRegressor]
    :param size_train: size of the training points for BIC calculation
    :param X: validation or test dataset (untransformed)
    :param y: validation or test dataset (in untransformed form)
    :param X_scaler: scaler used for normalize X
    :param y_scaler: scaler used for normalize Y
    :param inverse_transform: whether inverse transform y hat for calculating RMSE and R2 (i.e. in physical units)
    :return: pd.DataFrame
    - Columns: ["Kernel", "Params", "LML", "BIC", "RMSE", "R2", "Optimized Kernel"]
    - Sorted by RMSE (ascending)
    """
    rows = []
    for name, model in gp_models.items():
        lml = calculate_lml(model)
        bic = calculate_bic(model, size_train)
        kpar = model.kernel_.theta.size
        rmse = calculate_rmse(model, X, y, X_scaler, y_scaler, inverse_transform)
        r2 = calculate_r2(model, X, y, X_scaler, y_scaler, inverse_transform)

        rows.append({
            "Kernel": name,
            "Params": int(kpar),
            "LML": float(lml),
            "BIC": float(bic),
            "RMSE": float(rmse),
            "R2": float(r2),
            "Optimized Kernel": str(model.kernel_),
        })

    summary = pd.DataFrame(rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    return summary


def evaluate_gp_models_training(models: dict[str, sklearn.gaussian_process.GaussianProcessRegressor], X_val: np.ndarray, y_val: np.ndarray,
                                X_scaler, y_scaler, size: int, inverse_transform: bool, level: int) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        rmse = calculate_rmse(model, X_val, y_val, X_scaler, y_scaler, inverse_transform)
        r2 = calculate_r2(model, X_val, y_val, X_scaler, y_scaler, inverse_transform)
        bic = calculate_bic(model, size)
        lml = calculate_lml(model)
        kpar = int(model.kernel_.theta.size)

        rows.append({
            "Level": level,
            "Kernel": name if level == 0 else f"L{level}: {name}",
            "Params": kpar,
            "LML": lml,
            "BIC": bic,
            "RMSE": rmse,
            "R2": r2,
            "Optimized Kernel": str(model.kernel_)
        })
    summary = pd.DataFrame(rows)
    return summary


def evaluate_fix_split_train(y_test: np.ndarray, y_hat_scaled: np.ndarray, y_scaler: StandardScaler, inverse_transform: bool = False):
    if inverse_transform:
        y_hat = y_scaler.inverse_transform(y_hat_scaled.reshape(-1, 1)).ravel()
        rmse = float(np.sqrt(mean_squared_error(y_test, y_hat)))
        r2 = float(r2_score(y_test, y_hat))
    else:
        y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(y_test_scaled, y_hat_scaled))
        r2 = float(r2_score(y_test_scaled, y_hat_scaled))

    return {"RMSE": rmse, "r2": r2, "inverse_transformed": inverse_transform}


def evaluate_pca_L(L: int, model: PCA, X_center: np.ndarray) -> float:
    Z = model.transform(X_center)[:, :L]
    X_hat = (Z @ model.components_[:L, :])
    rmse = np.sqrt(np.mean((X_center - X_hat) ** 2))

    return rmse


def evaluate_svm(best_model, X_test, y_test, title_prefix="Best SVM"):
    y_pred = best_model.predict(X_test)

    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(X_test)[:, 1]
    else:
        y_score = best_model.decision_function(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)

    print(f"\n{title_prefix} — Test Set Performance")
    print("-" * 50)
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}\n")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    return acc, prec, rec, f1, auc
