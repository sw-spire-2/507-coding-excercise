from typing import Dict, Any, Optional

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA


def summarize_gp_models(gp_models, n=1000):
    """
    Summarize trained GP models in tabular form
    :param gp_models: dict[str, GaussianProcessRegressor]
    :param n: for BIC calculations, default set to 1000 (size of randomly selected H3O+ PES subset)
    :return: pd.DataFrame, sorted by BIC (lower is better)
    """
    rows = []
    for name, gp in gp_models.items():
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        n_param = gp.kernel_.theta.size
        row = {"Kernel": name,
               "Params": n_param,
               "LML": lml,
               "Optimized Kernel": str(gp.kernel_),
               "BIC": n_param * np.log(n) - 2.0 * lml
               }
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("BIC", ascending=True).reset_index(drop=True)
    summary["LML"] = summary["LML"].round(3)
    summary["BIC"] = summary["BIC"].round(2)
    return summary


def display_summary(summary: pd.DataFrame) -> None:
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)

    print(summary.to_string(index=False))


def save_csv(summary: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)


def save_gp_models(models: dict, X_scaler, y_scaler, train_i: np.ndarray, val_i: np.ndarray, test_i, meta: dict | None,
                   path: str | Path = "saved_models/models",):
    """
    - Save trained GP models, normalization scalers, in .joblib format
    - Save indices of selected training points in .npy format
    - Export summary data (optional) in JSON format
    :param path: str or Path, path to be saved
    :param models: dict[str, GaussianProcessRegressor], trained GP models
    :param X_scaler: sklearn.preprocessing.StandardScaler, X_scaler for normalization
    :param y_scaler: sklearn.preprocessing.StandardScaler, y_scaler for normalization
    :param train_i: indices selected as training set
    :param val_i: indices selected as validation set
    :param test_i: indices selected as test set
    :param meta: dict, summary of training parameters to be exported into JSON format
    :return: None
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, path / f"gp_{name}.joblib")

    joblib.dump(X_scaler, path / "X_scaler.joblib")
    joblib.dump(y_scaler, path / "y_scaler.joblib")
    np.save(path / "train_i.npy", train_i)
    np.save(path / "val_i.npy", val_i)
    np.save(path / "test_i.npy", test_i)

    meta = dict(meta) if meta else {}
    meta.setdefault("saved_at", datetime.now().isoformat())
    with open(path / "meta.json", "w") as file:
        json.dump(meta, file, indent=2)

    print(f"Saved {len(models)} models to {path}")


def save_krr_models(summary: Dict[int, Dict[str, Any]], path: str | Path = "saved_models/krr_models", meta: Optional[dict] = None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if not summary:
        print(f"- No KRR models to save. Exit.")
        return

    first = next(iter(summary.values()))
    val_i = first["val_indices"]
    test_i = first["test_indices"]
    np.save(path / "val_i.npy", val_i)
    np.save(path / "test_i.npy", test_i)

    for n, record in summary.items():
        joblib.dump(record["model"], path / f"krr_n{n}.joblib")
        joblib.dump(record["X_scaler"], path / f"X_scaler_n{n}.joblib")
        joblib.dump(record["y_scaler"], path / f"y_scaler_n{n}.joblib")
        np.save(path / f"train_i_n{n}.npy", record["train_indices"])

    summary_table = {
        n: {
            "best_params": rec["best_params"],
            "cv_best_mse": rec["cv_best_mse"],
            "test_RMSE": rec["test_RMSE"],
            "test_r2": rec["test_r2"],
            "total_fit_time_sec": rec["total_fit_time_sec"],
            "predict_time_sec": rec["predict_time_sec"],
        }
        for n, rec in summary.items()
    }

    with open(path / "summary.json", "w") as file:
        json.dump(summary_table, file, indent=2)

    meta = dict(meta) if meta else {}
    meta.setdefault("saved_at", datetime.now().isoformat())
    meta.setdefault("num_models", len(summary))
    with open(path / "meta.json", "w") as file:
        json.dump(meta, file, indent=2)

    print(f"- Saved {len(summary)} KRR models to: {path}")


def save_nn_models(summary: Dict[int, Dict[str, Any]],path: str | Path = "saved_models/nn_models",meta: Optional[dict] = None,) -> None:

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    if not summary:
        print(f"- No NN models to save. Exit.")
        return

    first = next(iter(summary.values()))
    np.save(path / "val_i.npy", first["val_indices"])
    np.save(path / "test_i.npy", first["test_indices"])

    for n, rec in summary.items():
        model_path = path / f"nn_n{n}.keras"
        rec["model"].save(model_path)

        joblib.dump(rec["X_scaler"], path / f"X_scaler_n{n}.joblib")
        joblib.dump(rec["y_scaler"], path / f"y_scaler_n{n}.joblib")
        np.save(path / f"train_i_n{n}.npy", rec["train_indices"])

    summary_table = {
        int(n): {
            "best_params":          rec.get("best_params", {}),
            "nn_training":          rec.get("nn_training", {}),
            "val_RMSE":             rec.get("val_RMSE"),
            "val_r2":               rec.get("val_r2"),
            "test_RMSE":            rec.get("test_RMSE"),
            "test_r2":              rec.get("test_r2"),
            "total_fit_time_sec":   rec.get("total_fit_time_sec"),
            "predict_time_sec":     rec.get("predict_time_sec"),
        }
        for n, rec in summary.items()
    }
    with open(path / "summary.json", "w") as file:
        json.dump(summary_table, file, indent=2)

    meta_out = dict(meta) if meta else {}
    meta_out.setdefault("saved_at", datetime.now().isoformat())
    meta_out.setdefault("num_models", len(summary))
    with open(path / "meta.json", "w") as file:
        json.dump(meta_out, file, indent=2)

    print(f"- Saved {len(summary)} NN models to: {path}")


def save_pca_model(model: PCA, L: int, rmse: float, explained_variance_ratio: np.ndarray, cumulative_evr: np.ndarray,
                   Sigma: np.ndarray, path: str | Path = "models/pca_model", meta: dict | None = None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path / "pca_model.joblib")
    np.save(path / "Sigma.npy", Sigma)
    np.save(path / "explained_variance_ratio.npy", explained_variance_ratio)
    np.save(path / "cumulative_evr.npy", cumulative_evr)

    meta = dict(meta) if meta else {}
    meta.update({
        "L": L,
        "rmse": float(rmse),
        "timestamp": datetime.now().isoformat(),
        "n_components": model.n_components_,
    })
    with open(path / "meta.json", "w") as file:
        json.dump(meta, file, indent=2)

    print(f"PCA model and related files saved to {path}")


def save_svm_model(grid, best_model, acc, prec, rec, f1, auc, path: str | Path = "models/svm_model",
    class_counts: dict | None = None, seed: int = 507, refit_metric: str = "f1"):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, path / "best_model.joblib")
    cv_data = pd.DataFrame(grid.cv_results_).copy()

    for column in [
        "param_svc__kernel", "param_svc__C", "param_svc__gamma",
        "mean_test_accuracy", "mean_test_precision", "mean_test_recall",
        "mean_test_f1", "mean_test_roc_auc", "rank_test_" + grid.refit
    ]:
        if column not in cv_data.columns:
            cv_data[column] = np.nan

    cv_data["param_svc__gamma"] = cv_data["param_svc__gamma"].astype(object)
    cv_data.loc[cv_data["param_svc__kernel"] == "linear", "param_svc__gamma"] = "-"
    cv_data.sort_values("rank_test_" + grid.refit, inplace=True)

    cv_data.to_csv(path / "cv_results.csv", index=False)

    class_counts = dict(class_counts or {})
    class_counts_total = class_counts.get("class_counts_total")
    class_counts_train = class_counts.get("class_counts_train")
    class_counts_test = class_counts.get("class_counts_test")

    summary_out = dict({})
    summary_out.update({
        "timestamp": datetime.now().isoformat(),
        "seed": int(seed),
        "refit_metric": str(refit_metric),
        "best_params": {k: (v if not hasattr(v, "item") else v.item()) for k, v in grid.best_params_.items()},
        "best_cv_score": float(grid.best_score_),
        "test_metrics": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc),
        },
        "class_counts": class_counts,
        "scorer_keys": list(grid.scorer_.keys()) if hasattr(grid, "scorer_") else None,
        "n_splits": getattr(grid.cv, "n_splits", None),
    })

    with open(path / "summary.json", "w") as file:
        json.dump(summary_out, file, indent=2)

    svc = best_model.named_steps["svc"]
    if getattr(svc, "kernel", None) == "linear":
        np.save(path / "linear_weights.npy", svc.coef_.ravel())
        np.save(path / "linear_intercept.npy", svc.intercept_)
        print("Saved linear weights and intercept.")

    print(f"SVM model and related files saved to {path}")

