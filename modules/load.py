import joblib
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from typing import Tuple, Iterable, Dict, Any, Optional, List
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_h3o_data(path="H3O+.csv"):
    """
    Load all data from ../H3O+.csv
    :return: list, list
    """
    data = load_csv(path)
    X = data[["R1", "R2", "R3", "R4", "R5", "R6"]].values
    y = data["Energy"].values

    return X, y


def split_h3o_dataset(seed=507, size_train=1000, size_val: int | None = 1000, size_test: int | None = None,)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split H3O+ dataset into training, validating, test datasets
    :param seed: seed for reproducing the same subsets
    :param size_tr: size of training dataset
    :param size_val: size of validation dataset, if None, all remaining go to test.
    :param size_test: size of test dataset, if None, uses all remaining after training and validation subsets.
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    np.random.seed(seed)

    X, y = load_h3o_data()
    N = len(X)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)

    if size_val is None:
        size_val = 0
    if size_test is None:
        size_test = N - size_train - size_val

    cut1 = size_train
    cut2 = size_train + size_val

    train_i = perm[:cut1]
    val_i = perm[cut1:cut2] if size_val > 0 else np.array([], dtype=int)
    test_i = perm[cut2:cut2 + size_test]

    return train_i, val_i, test_i


def load_5d_classification_data(path: str="5D-data-for-linear-classification.csv") -> Tuple[np.ndarray, np.ndarray]:
    data = load_csv(path)
    X = data[["X1", "X2", "X3", "X4", "X5"]].values
    y = data["Class_Label"].values.astype(int)
    return X, y


def split_5d_data(X, y, test_size=0.25, seed=507):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def load_gp_models(path: str | Path = "saved_models/models"):
    """
    - Load trained GP models, normalization scalers, in .joblib format
    - Load indices of selected training, validation, test points in .npy format
    - Load summary data (optional) in JSON format
    :param path: Path of the saved models
    :return: dict[str, GaussianProcessRegressor], sklearn.preprocessing.StandardScaler, sklearn.preprocessing.StandardScaler, np.ndarray, np.ndarray, np.ndarray, dict
    """
    path = Path(path)
    gp_models = {}

    for file in path.glob("gp_*.joblib"):
        name = file.stem.removeprefix("gp_")
        gp_models[name] = joblib.load(file)

    X_scaler = joblib.load(path / "X_scaler.joblib")
    y_scaler = joblib.load(path / "y_scaler.joblib")
    train_i = np.load(path / "train_i.npy")
    val_i = np.load(path / "val_i.npy")
    test_i = np.load(path / "test_i.npy")
    with open(path / "meta.json") as file:
        meta = json.load(file)

    return gp_models, X_scaler, y_scaler, train_i, val_i, test_i, meta


def scale_dataset(X: np.ndarray, y: np.ndarray, X_scaler: StandardScaler, y_scaler: StandardScaler):
    """
    Transform H3O+ validation/test data points by rescaling their mean to 0 and standard deviation to 1.
    :param: X
    :param: y
    :param: X_scaler
    :param: y_scaler
    :return: X_scaled: ndarray of shape (1000, 6), standardized coordinates (R1–R6).
             y_scaled: ndarray of shape (1000,), standardized potential energy values.
    """
    X_scaled = X_scaler.transform(X)
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()

    return X_scaled, y_scaled


def fix_splits_h3o(size_total: int, array_size_train: Iterable[int] = (60, 600, 1000, 2000), size_val: int = 1000, size_test: int = 3000, seed: int = 507) \
        -> Tuple[Dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    array_size_train = tuple(sorted(array_size_train))
    max_train = array_size_train[-1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(size_total)

    remaining = size_total - max_train
    if remaining < 0:
        raise ValueError(f"max(array_size_train)={max_train} exceeds dataset size={size_total}.")
    if remaining < (size_val + size_test):
        raise ValueError(f"Insufficient remaining data points ({remaining}) required for validation ({size_val}) and test sets ({size_test}).")

    val_start = max_train
    val_stop = val_start + size_val
    test_start = val_stop
    test_stop = test_start + size_test

    val_i = perm[val_start:val_stop]
    test_i = perm[test_start:test_stop]
    dict_train_i = {size_train: perm[:size_train] for size_train in array_size_train}

    return dict_train_i, val_i, test_i, perm


def load_krr_models(path: str | Path = "saved_models/krr_models", ns: Optional[list[int]] = None) \
        -> Tuple[Dict[int, Any], Dict[int, Any], Dict[int, Any], Dict[int, np.ndarray], np.ndarray, np.ndarray, Dict[int, Dict[str, Any]], Dict[str, Any]]:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    val_path = path / "val_i.npy"
    test_path = path / "test_i.npy"
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("val_i.npy or test_i.npy is missing in the model directory.")

    val_i = np.load(val_path, allow_pickle=False)
    test_i = np.load(test_path, allow_pickle=False)

    summary_path = path / "summary.json"
    meta_path = path / "meta.json"
    summary_table: Dict[int, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}

    if summary_path.exists():
        with open(summary_path) as file:
            raw = json.load(file)
            summary_table = {int(k): v for k, v in raw.items()}
    if meta_path.exists():
        with open(meta_path) as file:
            meta = json.load(file)

    models_by_n: Dict[int, Any] = {}
    X_scalers_by_n: Dict[int, Any] = {}
    y_scalers_by_n: Dict[int, Any] = {}
    dict_train_i: Dict[int, np.ndarray] = {}

    pattern = re.compile(r"krr_n(\d+)\.joblib$")
    found_ns: list[int] = []
    for file in path.glob("krr_n*.joblib"):
        m = pattern.search(file.name)
        if m:
            found_ns.append(int(m.group(1)))

    if not found_ns:
        raise FileNotFoundError(f"No models found in {path} (expected files like krr_n{ '{n}' }.joblib).")

    target_ns = sorted(set(found_ns) & set(ns)) if ns is not None else sorted(found_ns)
    if ns is not None:
        missing = sorted(set(ns) - set(found_ns))
        if missing:
            raise FileNotFoundError(f"Requested n not found: {missing}. Available: {sorted(found_ns)}")

    for n in target_ns:
        model_path = path / f"krr_n{n}.joblib"
        xsc_path   = path / f"X_scaler_n{n}.joblib"
        ysc_path   = path / f"y_scaler_n{n}.joblib"
        tri_path   = path / f"train_i_n{n}.npy"

        if not (model_path.exists() and xsc_path.exists() and ysc_path.exists() and tri_path.exists()):
            raise FileNotFoundError(
                f"Missing one or more files for n={n}: "
                f"{model_path.name}, {xsc_path.name}, {ysc_path.name}, {tri_path.name}"
            )

        models_by_n[n] = joblib.load(model_path)
        X_scalers_by_n[n] = joblib.load(xsc_path)
        y_scalers_by_n[n] = joblib.load(ysc_path)
        dict_train_i[n] = np.load(tri_path, allow_pickle=False)

    return models_by_n, X_scalers_by_n, y_scalers_by_n, dict_train_i, val_i, test_i, summary_table, meta


def load_nn_models(path: str | Path = "saved_models/nn_models", ns: Optional[List[int]] = None, compile_models: bool = False, custom_objects: Optional[Dict[str, Any]] = None)\
        -> Tuple[Dict[int, tf.keras.Model],Dict[int, Any], Dict[int, Any], Dict[int, np.ndarray], np.ndarray, np.ndarray, Dict[int, Dict[str, Any]], Dict[str, Any]]:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")

    val_path = path / "val_i.npy"
    test_path = path / "test_i.npy"
    if not val_path.exists() or not test_path.exists():
        raise FileNotFoundError("val_i.npy or test_i.npy is missing in the model directory.")
    val_i = np.load(val_path, allow_pickle=False)
    test_i = np.load(test_path, allow_pickle=False)


    summary_table: Dict[int, Dict[str, Any]] = {}
    meta: Dict[str, Any] = {}

    summary_path = path / "summary.json"
    if summary_path.exists():
        with open(summary_path) as file:
            raw = json.load(file)
            summary_table = {int(k): v for k, v in raw.items()}

    m_path = path / "meta.json"
    if m_path.exists():
        with open(m_path) as file:
            meta = json.load(file)

    pattern = re.compile(r"nn_n(\d+)\.keras$")
    found_ns: List[int] = []
    for f in path.glob("nn_n*.keras"):
        m = pattern.search(f.name)
        if m:
            found_ns.append(int(m.group(1)))

    if not found_ns:
        raise FileNotFoundError(f"No NN models found in {path} (expected files like nn_n{{n}}.keras).")

    target_ns = sorted(found_ns) if ns is None else sorted(set(found_ns) & set(ns))
    if ns is not None:
        missing = sorted(set(ns) - set(found_ns))
        if missing:
            raise FileNotFoundError(f"Requested n not found: {missing}. Available: {sorted(found_ns)}")

    models_by_n: Dict[int, tf.keras.Model] = {}
    X_scalers_by_n: Dict[int, Any] = {}
    y_scalers_by_n: Dict[int, Any] = {}
    dict_train_i: Dict[int, np.ndarray] = {}

    for n in target_ns:
        model_fp = path / f"nn_n{n}.keras"
        xsc_fp   = path / f"X_scaler_n{n}.joblib"
        ysc_fp   = path / f"y_scaler_n{n}.joblib"
        tri_fp   = path / f"train_i_n{n}.npy"

        if not (model_fp.exists() and xsc_fp.exists() and ysc_fp.exists() and tri_fp.exists()):
            raise FileNotFoundError(
                f"Missing one or more files for n={n}: "
                f"{model_fp.name}, {xsc_fp.name}, {ysc_fp.name}, {tri_fp.name}"
            )

        models_by_n[n] = tf.keras.models.load_model(
            model_fp, compile=compile_models, custom_objects=custom_objects
        )
        X_scalers_by_n[n] = joblib.load(xsc_fp)
        y_scalers_by_n[n] = joblib.load(ysc_fp)
        dict_train_i[n] = np.load(tri_fp, allow_pickle=False)

    return models_by_n, X_scalers_by_n, y_scalers_by_n, dict_train_i, val_i, test_i, summary_table, meta


def load_pca_model(path: str | Path = "models/pca_model"):
    path = Path(path)

    model = joblib.load(path / "pca_model.joblib")
    Sigma = np.load(path / "Sigma.npy")
    explained_variance_ratio = np.load(path / "explained_variance_ratio.npy")
    cumulative_evr = np.load(path / "cumulative_evr.npy")

    with open(path / "meta.json", "r") as file:
        meta = json.load(file)

    print(f"PCA model loaded from {path}")
    return model, Sigma, explained_variance_ratio, cumulative_evr, meta


def load_svm_model(path: str | Path = "models/svm_model"):
    path = Path(path)

    best_model = joblib.load(path / "best_model.joblib")

    cv_results_path = path / "cv_results.csv"
    cv_results = pd.read_csv(cv_results_path) if cv_results_path.exists() else None

    w_path = path / "linear_weights.npy"
    b_path = path / "linear_intercept.npy"
    linear_weights = np.load(w_path) if w_path.exists() else None
    linear_intercept = np.load(b_path) if b_path.exists() else None

    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None

    return best_model, cv_results, linear_weights, linear_intercept, summary
