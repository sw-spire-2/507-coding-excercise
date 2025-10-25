import time
from typing import Tuple, Dict, Any, Sequence
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import layers, regularizers, callbacks, Sequential
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


def _build_mlp(input_dim: int, hidden_layers: Sequence[int] = (128, 64),
               activation: str = "relu", l2_weight: float = 0.0, dropout: float = 0.0,) -> tf.keras.Model:
    model = Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for h in hidden_layers:
        model.add(layers.Dense(h, activation=activation,
                               kernel_regularizer=regularizers.l2(l2_weight) if l2_weight > 0 else None))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1))
    return model


def _train_nn(X_train_scaled: np.ndarray, y_train_scaled: np.ndarray, X_val_scaled: np.ndarray, y_val_scaled: np.ndarray,
              hidden_layers: Sequence[int] = (128, 64), activation: str = "relu", l2_weight: float = 0.0,
              dropout: float = 0.0, lr: float = 1e-3, epochs: int = 2000, batch_size: int = 128,patience: int = 100) \
        -> Tuple[tf.keras.Model, Dict[str, Any], float]:
    model = _build_mlp(X_train_scaled.shape[1], hidden_layers, activation, l2_weight, dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"]
    )

    es = callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0,
    )
    fit_time = time.time() - t0

    training_info = {
        "best_val_loss": float(np.min(hist.history["val_loss"])),
        "epochs_trained": int(len(hist.history["loss"])),
    }
    return model, training_info, fit_time


def _time_predict(model: tf.keras.Model, X: np.ndarray) -> Tuple[np.ndarray, float]:
    t0 = time.time()
    yhat = model.predict(X, verbose=0).ravel()
    pred_time = time.time() - t0
    return yhat, pred_time


def train_nn_fix_split(X: np.ndarray, y: np.ndarray, dict_train_i: Dict[int, np.ndarray], val_i: np.ndarray, test_i: np.ndarray,
    hidden_layers: Sequence[int] = (128, 64), activation: str = "relu", l2_weight: float = 0.0, dropout: float = 0.0, lr: float = 1e-3,
    epochs: int = 2000, batch_size: int = 128, patience: int = 100, inverse_transform: bool = True, seed: int = 507) -> Dict[int, Dict[str, Any]]:
    summary: Dict[int, Dict[str, Any]] = {}

    X_val, y_val = X[val_i],  y[val_i]
    X_test, y_test = X[test_i], y[test_i]

    tf.keras.utils.set_random_seed(seed)

    for n, train_i in dict_train_i.items():
        X_train, y_train = X[train_i], y[train_i]

        # scale (fit on train only)
        X_scaler, y_scaler = _get_scaler(X_train, y_train)
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled = \
            _normalize(X_train, y_train, X_val, y_val, X_test, y_test, X_scaler, y_scaler)

        X_train_scaled = X_train_scaled.astype(np.float32)
        y_train_scaled = y_train_scaled.astype(np.float32)
        X_val_scaled = X_val_scaled.astype(np.float32)
        y_val_scaled = y_val_scaled.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)

        # train NN
        model, tr_info, fit_time = _train_nn(
            X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
            hidden_layers=hidden_layers,
            activation=activation,
            l2_weight=l2_weight,
            dropout=dropout,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )

        y_hat_test_s, predict_time = _time_predict(model, X_test_scaled)
        eval_test = evaluate_fix_split_train(
            y_test=y_test,
            y_hat_scaled=y_hat_test_s,
            y_scaler=y_scaler,
            inverse_transform=inverse_transform,
        )

        y_hat_val_s, _ = _time_predict(model, X_val_scaled)
        eval_val = evaluate_fix_split_train(
            y_test=y_val,
            y_hat_scaled=y_hat_val_s,
            y_scaler=y_scaler,
            inverse_transform=inverse_transform,
        )

        summary[n] = {
            "model": model,
            "X_scaler": X_scaler,
            "y_scaler": y_scaler,
            "best_params": {
                "hidden_layers": list(hidden_layers),
                "activation": activation,
                "l2_weight": float(l2_weight),
                "dropout": float(dropout),
                "lr": float(lr),
                "batch_size": int(batch_size),
                "patience": int(patience),
            },
            "cv_best_mse": None,
            "cv_time_sec": 0.0,
            "refit_time_sec": float(fit_time),
            "total_fit_time_sec": float(fit_time),
            "predict_time_sec": float(predict_time),
            "test_RMSE": float(eval_test["RMSE"]),
            "test_r2": float(eval_test["r2"]),
            "inverse_transformed": bool(eval_test["inverse_transformed"]),
            "val_RMSE": float(eval_val["RMSE"]),
            "val_r2": float(eval_val["r2"]),
            "train_indices": train_i,
            "val_indices": val_i,
            "test_indices": test_i,
            "nn_training": tr_info,
        }

    return summary
