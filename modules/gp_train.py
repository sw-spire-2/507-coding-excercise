import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, DotProduct,
    WhiteKernel, ConstantKernel as C
)
from sklearn.metrics import mean_squared_error, r2_score
from modules.load import split_h3o_dataset, load_h3o_data
from modules.save import save_csv
from modules.evaluation import evaluate_gp_models_training


def gp_normalize():
    """
    Load 1000 selected H3O+ data points and then standardize by rescaling their mean to 0 and standard deviation to 1.
    - Ensure all feature contribute equally to kernel optimization.
    :return: X_scaled: ndarray of shape (1000, 6), standardized coordinates (R1–R6).
             y_scaled: ndarray of shape (1000,), standardized potential energy values.
    """
    X, y = load_h3o_data()
    train_i, val_i, test_i = split_h3o_dataset()

    X_scaler = StandardScaler().fit(X[train_i])
    y_scaler = StandardScaler().fit(y[train_i].reshape(-1, 1))

    X_scaled = X_scaler.transform(X[train_i])
    y_scaled = y_scaler.transform(y[train_i].reshape(-1, 1)).ravel()

    return X_scaled, y_scaled, X_scaler, y_scaler, train_i, val_i, test_i


def build_ard_kernel(y_variance, X_dims=6):
    """
    Build C * RBF(ARD) + WhiteKernel as baseline for identifying key dimensions
    - Constant amplitude C(y_var, (1e-8, 1e4)) → expected range around y variance, optimized based on ARD GP model
    - RBF(ARD) length_scale_bounds=(1e-2, 1e2) → expected range for ℓ across 6 dimensions
    - White noise noise_level_bounds=(1e-10, 1e-2) → expected to remain small for PES data
    :param y_variance: Initial variance estimate of the scaled PES
    :param X_dims: Dimensionality, 6 for H3O+ dataset
    :return: sklearn.gaussian_process.kernels.Kernel, kernel object to be optimized by GaussianProcessRegressor
    """
    return C(y_variance, (1e-8, 1e4)) * \
           RBF(length_scale=np.ones(X_dims), length_scale_bounds=(1e-2, 1e2)) + \
           WhiteKernel(noise_level=0.01*y_variance, noise_level_bounds=(1e-10, 1e-2))


def build_kernels(y_variance):
    """
    Build the 4 simple kernels from sklearn for comparison in the form of C * BasicKernel + WhiteKernel
    - Constant amplitude C(y_var, (1e-8, 1e4)) → expected range around y variance, optimized based on ARD GP model
    - RBF / Matern / RationalQuadratic / DotProduct: length_scale_bounds=(1e-3, 1e3) → expected range for ℓ across 6 dimensions
    - White noise noise_level_bounds=(1e-10, 1e-2) → expected to remain small for PES data
    :param y_variance: Initial variance estimate of the scaled PES
    :return: kernels: dict[str, sklearn.gaussian_process.kernels.Kernel], kernel object to be optimized by GaussianProcessRegressor
    """
    amp_bounds = (1e-8, 1e4)
    len_scal_bounds = (1e-3, 1e3)
    noise_bounds = (1e-10, 1e-2)

    kernels = {
        "RBF": C(y_variance, amp_bounds) * RBF(1.0, len_scal_bounds) +
               WhiteKernel(0.01 * y_variance, noise_bounds),
        "Matern": C(y_variance, amp_bounds) * Matern(1.0, nu=1.5, length_scale_bounds=len_scal_bounds) +
               WhiteKernel(0.01 * y_variance, noise_bounds),
        "RationalQuadratic": C(y_variance, amp_bounds) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=len_scal_bounds) +
               WhiteKernel(0.01 * y_variance, noise_bounds),
        "DotProduct": C(y_variance, amp_bounds) * DotProduct(sigma_0=1.0) +
               WhiteKernel(0.01 * y_variance, noise_bounds),
    }
    return kernels


def train_gp_ard(n_restarts=9, seed=507):
    """
    Train a reference ARD GP model for assessing ℓ (length-scale parameter) across all X dimensions
    - Small ℓ → high sensitivity / strong relevance
    - Large ℓ → weak sensitivity / low relevance
    :param n_restarts: Number of repeats for avoiding LML local minimal
    :param seed: Fix the initial parameter guess for reproducibility
    :return: gp_ard : sklearn.gaussian_process.GaussianProcessRegressor
    """
    gp_ard = None
    max_lml = -np.inf

    X_sub, y_sub, X_scaler, y_scaler, train_i, val_i, test_i = gp_normalize()
    y_variance = np.var(y_sub)

    kernel = build_ard_kernel(y_variance)

    for i in range(n_restarts):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=0,
            normalize_y=False,
            random_state=seed + i
        )

        gp.fit(X_sub, y_sub)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)
        print(f"Restart {i + 1}/{n_restarts} → LML = {lml:.3f}")
        print(f"Kernel {i + 1}: {gp.kernel_}")

        if lml > max_lml:
            max_lml, gp_ard = lml, gp

    y_hat, _ = gp_ard.predict(X_sub, return_std=True)
    print("===== ARD kernel =====")
    print(gp_ard.kernel_)
    print("MSE:", f"{mean_squared_error(y_sub, y_hat):.3e}",
          "R²:", f"{r2_score(y_sub, y_hat):.4f}")

    return gp_ard


def train_gp_basic(n_restarts=9, seed=507):
    """
    Train GP models of the 4 basic kernels provided by sklearn
    :param n_restarts: Number of repeats for avoiding LML local minimal
    :param seed: Fix the initial parameter guess for reproducibility
    :return: gp_models : dict[str, sklearn.gaussian_process.GaussianProcessRegressor]
    """
    gp_models = {}
    X_sub, y_sub, X_scaler, y_scaler, train_i, val_i, test_i = gp_normalize()
    y_variance = np.var(y_sub)
    kernels = build_kernels(y_variance)

    for name, kernel in kernels.items():
        print(f"Training {name} kernel...")
        max_lml = -np.inf
        gp_best = None

        for i in range(n_restarts):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=0,
                normalize_y=False,
                random_state=seed + i
            )

            gp.fit(X_sub, y_sub)
            lml = gp.log_marginal_likelihood(gp.kernel_.theta)
            print(f"Restart {i + 1}/{n_restarts} → LML = {lml:.3f}")
            print(f"Kernel {name} {i + 1}: {gp.kernel_}")
            if lml > max_lml:
                max_lml, gp_best = lml, gp

        print(f"→ Best LML for {name}: {max_lml:.3f}")
        print(f"  Optimized kernel: {gp_best.kernel_}")
        gp_models[name] = gp_best

    return gp_models


def new_constant_kernel(y_variance: float, bounds=(1e-8, 1e4)):
    return C(y_variance, bounds)


def new_white_kernel(y_variance: float, bounds=(1e-10, 1e-2)):
    return WhiteKernel(noise_level=0.01 * y_variance, noise_level_bounds=bounds)


def make_kernel(name: str, y_variance: float, X_dims: int = 6):
    name = name.upper()
    if name == "RQ":
        return RationalQuadratic(length_scale=1.0, alpha=1.0)
    if name == "RBF":
        return RBF(length_scale=1.0)
    if name == "MAT":
        return Matern(length_scale=1.0, nu=1.5)
    if name == "LIN":
        return DotProduct(sigma_0=1.0)
    raise ValueError(f"Unknown basic kernel '{name}'")


def wrap_kernel(core, y_variance: float):
    return new_constant_kernel(y_variance) * core + new_white_kernel(y_variance)


def kernel_params_count(kernel) -> int:
    return int(kernel.theta.size)


def expand_kernel(cores: dict[str, object], y_variance: float) -> dict[str, object]:
    """
    Expand each core via additive: core + {RBF, RQ, MAT} and multiplicative: core * {LIN, RBF}
    Returns expanded kernels
    """
    additive_kernels = ["RBF", "RQ", "MAT"]
    multiplicative_kernels = ["LIN", "RBF"]
    expanded_kernels: dict[str, object] = {}

    for name, core in cores.items():
        for k in additive_kernels:
            expanded_kernels[f"({name}) + {k}"] = core + make_kernel(k, y_variance)
        for k in multiplicative_kernels:
            expanded_kernels[f"({name}) * {k}"] = core * make_kernel(k, y_variance)
    return expanded_kernels


def fit_gp(core_kernel, X_scaled, y_scaled, y_variance: float, n_restarts: int, seed: int):
    base = wrap_kernel(core_kernel, y_variance)
    gp = GaussianProcessRegressor(
        kernel=base,
        n_restarts_optimizer=n_restarts,
        normalize_y=False,
        random_state=seed,
    )
    gp.fit(X_scaled, y_scaled)
    return gp


def train_base_level(base_kernel: str, y_variance: float, X_training_scaled: np.ndarray, y_training_scaled: np.ndarray,n_restarts: int, seed: int):

    print(f"\n=== Level 0: base core = {base_kernel} ===")
    base= {base_kernel: make_kernel(base_kernel, y_variance)}
    models = {}

    for name, kernel in base.items():
        print(f"→ Fitting base kernel: {name}")
        gp = fit_gp(kernel, X_training_scaled, y_training_scaled, y_variance, n_restarts, seed)
        lml = gp.log_marginal_likelihood(gp.kernel_.theta)

        print(f"  LML: {lml:.3f} | Optimized: {gp.kernel_}\n")
        models[name] = gp

    return base, models


def prune_cores(models: dict[str, sklearn.gaussian_process.GaussianProcessRegressor],
                eval_df: pd.DataFrame, top_k: int, prune_metric: str) -> dict[str, object]:
    """
    Choose top-k cores by prune_metric ('rmse' asc or 'bic' asc) for the next level.
    Strips the 'Lx: ' prefix to map back to raw core names.
    """
    metric = prune_metric.lower()
    if metric == "bic":
        ordered = eval_df.sort_values("BIC", ascending=True)
    else:
        ordered = eval_df.sort_values("RMSE", ascending=True)

    raw_names = [name.split(": ", 1)[1] if ": " in name else name
                 for name in ordered["Kernel"].head(top_k)]
    return {name: models[name] for name in raw_names if name in models}


def train_gp_complex(X: np.ndarray, y: np.ndarray, train_i: np.ndarray, val_idx: np.ndarray, X_scaler, y_scaler,
                     seed: int = 507, n_restarts: int = 6, levels: int = 2, base_kernel: str = "RQ", inverse_transform: bool = True,
                     top_k: int = 5, prune_metric: str = "BIC", checkpoint_dir: str | Path = "models/gp_complex_checkpoints") \
        -> tuple[dict[str, object], pd.DataFrame]:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    X_train_scaled = X_scaler.transform(X[train_i])
    y_train_scaled = y_scaler.transform(y[train_i].reshape(-1, 1)).ravel()
    X_val, y_val = X[val_idx], y[val_idx]
    size = len(train_i)
    y_variance = float(np.var(y_train_scaled))

    # Level 0
    core_kernels, models = train_base_level(base_kernel, y_variance, X_train_scaled, y_train_scaled, n_restarts, seed)
    summary_L0 = evaluate_gp_models_training(models, X_val, y_val, X_scaler, y_scaler, size, inverse_transform, 0)
    save_csv(summary_L0, checkpoint_dir / f"level{0}_summary.csv")
    print(f"Saved level-0 summary → {checkpoint_dir}/level0_summary.csv")

    # Filter top-k and higher levels
    next_cores = prune_cores(core_kernels, summary_L0, top_k=top_k, prune_metric=prune_metric)
    summary_frame = [summary_L0]
    current_cores = next_cores
    for L in range(1, levels + 1):
        print(f"\n=== Level {L}: expanding {len(current_cores)} cores ===")
        t0 = time.time()
        expanded_kernels = expand_kernel(current_cores, y_variance)
        print(f"  Generated {len(expanded_kernels)} candidate cores at Level {L}")

        models_L= {}
        for idx, (name, core) in enumerate(expanded_kernels.items(), 1):
            print(f"  [L{L} {idx}/{len(expanded_kernels)}] Training: {name}")
            gp = fit_gp(core, X_train_scaled, y_train_scaled, y_variance, n_restarts, seed + 1000 * L)
            models_L[name] = gp

            models[f"L{L}: {name}"] = gp

        summary_L = evaluate_gp_models_training(models_L, X_val, y_val, X_scaler, y_scaler, size, inverse_transform, L)
        save_csv(summary_L, checkpoint_dir / f"level{L}_summary.csv")
        summary_frame.append(summary_L)

        current_cores = prune_cores(expanded_kernels, summary_L, top_k, prune_metric)
        dt_min = (time.time() - t0) / 60.0
        print(f"--- Level {L} complete ({dt_min:.2f} min). Kept top-{top_k} for next expansion. ---")

        if len(current_cores) == 0:
            print("No kernels retained for next level (early stop).")
            break

    evaluation = pd.concat(summary_frame, ignore_index=True)
    evaluation = evaluation.sort_values(["Level", "RMSE"], ascending=[True, True]).reset_index(drop=True)
    return models, evaluation
