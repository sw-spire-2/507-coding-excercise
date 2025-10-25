from matplotlib import pyplot as plt
from tabulate import tabulate
from modules.load import *
from modules.save import *
from modules.gp_train import gp_normalize, train_gp_basic, train_gp_complex
from modules.evaluation import evaluate_gp_models
from modules.krr_train import train_krr_fix_split
from modules.nn_train import train_nn_fix_split
from modules.svm_train import train_svm
from modules.plot import plot_gp_complexity, plot_krr_scaling, plot_compare_krr_nn, plot_svm_summary
from modules.synthesize import synthesize_pdc_data
from modules.pca_train import train_pca


# =============================================================================
# Exercise 1: Generate a 50-dimensional data set and perform PCA
# =============================================================================
def exec1(n: int = 4000, p: int = 50, threshold: float = 0.95, seed: int = 507):
    save_dir = "models/pca_model"
    print(f"\n- Generating synthetic data (n={n}, p={p})...")
    X, Sigma = synthesize_pdc_data(n, p, seed=seed)

    print("- Training PCA model...")
    model, L, rmse, explained_variance_ratio, cumulative_evr = train_pca(n=n, p=p, threshold=threshold, seed=seed)

    print("- Saving PCA model and statistics...")
    save_pca_model(model, L, rmse, explained_variance_ratio, cumulative_evr, Sigma, path=save_dir)

    print("- Summary:\n")
    table = [
        ["n (samples)", n],
        ["p (features)", p],
        [f"L (≥{threshold*100:.0f}% variance)", L],
        ["RMSE", f"{rmse:.5f}"],
        ["Variance explained", f"{100 * cumulative_evr[L-1]:.2f}%"],
    ]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
    print(f"\nModel saved in: {save_dir}\n")


def exec1_plot(threshold: float = 0.95, p: int = 50, save_dir: str | Path = "models/pca_model"):
    model, _, explained_variance_ratio, cumulative_evr, _ = load_pca_model(save_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scree Plot
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, p + 1), model.explained_variance_, marker='o', color='steelblue')
    plt.xlabel("Principal Component", fontsize=12)
    plt.ylabel("Eigenvalue (variance)", fontsize=12)
    plt.title("Scree Plot of PCA Components", fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_dir / "pca_scree_plot.png", dpi=300)
    plt.show()

    # 2. Cumulative Explained Variance
    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(1, p + 1), cumulative_evr, marker='o', color='darkorange')
    plt.axhline(threshold, linestyle='--', color='gray', label=f"{int(threshold * 100)}% threshold")
    plt.xlabel("Principal Component", fontsize=12)
    plt.ylabel("Cumulative Explained Variance", fontsize=12)
    plt.title("Cumulative Explained Variance Ratio", fontsize=13)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_dir / "pca_cumulative_evr.png", dpi=300)
    plt.show()

    print(f"- Figures saved in {save_dir}:")


# =============================================================================
# Exercise 2, Task 1: Train & Examine KRR Model Accuracy As Function of n
# =============================================================================
def exec2_task1():
    save_dir = "models/krr_h3o_models"
    print("- Loading H3O+ dataset...")
    X, y = load_h3o_data()
    print(f"   → Loaded {len(X)} samples with {X.shape[1]} features")

    print("\n- Creating fixed data splits...")
    dict_train_i, val_i, test_i, _ = fix_splits_h3o(size_total=len(X))
    print(f"   → Training sizes: {list(dict_train_i.keys())}")
    print(f"   → Validation size: {len(val_i)}, Test size: {len(test_i)}")

    print("\n- Training KRR models with cross-validation...")
    summary = train_krr_fix_split(X, y, dict_train_i, val_i, test_i)

    print("\n- Training complet. Summary of test results:")
    print("-" * 70)
    print(f"{'n_train':>10} | {'RMSE':>10} | {'R²':>10} | {'CV MSE':>12} | {'Fit Time (s)':>12}")
    print("-" * 70)
    for n, rec in summary.items():
        print(f"{n:10d} | {rec['test_RMSE']:10.4e} | {rec['test_r2']:10.4f} | "
              f"{rec['cv_best_mse']:12.4e} | {rec['total_fit_time_sec']:12.2f}")
    print("-" * 70)

    print(f"\n- Saving models and results to: {save_dir}")
    save_krr_models(summary, save_dir)


# =============================================================================
# Exercise 2, Task 2: Plot time requirement and accuracy with respect to n
# =============================================================================
def exec2_plot():
    plot_krr_scaling("models/krr_h3o_models/summary.json", save_dir="models/krr_h3o_models/plots")


# =============================================================================
# Exercise 3 SVM Classification of 5D data
# =============================================================================
def exec3():
    save_dir = "models/svm_model"

    print("\n- Loading 5D classification dataset...")
    X, y = load_5d_classification_data()

    print("- Training SVM model with GridSearchCV (linear + RBF kernels)...")
    grid, best_model, acc, prec, rec, f1, auc, counts_total, counts_train, counts_test = train_svm(X=X, y=y)
    class_counts = {
        "dataset": "5D-data-for-linear-classification.csv",
        "class_counts_total": counts_total,
        "class_counts_train": counts_train,
        "class_counts_test": counts_test,
    }

    print("- Saving trained SVM model, CV leaderboard, and summary...")
    save_svm_model(grid, best_model, acc, prec, rec, f1, auc, class_counts=class_counts, path=save_dir)

    print("\n- Summary of classification performance:\n")
    table = [
        ["# Samples", len(X)],
        ["# Features", X.shape[1]],
        ["Class distribution (total)", counts_total],
        ["Class distribution (train)", counts_train],
        ["Class distribution (test)", counts_test],
        ["Accuracy", f"{acc:.4f}"],
        ["Precision", f"{prec:.4f}"],
        ["Recall", f"{rec:.4f}"],
        ["F1-score", f"{f1:.4f}"],
        ["ROC-AUC", f"{auc:.4f}"],
        ["Best kernel", grid.best_params_.get("svc__kernel", "N/A")],
        ["Best C", grid.best_params_.get("svc__C", "N/A")],
        ["Best gamma", grid.best_params_.get("svc__gamma", "N/A")],
    ]

    print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))
    print(f"\nModel and results saved in: {save_dir}\n")


def exec3_plot():
    plot_svm_summary("models/svm_model/summary.json", save_dir="models/svm_model")

# =============================================================================
# Exercise 4: Train & Examine NN Model Accuracy As Function of n
# =============================================================================
def exec4_task1():
    save_dir = "models/nn_h3o_models"

    hidden_layers = (128, 64)
    activation = "relu"
    l2_weight = 0.0
    dropout = 0.0
    lr= 1e-3
    epochs = 2000
    batch_size = 128
    patience = 100
    seed = 507
    inverse_transform= True

    print("- Loading H3O+ dataset...")
    X, y = load_h3o_data()
    print(f"   → Loaded {len(X)} samples with {X.shape[1]} features")

    print("\n- Creating fixed data splits...")
    dict_train_i, val_i, test_i, _ = fix_splits_h3o(size_total=len(X))
    print(f"   → Training sizes: {list(dict_train_i.keys())}")
    print(f"   → Validation size: {len(val_i)}, Test size: {len(test_i)}")

    print("\n- Training Neural Network models (fixed split + early stopping)...")
    nn_summary = train_nn_fix_split(
        X, y, dict_train_i, val_i, test_i,
        hidden_layers=hidden_layers,
        activation=activation,
        l2_weight=l2_weight,
        dropout=dropout,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        inverse_transform=inverse_transform,
        seed=seed,
    )

    print("\n- Training complete. Summary of results (validation & test):")
    print("-" * 100)
    print(f"{'n_train':>8} | {'Val RMSE':>10} | {'Val R²':>8} | {'Test RMSE':>10} | {'Test R²':>8} | {'Fit Time (s)':>12} | {'Epochs':>6}")
    print("-" * 100)
    for n, rec in nn_summary.items():
        epochs_trained = rec.get("nn_training", {}).get("epochs_trained", None)
        print(f"{n:8d} | {rec['val_RMSE']:10.4e} | {rec['val_r2']:8.4f} | "
              f"{rec['test_RMSE']:10.4e} | {rec['test_r2']:8.4f} | "
              f"{rec['total_fit_time_sec']:12.2f} | {epochs_trained:6d}")
    print("-" * 100)

    print(f"\n- Saving models and results to: {save_dir}")
    save_nn_models(nn_summary, save_dir, meta={
        "hidden_layers": list(hidden_layers),
        "activation": activation,
        "l2_weight": l2_weight,
        "dropout": dropout,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "seed": seed,
        "inverse_transform": inverse_transform,
        "description": "NN regression on H3O+ with fixed splits",
    })
    print("- All NN models and metadata saved.")


def exec4_plot():
    plot_compare_krr_nn("models/krr_h3o_models/summary.json", "models/nn_h3o_models/summary.json", save_dir="models/compare_krr_nn")


# =============================================================================
# Exercise 5, Task 1: Train 4 GP models with sklearn's basic kernels
# =============================================================================
def exec5_task1():
    save_dir = "models/basic_gp_models"
    X_sub, y_sub, X_scaler, y_scaler,  train_i, val_i, test_i = gp_normalize()
    basic_gp_models = train_gp_basic()

    summary = summarize_gp_models(basic_gp_models, len(train_i))
    display_summary(summary)

    basic_meta = {"seed": 507, "size": 1000, "description": "train 4 GP models with sklearn's basic kernels"}
    save_csv(summary, Path(f"{save_dir}/training_summary.csv"))
    save_gp_models(basic_gp_models, X_scaler, y_scaler, train_i, val_i, test_i, basic_meta, Path(save_dir))
    print(f"\nAll models saved to: {save_dir}\n")


def exec5_task2():
    save_dir = "models/basic_gp_models"

    basic_gp_models, X_scaler, y_scaler,  train_i, val_i, test_i, meta = load_gp_models(Path("models/basic_gp_models"))
    X, y = load_h3o_data()
    X_val = X[val_i]
    y_val = y[val_i]
    size = len(train_i)

    basic_evaluation = evaluate_gp_models(basic_gp_models, size, X_val, y_val, X_scaler, y_scaler, True)
    save_csv(basic_evaluation, f"{save_dir}/evaluation_summary.csv")
    print(basic_evaluation.to_string(index=False))


# =============================================================================
# Exercise 5, Task 3: Improve the generalization by building more complex kernels
# =============================================================================
def exec5_task3():
    save_dir = Path("models/level2_gp_models")
    save_dir.mkdir(parents=True, exist_ok=True)

    basic_gp_models, X_scaler, y_scaler,  train_i, val_i, test_i, meta_basic = load_gp_models(Path("models/basic_gp_models"))
    X, y = load_h3o_data()
    seed = meta_basic.get("seed") if isinstance(meta_basic, dict) else 507
    models_complex, evaluation = train_gp_complex(X, y, train_i, val_i, X_scaler, y_scaler, seed, 6, 2, "RQ", True)

    meta_complex = {
        "description": "Level-2 complexity expansion for C*(RQ)+White kernel",
        "seed": seed,
        "n_restarts": 6,
        "levels": 2,
        "base_kernel": "RQ",
        "size_train": int(len(train_i)),
        "source": "models/basic_gp_models",
        "time": datetime.now().isoformat(),
    }
    save_gp_models(models_complex, X_scaler, y_scaler, train_i, val_i, test_i, meta_complex, save_dir)

    print("\n=== Validation summary for complex kernels (sorted by RMSE) ===")
    display_summary(evaluation)
    evaluation = evaluation.sort_values(["Level", "RMSE"], ascending=[True, True]).reset_index(drop=True)
    save_csv(evaluation, f"{save_dir}/summary_gp_complex_level2.csv")
    print(f"\nSaved models and summary to: {save_dir}")


def exec5_test():
    save_dir = Path("models/level2_gp_models")
    complex_gp_models, X_scaler, y_scaler, train_i, val_i, test_i, meta = load_gp_models(Path("models/level2_gp_models"))
    X, y = load_h3o_data()
    X_test = X[test_i]
    y_test = y[test_i]
    size = len(train_i)

    test_evaluation = evaluate_gp_models(complex_gp_models, size, X_test, y_test, X_scaler, y_scaler, True)
    save_csv(test_evaluation, f"{save_dir}/test_evaluation_summary.csv")
    print(test_evaluation.to_string(index=False))


def exec5_plot():
    save_dir = Path("models/level2_gp_models")
    selected_models = load_csv(f"{save_dir}/selected_models_plotting.csv")
    plot_gp_complexity(selected_models, "Params", save_dir)


#if __name__ == "__main__":
