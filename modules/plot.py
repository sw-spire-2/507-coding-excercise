import matplotlib.pyplot as plt
from pathlib import Path
import json
import numpy as np


def _to_int_dict(d):
    return {int(k): int(v) for k, v in d.items()} if isinstance(d, dict) else {}

def plot_gp_complexity(selected_models: str | Path, x_axis: str = "Params", save_dir: str | Path | None = None):
    if x_axis in selected_models.columns:
        selected_models = selected_models.sort_values(x_axis, ascending=True)

    # RMSE vs complexity
    if x_axis in selected_models.columns and "RMSE" in selected_models.columns:
        plt.figure()
        plt.plot(selected_models[x_axis].values, selected_models["RMSE"].values, marker="o")
        for _, row in selected_models.iterrows():
            label = row["Kernel"] if "Kernel" in selected_models.columns else ""
            plt.annotate(label, (row[x_axis], row["RMSE"]),
                         xytext=(5, 5), textcoords="offset points", fontsize=8)
        plt.xlabel(x_axis)
        plt.ylabel("RMSE (inverse_scale = True)")
        plt.title("Generalization Error vs Model Complexity (RMSE)")
        plt.tight_layout()
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / "rmse_vs_complexity.png", dpi=200)
        plt.show()

    # BIC vs complexity
    if x_axis in selected_models.columns and "BIC" in selected_models.columns:
        plt.figure()
        plt.plot(selected_models[x_axis].values, selected_models["BIC"].values, marker="o")
        for _, row in selected_models.iterrows():
            label = row["Kernel"] if "Kernel" in selected_models.columns else ""
            plt.annotate(label, (row[x_axis], row["BIC"]),
                         xytext=(5, 5), textcoords="offset points", fontsize=8)
        plt.xlabel(x_axis)
        plt.ylabel("BIC")
        plt.title("BIC vs Model Complexity")
        plt.tight_layout()
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(save_dir) / "bic_vs_complexity.png", dpi=200)
        plt.show()


def plot_krr_scaling(summary_path: str | Path, save_dir: str | Path | None = None):
    summary_path = Path(summary_path)
    with open(summary_path) as f:
        summary = json.load(f)

    n_training = np.array([int(k) for k in summary.keys()])
    rmse = np.array([v["test_RMSE"] for v in summary.values()])
    r2 = np.array([v["test_r2"] for v in summary.values()])
    fit_time = np.array([v["total_fit_time_sec"] for v in summary.values()])

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    #1. RMSE vs Training Size (log–log)
    plt.figure()
    plt.loglog(n_training, rmse, "o-", color="tab:blue", base=10)
    for n, val in zip(n_training, rmse):
        plt.annotate(f"{n}", (n, val), xytext=(5, 5), textcoords="offset points", fontsize=8)
    plt.xlabel("Training size (n)")
    plt.ylabel("Test RMSE")
    plt.title("KRR Accuracy Convergence (log–log)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "krr_rmse_vs_n.png", dpi=200)
    plt.show()

    #2. R² vs Training Size
    plt.figure()
    plt.plot(n_training, r2, "s-", color="tab:green")
    plt.xlabel("Training size (n)")
    plt.ylabel("Test R²")
    plt.title("KRR Generalization (R² vs n)")
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "krr_r2_vs_n.png", dpi=200)
    plt.show()

    #3. Fit Time vs Training Size (log–log)
    plt.figure()
    plt.loglog(n_training, fit_time, "o--", color="tab:orange", base=10)
    for n, t in zip(n_training, fit_time):
        plt.annotate(f"{n}", (n, t), xytext=(5, 5), textcoords="offset points", fontsize=8)
    plt.xlabel("Training size (n)")
    plt.ylabel("Total Fit Time (s)")
    plt.title("KRR Computational Scaling (log–log)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "krr_fit_time_vs_n.png", dpi=200)
    plt.show()

    #4. Combined RMSE–Time Plot
    fig, ax1 = plt.subplots()
    color1 = "tab:blue"
    ax1.set_xlabel("Training size (n)")
    ax1.set_ylabel("Test RMSE", color=color1)
    ax1.loglog(n_training, rmse, "o-", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Total Fit Time (s)", color=color2)
    ax2.loglog(n_training, fit_time, "s--", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("KRR Scaling: Accuracy vs Training Cost")
    fig.tight_layout()
    plt.grid(True, which="both", ls="--", lw=0.5)
    if save_dir:
        plt.savefig(save_dir / "krr_scaling_combined.png", dpi=200)
    plt.show()

    print(f"Plots generated{' and saved to ' + str(save_dir) if save_dir else ''}.")


def _load_summary_json(summary_path):
    with open(summary_path) as file:
        data = json.load(file)
    items = sorted(((int(k), v) for k, v in data.items()), key=lambda kv: kv[0])
    n = np.array([k for k, _ in items], dtype=int)
    rmse = np.array([v["test_RMSE"] for _, v in items], dtype=float)
    fit_time = np.array([v.get("total_fit_time_sec", np.nan) for _, v in items], dtype=float)
    return n, rmse, fit_time

def plot_compare_krr_nn(krr_summary_path: str | Path, nn_summary_path: str | Path, save_dir: str | Path | None = None, krr_label: str = "KRR (RBF)", nn_label: str = "NN (MLP)",):
    krr_summary_path = Path(krr_summary_path)
    nn_summary_path = Path(nn_summary_path)

    if not krr_summary_path.exists():
        raise FileNotFoundError(f"KRR summary not found: {krr_summary_path}")
    if not nn_summary_path.exists():
        raise FileNotFoundError(f"NN summary not found: {nn_summary_path}")

    n_krr, rmse_krr, time_krr = _load_summary_json(krr_summary_path)
    n_nn, rmse_nn, time_nn = _load_summary_json(nn_summary_path)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    #1. RMSE vs n (overlay)
    plt.figure()
    plt.loglog(n_krr, rmse_krr, "o-", label=krr_label)
    plt.loglog(n_nn,  rmse_nn,  "s--", label=nn_label)
    for x, y in zip(n_krr, rmse_krr):
        plt.annotate(f"{x}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    for x, y in zip(n_nn, rmse_nn):
        plt.annotate(f"{x}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    plt.xlabel("Training size (n)")
    plt.ylabel("Test RMSE")
    plt.title("Accuracy vs Training Size (log–log)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "compare_rmse_vs_n.png", dpi=200)
    plt.show()

    #2. Fit time vs n (overlay)
    plt.figure()
    plt.loglog(n_krr, time_krr, "o-", label=krr_label)
    plt.loglog(n_nn,  time_nn,  "s--", label=nn_label)
    for x, y in zip(n_krr, time_krr):
        plt.annotate(f"{x}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    for x, y in zip(n_nn, time_nn):
        plt.annotate(f"{x}", (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    plt.xlabel("Training size (n)")
    plt.ylabel("Total Fit Time (s)")
    plt.title("Training Time vs Training Size (log–log)")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "compare_fit_time_vs_n.png", dpi=200)
    plt.show()

    print("- Plots generated" + (f" and saved to {save_dir}" if save_dir else ""))


def plot_svm_summary(summary_path: str | Path, save_dir: str | Path | None = None):
    summary_path = Path(summary_path)
    with open(summary_path) as f:
        s = json.load(f)

    tm = s.get("test_metrics", {})
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    vals = [float(tm.get(m, np.nan)) for m in metrics]

    cc = s.get("class_counts", {})

    total = _to_int_dict(cc.get("class_counts_total", {}))
    train = _to_int_dict(cc.get("class_counts_train", {}))
    test = _to_int_dict(cc.get("class_counts_test", {}))
    dataset_name = cc.get("dataset")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metrics bar
    plt.figure()
    plt.bar(metrics, vals)
    plt.ylim(0, 1.05)
    for i, v in enumerate(vals):
        plt.text(i, min(1.02, v + 0.02), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.title("SVM Test Metrics")
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "svm_test_metrics.png", dpi=200)
    plt.show()

    # 2. Class counts bar
    groups = ["total", "train", "test"]
    zeros = [total.get(0, 0), train.get(0, 0), test.get(0, 0)]
    ones = [total.get(1, 0), train.get(1, 0), test.get(1, 0)]

    x = np.arange(len(groups))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, zeros, width, label="Class 0")
    plt.bar(x + width/2, ones,  width, label="Class 1")
    plt.xticks(x, groups)
    plt.ylabel("Count")
    title = "Class Counts" + (f" — {dataset_name}" if dataset_name else "")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / "svm_class_counts.png", dpi=200)
    plt.show()

    if save_dir:
        print(f"\nFigures saved to: {save_dir}\n")
