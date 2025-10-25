import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from modules.evaluation import evaluate_pca_L
from modules.synthesize import synthesize_pdc_data


def train_pca(n: int, p: int, threshold: float = 0.95, seed: int = 507):
    X, Sigma = synthesize_pdc_data(n, p, seed)
    X_center = X - X.mean(axis=0)

    model = PCA(n_components=p, svd_solver="full", random_state=0)
    model.fit(X_center)
    explained_variance_ratio = model.explained_variance_ratio_
    cumulative_evr = np.cumsum(explained_variance_ratio)

    L = int(np.searchsorted(cumulative_evr, threshold) + 1)
    rmse = evaluate_pca_L(L, model, X_center)
    print(f"Chosen L (≥ {threshold * 100:.0f}% variance): {L} of {p}")
    print(f"Reconstruction RMSE with L={L}: {rmse:.4f}")

    return model, L, rmse, explained_variance_ratio, cumulative_evr

