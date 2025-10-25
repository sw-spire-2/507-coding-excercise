import numpy as np


def random_spd_cov(p: int, jitter: float = 1.0, seed: int = 507) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    C = A @ A.T
    C += jitter * np.eye(p)
    d = np.sqrt(np.diag(C))
    C = C / (d[:, None] * d[None, :])

    s = rng.uniform(0.5, 2.0, size=p)
    Sigma = (s[:, None] * C) * s[None, :]
    return Sigma


def synthesize_pdc_data(n: int, p: int, seed: int = 507):
    Sigma = random_spd_cov(p, jitter=1.0, seed=seed)
    mean = np.zeros(p)
    X = np.random.default_rng(seed+1).multivariate_normal(mean, Sigma, size=n)
    return X, Sigma
