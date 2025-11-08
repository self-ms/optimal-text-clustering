from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA

@dataclass
class ReductionConfig:
    var_ratio: float = 0.95
    max_components: int = 256
    plot_2d_method: str = "pca"  # "pca" only for now

@dataclass
class ReductionResult:
    for_clustering: np.ndarray
    for_plot_2d: np.ndarray

def adaptive_pca(X: np.ndarray, cfg: ReductionConfig) -> ReductionResult:
    if X.size == 0:
        return ReductionResult(X, X)
    n, d = X.shape
    k = min(cfg.max_components, d, n) if n > 2 else min(d, 2)
    pca_full = PCA(n_components=k, svd_solver="auto", random_state=42)
    Z = pca_full.fit_transform(X)
    cums = np.cumsum(pca_full.explained_variance_ratio_)
    r = int(np.searchsorted(cums, cfg.var_ratio) + 1)
    r = max(2, min(r, k))
    Zc = Z[:, :r]
    pca2 = PCA(n_components=2, svd_solver="auto", random_state=42)
    Z2 = pca2.fit_transform(X)
    return ReductionResult(for_clustering=Zc, for_plot_2d=Z2)
