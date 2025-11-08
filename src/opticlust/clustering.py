from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from numpy import unique, where
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.manifold import TSNE

@dataclass
class ClusterConfig:
    method: str = "dbscan"
    # DBSCAN params + sweep
    min_samples: int = 5
    metric: str = "euclidean"
    eps_start: float = 0.1
    eps_end: float = 5.0
    eps_lr: float = 0.1
    # Optional t-SNE preprojection
    use_tsne: bool = False
    tsne_perplexity: Optional[float] = 30.0
    tsne_random_state: int = 42

def _valid_silhouette(X: np.ndarray, labels: np.ndarray, metric: str) -> Optional[float]:
    mask = labels != -1
    if mask.sum() < 2:
        return None
    uniq = np.unique(labels[mask])
    if len(uniq) < 2:
        return None
    try:
        return float(metrics.silhouette_score(X[mask], labels[mask], metric=metric))
    except Exception:
        return None

def _sweep_dbscan(X: np.ndarray, cfg: ClusterConfig) -> float:
    r = cfg.eps_start
    step = cfg.eps_lr if cfg.eps_lr > 0 else (cfg.eps_end - cfg.eps_start) / 20.0
    best_r = r
    best_s = -np.inf
    while r <= cfg.eps_end + 1e-12:
        model = DBSCAN(eps=r, min_samples=cfg.min_samples, metric=cfg.metric)
        Y = model.fit_predict(X)
        num_clusters = len([lab for lab in unique(Y) if lab != -1])
        score = -np.inf
        if num_clusters >= 2:
            sil = _valid_silhouette(X, Y, metric=cfg.metric)
            if sil is not None:
                penalty = 1.0 - min(num_clusters, 10) / 50.0
                score = sil * penalty
        if score > best_s:
            best_s = score
            best_r = r
        r += step
    return best_r

def run_dbscan(X: np.ndarray, cfg: ClusterConfig) -> List[Dict[str, Any]]:
    if cfg.use_tsne:
        n = len(X)
        max_perp = max(5, min(cfg.tsne_perplexity or 30.0, (n - 1) / 3))
        X_ = TSNE(n_components=2, learning_rate="auto", init="pca",
                  perplexity=max_perp, random_state=cfg.tsne_random_state).fit_transform(X)
        metric = "euclidean"
    else:
        X_ = X
        metric = cfg.metric

    best_eps = _sweep_dbscan(X_, cfg)
    model = DBSCAN(eps=best_eps, min_samples=cfg.min_samples, metric=metric)
    Y = model.fit_predict(X_)
    labels = unique(Y)

    core_idx = getattr(model, "core_sample_indices_", None)
    core_set = set(core_idx.tolist()) if core_idx is not None else set()

    clusters: List[Dict[str, Any]] = []
    for label in labels:
        if label == -1:
            continue
        row_ix = where(Y == label)[0].tolist()
        core_sample = next((i for i in row_ix if i in core_set), row_ix[0])
        clusters.append({"class": int(label), "samples_idx": row_ix, "core_sample": int(core_sample)})
    return clusters
