from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import datetime

def _cluster_medoid_indices(X: np.ndarray, labels: np.ndarray, top_n: int = 3) -> dict[int, list[int]]:
    idxs: dict[int, list[int]] = {}
    for cl in sorted([c for c in set(labels) if c >= 0]):
        members = np.where(labels == cl)[0]
        if len(members) == 0:
            idxs[cl] = []
            continue
        Xc = X[members]
        centroid = normalize(Xc.mean(axis=0, keepdims=True), norm="l2")
        sims = Xc @ centroid.T
        order = np.argsort(-sims.ravel())[: min(top_n, len(members))]
        idxs[cl] = members[order].tolist()
    return idxs

def _tfidf_keywords(texts: List[str], labels: np.ndarray, top_k: int = 10) -> dict[int, list[str]]:
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2)
    Xtf = vec.fit_transform(texts)
    vocab = np.array(vec.get_feature_names_out())
    out: dict[int, list[str]] = {}
    for cl in sorted([c for c in set(labels) if c >= 0]):
        mask = labels == cl
        if mask.sum() == 0:
            out[cl] = []
            continue
        scores = Xtf[mask].mean(axis=0).A1
        top = np.argsort(-scores)[:top_k]
        out[cl] = vocab[top].tolist()
    return out

def build_report(texts: List[str], labels: np.ndarray, emb_space: np.ndarray, method: str, params: dict[str, Any], top_n: int = 5) -> dict[str, Any]:
    n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
    noise_ratio = float((labels == -1).mean())
    Xn = normalize(emb_space, norm="l2", copy=True)
    reps = _cluster_medoid_indices(Xn, labels, top_n=min(top_n, 10))
    kw = _tfidf_keywords(texts, labels, top_k=12)
    clusters = []
    for cl in sorted([c for c in set(labels) if c >= 0]):
        members = np.where(labels == cl)[0]
        clusters.append({
            "cluster_id": int(cl),
            "size": int(len(members)),
            "representatives": [{"index": int(i), "text": texts[i]} for i in reps.get(cl, [])],
            "top_keywords": kw.get(cl, []),
        })
    return {
        "method": method,
        "params": params,
        "n_clusters": int(n_clusters),
        "noise_ratio": noise_ratio,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "clusters": clusters,
    }
