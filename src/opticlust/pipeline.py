from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd

from .embedding import EmbeddingConfig, embed_texts
from .reduction import ReductionConfig, adaptive_pca, ReductionResult
from .clustering import ClusterConfig, run_dbscan
from . import reporting

@dataclass
class ReportingConfig:
    top_representatives: int = 5

@dataclass
class FitResult:
    result_df: pd.DataFrame
    report: Dict[str, Any]
    embeddings: np.ndarray
    reduced: ReductionResult

class Text2Clusters:
    def __init__(
        self,
        embedding: EmbeddingConfig = EmbeddingConfig(),
        reduction: ReductionConfig = ReductionConfig(),
        clustering: ClusterConfig = ClusterConfig(),
        reporting_cfg: ReportingConfig = ReportingConfig()
    ) -> None:
        self.embedding = embedding
        self.reduction = reduction
        self.clustering = clustering
        self.reporting_cfg = reporting_cfg

    def _ensure_texts(self, data: Union[pd.DataFrame, List[str]], text_col: str) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            if text_col not in data.columns:
                raise ValueError(f"text_col '{text_col}' not in DataFrame columns: {list(data.columns)}")
            df = data.copy()
            df[text_col] = df[text_col].astype(str)
            return df
        # list[str]
        return pd.DataFrame({text_col: list(map(str, data))})

    def fit(self, data: Union[pd.DataFrame, List[str]], text_col: str = "text", id_col: Optional[str] = None) -> FitResult:
        df = self._ensure_texts(data, text_col=text_col)
        texts = df[text_col].tolist()

        X = embed_texts(texts, self.embedding)
        red = adaptive_pca(X, self.reduction)
        Z = red.for_clustering

        # Currently only DBSCAN
        clusters = run_dbscan(Z, self.clustering)

        # Build labels vector and assignment DataFrame
        n = len(df)
        labels = np.full(n, -1, dtype=int)
        for c in clusters:
            for idx in c["samples_idx"]:
                labels[int(idx)] = int(c["class"])

        out_df = df.copy()
        out_df["cluster"] = labels

        # Report
        rep = reporting.build_report(texts, labels, Z, method="dbscan", params=vars(self.clustering), top_n=self.reporting_cfg.top_representatives)

        return FitResult(result_df=out_df, report=rep, embeddings=X, reduced=red)
