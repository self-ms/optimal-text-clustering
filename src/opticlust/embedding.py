from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal, List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

DeviceStr = Literal["auto","cpu","cuda","mps"]

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device: DeviceStr = "auto"
    batch_size: int = 16
    max_length: int = 256  # truncate long texts

def _pick_device(device: DeviceStr) -> str:
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available.")
        return "mps"
    if device == "cpu":
        return "cpu"
    raise ValueError("device must be one of: auto, cpu, cuda, mps")

def _mean_pool(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    denom = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return summed / denom

def embed_texts(texts: Iterable[str], cfg: EmbeddingConfig) -> np.ndarray:
    texts = list(map(str, texts))
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    dev = _pick_device(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModel.from_pretrained(cfg.model_name)
    model.to(dev).eval()

    vecs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), cfg.batch_size), desc="Embedding", unit="batch"):
        batch = texts[i:i+cfg.batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt")
        if dev in ("cuda","mps"):
            enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            pooled = _mean_pool(out, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vecs.append(pooled.detach().cpu().numpy())
    return np.concatenate(vecs, axis=0).astype("float32")
