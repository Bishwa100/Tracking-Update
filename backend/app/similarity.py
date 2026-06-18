"""
Embedding similarity helpers (in-memory).

Single source of truth for cosine similarity / normalization used outside the
database. DB gallery search still uses pgvector's `<=>` operator in
identity_resolver — these helpers are for NumPy-side comparisons (temporal gate,
gallery diversity, per-visitor threshold stats) so they all agree.
"""

from typing import List, Sequence

import numpy as np


def normalize_embedding(embedding) -> List[float]:
    """L2-normalize a vector and return it as a plain Python list."""
    arr = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def cosine_similarity(a, b, assume_normalized: bool = False) -> float:
    """
    Cosine similarity of two vectors. When both are already L2-normalized pass
    assume_normalized=True to skip the (redundant) norm division.
    """
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if assume_normalized:
        return float(np.dot(va, vb))
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / denom)


def pairwise_cosine(embeddings: Sequence) -> np.ndarray:
    """
    Upper-triangular pairwise cosine similarities of a set of (assumed
    L2-normalized) embeddings, returned as a flat array. Empty/singleton inputs
    return an empty array.
    """
    if embeddings is None or len(embeddings) < 2:
        return np.empty(0, dtype=np.float32)
    mat = np.asarray(embeddings, dtype=np.float32)
    sims = mat @ mat.T
    iu = np.triu_indices(mat.shape[0], k=1)
    return sims[iu].astype(np.float32)
