"""벡터 검색과 선택적 reranking."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from rag_eval.corpus import Chunk


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


def _l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


def embed_cached(embedder: Embedder, model: str, texts: list[str], cache_dir: Path) -> np.ndarray:
    """모델명+텍스트 해시로 임베딩을 캐시한다. 설정을 바꿔 재실행할 때 시간을 크게 줄인다."""
    h = hashlib.sha256(model.encode())
    for t in texts:
        h.update(b"\x00" + t.encode())
    path = cache_dir / f"{h.hexdigest()[:24]}.npy"
    if path.is_file():
        cached = np.load(path)
        if cached.shape[0] == len(texts):
            return cached
    vectors = embedder.embed(texts)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError(f"임베딩 결과 shape {vectors.shape}가 입력 수 {len(texts)}와 맞지 않습니다")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError("청크 수와 벡터 수가 다릅니다")
        self.chunks = chunks
        self.matrix = _l2_normalize(vectors.astype(np.float32))

    def search(self, query_vec: np.ndarray, k: int) -> list[Chunk]:
        q = _l2_normalize(query_vec.reshape(1, -1).astype(np.float32))[0]
        scores = self.matrix @ q
        k = min(k, len(self.chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [self.chunks[i] for i in top]


def rerank(reranker: Reranker, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
    scores = reranker.score(query, [c.text for c in chunks])
    order = sorted(range(len(chunks)), key=lambda i: -scores[i])
    return [chunks[i] for i in order[:k]]


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder 래퍼. `pip install .[rerank]` 필요."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "reranker 사용에는 sentence-transformers가 필요합니다: pip install -e '.[rerank]'"
            ) from e
        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]
