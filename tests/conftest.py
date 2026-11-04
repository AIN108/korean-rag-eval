from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


class FakeEmbedder:
    """문자 bigram 해싱 벡터. Ollama 없이 결정적으로 동작하며, 겹치는 단어가 많을수록 가깝다."""

    dim = 256

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = "".join(t.split())
            for a, b in zip(s, s[1:], strict=False):
                h = int(hashlib.md5((a + b).encode()).hexdigest(), 16) % self.dim
                out[i, h] += 1.0
        return out


class FakeChat:
    def __init__(self, answer: str = "사용 3일 전까지 예약해야 합니다.", judge: str = '{"correct": true}'):
        self.answer, self.judge = answer, judge
        self.calls: list[bool] = []

    def generate(self, messages, json_mode: bool = False) -> str:
        self.calls.append(json_mode)
        return self.judge if json_mode else self.answer


@pytest.fixture
def sample_docs_dir() -> Path:
    return ROOT / "data" / "docs"


@pytest.fixture
def sample_qa_path() -> Path:
    return ROOT / "data" / "qa" / "sample.jsonl"
