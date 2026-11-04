"""Ollama HTTP API 클라이언트 (임베딩, 채팅)."""

from __future__ import annotations

from typing import Any

import numpy as np
import requests


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.startswith(("http://", "https://")):
            self.base_url = "http://" + self.base_url
        self.timeout = timeout
        self._session = requests.Session()

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise OllamaError(
                f"Ollama 서버에 연결할 수 없습니다 ({self.base_url}). `ollama serve` 실행 여부를 확인하세요."
            ) from e
        if resp.status_code == 404 and "model" in payload:
            raise OllamaError(
                f"모델 '{payload['model']}'을 찾을 수 없습니다. `ollama pull {payload['model']}`로 받으세요."
            )
        if not resp.ok:
            raise OllamaError(f"Ollama 요청 실패 {resp.status_code} {path}: {resp.text[:300]}")
        return resp.json()

    def close(self) -> None:
        self._session.close()


class OllamaEmbedder:
    def __init__(self, client: OllamaClient, model: str, batch_size: int = 32) -> None:
        self.client, self.model, self.batch_size = client, model, batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            data = self.client.post("/api/embed", {"model": self.model, "input": batch})
            got = data.get("embeddings")
            if not isinstance(got, list) or len(got) != len(batch):
                raise OllamaError(f"임베딩 응답 형식이 예상과 다릅니다 (model={self.model})")
            vectors.extend(got)
        return np.asarray(vectors, dtype=np.float32)


class OllamaChat:
    def __init__(
        self, client: OllamaClient, model: str, temperature: float = 0.0, seed: int = 42
    ) -> None:
        self.client, self.model = client, model
        self.options = {"temperature": temperature, "seed": seed}

    def generate(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self.options,
        }
        if json_mode:
            payload["format"] = "json"
        data = self.client.post("/api/chat", payload)
        content = (data.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise OllamaError(f"채팅 응답에 message.content가 없습니다 (model={self.model})")
        return content.strip()
