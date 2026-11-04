"""실험 설정(YAML) 로드와 검증."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """설정 파일이 잘못되었을 때 발생한다."""


@dataclass(frozen=True)
class ChunkConfig:
    size: int
    overlap: int

    @property
    def label(self) -> str:
        return f"{self.size}/{self.overlap}"


EMBED_BACKENDS = ("ollama", "hf")


@dataclass(frozen=True)
class EmbedSpec:
    """임베딩 모델 한 개의 실험 설정.

    같은 모델도 접두어 유무를 다른 실험 조건으로 비교할 수 있도록 표시 이름(name)을 따로 둔다.
    (예: multilingual-e5 계열은 질문에 "query: ", 문서에 "passage: "를 붙이도록 권장된다)
    """

    name: str
    model: str
    backend: str = "ollama"
    query_prefix: str = ""
    passage_prefix: str = ""

    @property
    def cache_id(self) -> str:
        return f"{self.backend}:{self.model}"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    docs_dir: Path
    qa_path: Path
    output_dir: Path
    ollama_url: str
    chunking: list[ChunkConfig]
    embed_models: list[EmbedSpec]
    llm_models: list[str]
    rerankers: list[str | None]
    query_expansions: list[Path | None] = field(default_factory=lambda: [None])
    top_k: int = 4
    rerank_candidates: int = 20
    min_evidence_coverage: float = 0.5
    judge_model: str | None = None
    temperature: float = 0.0
    seed: int = 42
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def _require_list(data: dict[str, Any], key: str, *, allow_empty: bool = False) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise ConfigError(f"'{key}'는 리스트여야 합니다 (현재: {type(value).__name__})")
    if not value and not allow_empty:
        raise ConfigError(f"'{key}'에 최소 1개 항목이 필요합니다")
    return value


def _parse_chunking(items: list[Any]) -> list[ChunkConfig]:
    result = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "size" not in item:
            raise ConfigError(f"chunking[{i}]에는 'size'(와 선택적으로 'overlap')가 필요합니다")
        size, overlap = int(item["size"]), int(item.get("overlap", 0))
        if size <= 0:
            raise ConfigError(f"chunking[{i}].size는 양수여야 합니다 (현재: {size})")
        if not 0 <= overlap < size:
            raise ConfigError(f"chunking[{i}].overlap은 0 이상 size 미만이어야 합니다 (현재: {overlap})")
        result.append(ChunkConfig(size=size, overlap=overlap))
    return result


def _parse_embed_models(items: list[Any]) -> list[EmbedSpec]:
    """문자열(Ollama 모델명) 또는 {model, name, backend, query_prefix, passage_prefix} 매핑."""
    specs = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            specs.append(EmbedSpec(name=item, model=item))
            continue
        if not isinstance(item, dict) or not item.get("model"):
            raise ConfigError(f"embed_models[{i}]는 모델명 문자열이거나 'model' 키가 있는 매핑이어야 합니다")
        backend = item.get("backend", "ollama")
        if backend not in EMBED_BACKENDS:
            raise ConfigError(
                f"embed_models[{i}].backend는 {EMBED_BACKENDS} 중 하나여야 합니다 (현재: {backend})"
            )
        specs.append(
            EmbedSpec(
                name=str(item.get("name") or item["model"]),
                model=str(item["model"]),
                backend=backend,
                query_prefix=str(item.get("query_prefix", "")),
                passage_prefix=str(item.get("passage_prefix", "")),
            )
        )
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ConfigError(f"embed_models의 name이 중복됩니다: {names}")
    return specs


def _parse_query_expansions(items: list[Any]) -> list[Path | None]:
    result: list[Path | None] = []
    for i, item in enumerate(items):
        if item is None:
            result.append(None)
            continue
        path = Path(item)
        if not path.is_file():
            raise ConfigError(f"query_expansions[{i}] 용어집 파일이 없습니다: {path}")
        result.append(path)
    return result


def load_config(path: str | Path) -> ExperimentConfig:
    """YAML 설정을 읽어 검증한다. 상대 경로는 현재 작업 디렉터리 기준으로 해석한다."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError("설정 파일 최상위는 매핑이어야 합니다")

    for key in ("name", "docs_dir", "qa_path"):
        if not data.get(key):
            raise ConfigError(f"필수 항목 '{key}'가 없습니다")

    rerankers = [r or None for r in data.get("rerankers", [None])]
    top_k = int(data.get("top_k", 4))
    rerank_candidates = int(data.get("rerank_candidates", 20))
    coverage = float(data.get("min_evidence_coverage", 0.5))
    if top_k < 1:
        raise ConfigError("top_k는 1 이상이어야 합니다")
    if any(rerankers) and rerank_candidates < top_k:
        raise ConfigError("rerank_candidates는 top_k 이상이어야 합니다")
    if not 0.0 < coverage <= 1.0:
        raise ConfigError("min_evidence_coverage는 (0, 1] 범위여야 합니다")

    return ExperimentConfig(
        name=str(data["name"]),
        docs_dir=Path(data["docs_dir"]),
        qa_path=Path(data["qa_path"]),
        output_dir=Path(data.get("output_dir", "results")),
        # 환경변수가 있으면 설정 파일보다 우선한다 (Ollama CLI와 같은 변수명).
        ollama_url=os.environ.get("OLLAMA_HOST") or data.get("ollama_url", "http://localhost:11434"),
        chunking=_parse_chunking(_require_list(data, "chunking")),
        embed_models=_parse_embed_models(_require_list(data, "embed_models")),
        llm_models=[str(m) for m in _require_list(data, "llm_models", allow_empty=True)],
        rerankers=rerankers,
        query_expansions=_parse_query_expansions(data.get("query_expansions") or [None]),
        top_k=top_k,
        rerank_candidates=rerank_candidates,
        min_evidence_coverage=coverage,
        judge_model=data.get("judge_model") or None,
        temperature=float(data.get("temperature", 0.0)),
        seed=int(data.get("seed", 42)),
        raw=data,
    )
