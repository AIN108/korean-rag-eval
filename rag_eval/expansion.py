"""용어집 기반 질의 확장.

한국어 질문에 등장한 용어의 영어 대응어를 질문 뒤에 덧붙인다. 영어 원문 문서를 한국어로
질문할 때 검색 재현율을 높이려는 방법이며, 효과는 실험으로 확인한다.
"""

from __future__ import annotations

from pathlib import Path

import yaml

Glossary = dict[str, list[str]]


def load_glossary(path: Path) -> Glossary:
    """YAML 매핑 {용어: 대응어 문자열 또는 리스트}를 읽는다."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"용어집은 매핑이어야 합니다: {path}")
    glossary: Glossary = {}
    for key, value in data.items():
        terms = value if isinstance(value, list) else [value]
        cleaned = [str(t).strip() for t in terms if str(t).strip()]
        if not str(key).strip() or not cleaned:
            raise ValueError(f"용어집 항목이 비어 있습니다: {path} ({key!r})")
        glossary[str(key).strip()] = cleaned
    return glossary


def expand_query(question: str, glossary: Glossary | None) -> str:
    if not glossary:
        return question
    extra: list[str] = []
    for key, terms in glossary.items():
        if key in question:
            extra.extend(t for t in terms if t not in extra)
    return f"{question} {' '.join(extra)}" if extra else question


def expansion_label(path: Path | None) -> str:
    return path.stem if path else "none"
