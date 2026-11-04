"""평가용 질의응답셋(JSONL) 로드와 근거 위치 검증."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag_eval.corpus import Document, normalize_text

REQUIRED_FIELDS = ("id", "question", "answer", "doc_id", "evidence")


@dataclass(frozen=True)
class QAItem:
    qid: str
    question: str
    answer: str
    doc_id: str
    evidence: str
    start: int
    end: int


def locate_evidence(doc_text: str, evidence: str) -> tuple[int, int] | None:
    """공백·줄바꿈 차이를 무시하고 근거 문장의 원문 위치를 찾는다."""
    tokens = normalize_text(evidence).split()
    if not tokens:
        return None
    match = re.search(r"\s+".join(map(re.escape, tokens)), doc_text)
    return (match.start(), match.end()) if match else None


def load_qa(path: Path, documents: list[Document]) -> list[QAItem]:
    """QA셋을 읽고 모든 항목의 근거가 문서에 실제로 존재하는지 검증한다.

    오류는 한 번에 모아서 보고한다 (질의셋 작성 중 반복 수정을 줄이기 위해).
    """
    if not path.is_file():
        raise FileNotFoundError(f"QA 파일이 없습니다: {path}")
    docs = {d.doc_id: d for d in documents}
    items: list[QAItem] = []
    errors: list[str] = []
    seen: set[str] = set()

    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{lineno}행: JSON 파싱 실패 ({e.msg})")
                continue
            missing = [k for k in REQUIRED_FIELDS if not str(row.get(k, "")).strip()]
            if missing:
                errors.append(f"{lineno}행: 필수 필드 누락 {missing}")
                continue
            qid = str(row["id"])
            if qid in seen:
                errors.append(f"{lineno}행: 중복 id '{qid}'")
                continue
            seen.add(qid)
            doc = docs.get(row["doc_id"])
            if doc is None:
                errors.append(f"{lineno}행: 존재하지 않는 doc_id '{row['doc_id']}'")
                continue
            span = locate_evidence(doc.text, row["evidence"])
            if span is None:
                errors.append(f"{lineno}행: evidence가 '{doc.doc_id}' 본문에서 발견되지 않음")
                continue
            items.append(
                QAItem(
                    qid=qid,
                    question=normalize_text(row["question"]).strip(),
                    answer=normalize_text(row["answer"]).strip(),
                    doc_id=doc.doc_id,
                    evidence=row["evidence"],
                    start=span[0],
                    end=span[1],
                )
            )

    if errors:
        raise ValueError(f"QA셋 검증 실패 ({path}):\n  " + "\n  ".join(errors))
    if not items:
        raise ValueError(f"QA셋이 비어 있습니다: {path}")
    return items
