"""평가용 질의응답셋(JSONL) 로드와 근거 위치 검증."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag_eval.corpus import Document, normalize_text
from rag_eval.prompts import NOT_FOUND

SPLITS = ("unspecified", "dev", "test")


@dataclass(frozen=True)
class EvidenceSpan:
    doc_id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class QAItem:
    qid: str
    question: str
    answer: str
    # 기존 단일 근거 API와 위치 인자를 유지한다.
    doc_id: str = ""
    evidence: str = ""
    start: int = 0
    end: int = 0
    answerable: bool = True
    evidences: tuple[EvidenceSpan, ...] = ()
    answer_aliases: tuple[str, ...] = ()
    question_type: str = "fact"
    split: str = "unspecified"
    source_group: str = ""

    @property
    def supports(self) -> tuple[EvidenceSpan, ...]:
        if not self.answerable:
            return ()
        return self.evidences or (EvidenceSpan(self.doc_id, self.evidence, self.start, self.end),)


def locate_evidence(doc_text: str, evidence: str, occurrence: int = 1) -> tuple[int, int] | None:
    """공백·줄바꿈 차이를 무시하고 근거 문장의 원문 위치를 찾는다."""
    tokens = normalize_text(evidence).split()
    if not tokens or occurrence < 1:
        return None
    for number, match in enumerate(re.finditer(r"\s+".join(map(re.escape, tokens)), doc_text), start=1):
        if number == occurrence:
            return match.start(), match.end()
    return None


def _require_text(row: dict, *keys: str) -> None:
    missing = [key for key in keys if not isinstance(row.get(key), str) or not row[key].strip()]
    if missing:
        raise ValueError(f"필수 필드 누락 또는 문자열 아님 {missing}")


def _parse_item(row: dict, qid: str, docs: dict[str, Document]) -> QAItem:
    _require_text(row, "question")
    answerable = row.get("answerable", True)
    if not isinstance(answerable, bool):
        raise ValueError("answerable은 true/false여야 합니다")
    split = row.get("split", "unspecified")
    if split not in SPLITS:
        raise ValueError(f"split은 {SPLITS} 중 하나여야 합니다")
    question_type = row.get("question_type", "fact" if answerable else "unanswerable")
    source_group = row.get("source_group", "")
    if not isinstance(question_type, str) or not question_type.strip():
        raise ValueError("question_type은 비어 있지 않은 문자열이어야 합니다")
    if not isinstance(source_group, str):
        raise ValueError("source_group은 문자열이어야 합니다")
    aliases = row.get("answer_aliases", [])
    if not isinstance(aliases, list) or any(not isinstance(a, str) or not a.strip() for a in aliases):
        raise ValueError("answer_aliases는 비어 있지 않은 문자열의 리스트여야 합니다")

    spans: list[EvidenceSpan] = []
    if answerable:
        _require_text(row, "answer")
        answer = normalize_text(row["answer"]).strip()
        if "evidences" in row:
            if "doc_id" in row or "evidence" in row:
                raise ValueError("evidences와 기존 doc_id/evidence 형식을 함께 사용할 수 없습니다")
            evidence_rows = row["evidences"]
        else:
            _require_text(row, "doc_id", "evidence")
            evidence_rows = [{"doc_id": row["doc_id"], "text": row["evidence"]}]
        if not isinstance(evidence_rows, list) or not evidence_rows:
            raise ValueError("답변 가능한 질문에는 최소 1개의 evidences가 필요합니다")
        for ev in evidence_rows:
            if not isinstance(ev, dict):
                raise ValueError("evidences의 각 항목은 객체여야 합니다")
            _require_text(ev, "doc_id", "text")
            doc = docs.get(ev["doc_id"])
            if doc is None:
                raise ValueError(f"존재하지 않는 doc_id '{ev['doc_id']}'")
            occurrence = ev.get("occurrence", 1)
            if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 1:
                raise ValueError("근거 occurrence는 1 이상의 정수여야 합니다")
            span = locate_evidence(doc.text, ev["text"], occurrence)
            if span is None:
                raise ValueError(
                    f"evidence가 '{doc.doc_id}' 본문에서 발견되지 않음 (occurrence={occurrence})"
                )
            support = EvidenceSpan(doc.doc_id, normalize_text(ev["text"]), *span)
            if any((s.doc_id, s.start, s.end) == (support.doc_id, *span) for s in spans):
                raise ValueError("동일한 근거 구간이 중복되었습니다")
            spans.append(support)
    else:
        if row.get("evidences") not in (None, []) or row.get("doc_id") or row.get("evidence"):
            raise ValueError("답변 불가 질문에는 근거를 지정하지 않습니다")
        if aliases or row.get("answer", NOT_FOUND) != NOT_FOUND:
            raise ValueError(f"답변 불가 질문의 answer는 생략하거나 '{NOT_FOUND}'여야 합니다")
        answer = NOT_FOUND

    first = spans[0] if spans else EvidenceSpan("", "", 0, 0)
    return QAItem(
        qid=qid, question=normalize_text(row["question"]).strip(), answer=answer,
        doc_id=first.doc_id, evidence=first.text, start=first.start, end=first.end,
        answerable=answerable, evidences=tuple(spans),
        answer_aliases=tuple(normalize_text(a).strip() for a in aliases),
        question_type=question_type.strip(), split=split, source_group=source_group.strip(),
    )


def load_qa(path: Path, documents: list[Document], split: str | None = None) -> list[QAItem]:
    """QA셋을 읽고 모든 항목의 근거가 문서에 실제로 존재하는지 검증한다.

    오류는 한 번에 모아서 보고한다 (질의셋 작성 중 반복 수정을 줄이기 위해).
    """
    if not path.is_file():
        raise FileNotFoundError(f"QA 파일이 없습니다: {path}")
    if split is not None and split not in SPLITS:
        raise ValueError(f"잘못된 QA split: {split}")
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
            if not isinstance(row, dict):
                errors.append(f"{lineno}행: 각 행은 JSON 객체여야 합니다")
                continue
            if isinstance(row.get("id"), bool) or not isinstance(row.get("id"), (str, int)):
                errors.append(f"{lineno}행: 필수 필드 누락 또는 잘못된 id")
                continue
            qid = str(row["id"]).strip()
            if not qid:
                errors.append(f"{lineno}행: 필수 필드 누락 ['id']")
                continue
            if qid in seen:
                errors.append(f"{lineno}행: 중복 id '{qid}'")
                continue
            seen.add(qid)
            try:
                items.append(_parse_item(row, qid, docs))
            except ValueError as e:
                errors.append(f"{lineno}행: {e}")

    groups: dict[str, set[str]] = {}
    for item in items:
        if item.source_group and item.split != "unspecified":
            groups.setdefault(item.source_group, set()).add(item.split)
    for group, splits in groups.items():
        if len(splits) > 1:
            errors.append(f"source_group '{group}'이 dev/test에 함께 있습니다: {sorted(splits)}")

    if errors:
        raise ValueError(f"QA셋 검증 실패 ({path}):\n  " + "\n  ".join(errors))
    if not items:
        raise ValueError(f"QA셋이 비어 있습니다: {path}")
    if split is not None:
        items = [item for item in items if item.split == split]
        if not items:
            raise ValueError(f"QA셋에 split='{split}'인 질문이 없습니다: {path}")
    return items
