"""문서 로드와 청크 분할."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = (".txt", ".md")
# 청크 끝을 자를 때 선호하는 경계 문자 (앞쪽이 우선순위 높음).
_BOUNDARIES = ("\n\n", "\n", ". ", "다. ", "? ", "! ", " ")


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str


def normalize_text(text: str) -> str:
    """NFC 정규화와 줄바꿈 통일. 문서와 QA 근거 모두 같은 함수를 거친다."""
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")


def load_documents(docs_dir: Path) -> list[Document]:
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"문서 디렉터리가 없습니다: {docs_dir}")
    paths = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"{docs_dir}에 {'/'.join(SUPPORTED_SUFFIXES)} 문서가 없습니다")
    docs = []
    for p in paths:
        try:
            text = normalize_text(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError as e:
            raise ValueError(f"UTF-8로 읽을 수 없는 문서입니다: {p}") from e
        if text.strip():
            docs.append(Document(doc_id=p.relative_to(docs_dir).as_posix(), text=text))
    return docs


def _find_cut(text: str, start: int, hard_end: int) -> int:
    """[start, hard_end) 구간 뒤쪽 절반에서 가장 자연스러운 경계를 찾는다."""
    if hard_end >= len(text):
        return len(text)
    min_end = start + (hard_end - start) // 2
    for sep in _BOUNDARIES:
        pos = text.rfind(sep, min_end, hard_end)
        if pos != -1:
            return pos + len(sep)
    return hard_end


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    """문자 단위 슬라이딩 윈도우. 원문 위치(start, end)를 보존해 근거 매칭에 쓴다."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"잘못된 청크 설정: size={size}, overlap={overlap}")
    chunks: list[Chunk] = []
    text, start = doc.text, 0
    while start < len(text):
        end = _find_cut(text, start, start + size)
        piece = text[start:end]
        if piece.strip():
            chunks.append(
                Chunk(f"{doc.doc_id}#{len(chunks)}", doc.doc_id, start, end, piece)
            )
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_corpus(docs: list[Document], size: int, overlap: int) -> list[Chunk]:
    return [c for d in docs for c in chunk_document(d, size, overlap)]
