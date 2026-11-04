"""문서 로드와 청크 분할."""

from __future__ import annotations

import hashlib
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".txt", ".md", ".pdf")
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


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("PDF 문서에는 pypdf가 필요합니다: pip install -e '.[pdf]'") from e
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _read_pdf_cached(path: Path, cache_dir: Path | None) -> str:
    """PDF 텍스트 추출은 느리므로 (경로, 크기, 수정 시각) 기준으로 결과를 캐시한다."""
    if cache_dir is None:
        return _read_pdf(path)
    stat = path.stat()
    key = hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]
    cached = cache_dir / f"{key}.txt"
    if cached.is_file():
        return cached.read_text(encoding="utf-8")
    text = _read_pdf(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(text, encoding="utf-8")
    return text


def _read_document(path: Path, cache_dir: Path | None) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf_cached(path, cache_dir)
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"UTF-8로 읽을 수 없는 문서입니다: {path}") from e


def load_documents(docs_dir: Path, text_cache_dir: Path | None = None) -> list[Document]:
    """docs_dir 아래 .txt/.md/.pdf 문서를 읽는다. 텍스트가 없는 문서(스캔 PDF 등)는 경고 후 건너뛴다."""
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"문서 디렉터리가 없습니다: {docs_dir}")
    paths = sorted(p for p in docs_dir.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"{docs_dir}에 {'/'.join(SUPPORTED_SUFFIXES)} 문서가 없습니다")
    docs = []
    for p in paths:
        text = normalize_text(_read_document(p, text_cache_dir))
        if text.strip():
            docs.append(Document(doc_id=p.relative_to(docs_dir).as_posix(), text=text))
        else:
            log.warning("텍스트를 추출하지 못해 건너뜁니다 (스캔 PDF일 수 있음): %s", p)
    if not docs:
        raise ValueError(f"{docs_dir}에서 텍스트를 추출한 문서가 없습니다")
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
