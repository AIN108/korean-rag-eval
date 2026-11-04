import pytest

from rag_eval.corpus import Document, chunk_document, load_documents


def test_chunks_cover_text_and_keep_offsets():
    text = "가나다라마. " * 50
    doc = Document("d.md", text)
    chunks = chunk_document(doc, size=40, overlap=10)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for c in chunks:
        assert text[c.start : c.end] == c.text
        assert len(c.text) <= 40
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert cur.start <= prev.end  # 빈틈 없음


def test_chunk_prefers_sentence_boundary():
    doc = Document("d.md", "첫 문장입니다. 두 번째 문장은 조금 더 깁니다. 세 번째.")
    first = chunk_document(doc, size=30, overlap=0)[0]
    assert first.text.endswith("깁니다. ")


def test_short_document_single_chunk():
    chunks = chunk_document(Document("d.md", "짧은 문서"), size=100, overlap=10)
    assert [c.text for c in chunks] == ["짧은 문서"]


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_config(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(Document("d.md", "abc"), size, overlap)


def test_load_documents(sample_docs_dir, tmp_path):
    docs = load_documents(sample_docs_dir)
    assert [d.doc_id for d in docs] == ["sample_lab_rules.md"]
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path)
