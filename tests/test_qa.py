import json

import pytest

from rag_eval.corpus import Document, load_documents
from rag_eval.qa import load_qa, locate_evidence


def test_locate_evidence_ignores_whitespace():
    text = "앞부분.\n공용 GPU 서버는\n  예약해야 한다. 뒷부분"
    span = locate_evidence(text, "공용 GPU 서버는 예약해야 한다.")
    assert span is not None
    assert text[span[0] : span[1]].startswith("공용")


def test_sample_qa_is_valid(sample_docs_dir, sample_qa_path):
    items = load_qa(sample_qa_path, load_documents(sample_docs_dir))
    assert len(items) == 6
    assert all(it.end > it.start for it in items)


def test_errors_are_collected(tmp_path):
    docs = [Document("a.md", "실제 본문 문장입니다.")]
    rows = [
        {"id": "1", "question": "q", "answer": "a", "doc_id": "a.md", "evidence": "없는 문장"},
        {"id": "1", "question": "q", "answer": "a", "doc_id": "a.md", "evidence": "본문 문장"},
        {"id": "2", "question": "q", "answer": "a", "doc_id": "b.md", "evidence": "본문"},
        {"id": "3", "question": "", "answer": "a", "doc_id": "a.md", "evidence": "본문"},
    ]
    path = tmp_path / "qa.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n{bad", "utf-8")
    with pytest.raises(ValueError) as exc:
        load_qa(path, docs)
    msg = str(exc.value)
    for expected in ("1행", "2행: 중복 id", "3행", "4행: 필수 필드", "5행: JSON"):
        assert expected in msg
