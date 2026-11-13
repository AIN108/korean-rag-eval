import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from rag_eval.cli import cmd_validate
from rag_eval.corpus import load_documents
from rag_eval.qa import load_qa

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("version,document_n,question_n", [(1, 3, 30), (2, 4, 40)])
def test_bundled_corpus_matches_registered_sources(version, document_n, question_n):
    data = ROOT / f"data/country_studies_pilot_v{version}"
    docs = load_documents(data / "docs")
    sources = [json.loads(line) for line in (data / "sources.jsonl").read_text("utf-8").splitlines()]
    assert len(docs) == len(sources) == document_n
    hashes = {doc.doc_id: hashlib.sha256(doc.text.encode()).hexdigest() for doc in docs}
    assert hashes == {source["doc_id"]: source["text_sha256"] for source in sources}
    items = load_qa(data / "qa.jsonl", docs)
    assert len(items) == question_n
    assert Counter(item.question_type for item in items) == {
        kind: question_n // 5
        for kind in ("fact", "numeric", "comparison", "multi_evidence", "unanswerable")
    }


@pytest.mark.parametrize("config,expected_n", [
    ("country_studies_pilot_dev.yaml", 20),
    ("country_studies_pilot_test.yaml", 10),
    ("country_studies_pilot_v2_dev.yaml", 20),
    ("country_studies_pilot_v2_test.yaml", 20),
])
def test_configs_validate_without_private_files(monkeypatch, capsys, config, expected_n):
    monkeypatch.chdir(ROOT)
    assert cmd_validate(ROOT / "configs" / config) == 0
    assert f"질문 {expected_n}개" in capsys.readouterr().out


@pytest.mark.parametrize("version", [1, 2])
def test_curated_evidence_matches_annotated_pdf_pages(version):
    data = ROOT / f"data/country_studies_pilot_v{version}"
    rows = [json.loads(line) for line in (data / "qa.jsonl").read_text("utf-8").splitlines()]
    items = load_qa(data / "qa.jsonl", load_documents(data / "docs"))
    page_index = json.loads((data / "page_index.json").read_text("utf-8"))
    for item, row in zip(items, rows, strict=True):
        for span, evidence in zip(item.supports, row.get("evidences", []), strict=True):
            page = page_index[span.doc_id][evidence["pdf_page"] - 1]
            assert page["start"] <= span.start < span.end <= page["end"]


def test_v2_preserves_v1_questions_and_source_groups():
    v1 = ROOT / "data/country_studies_pilot_v1"
    v2 = ROOT / "data/country_studies_pilot_v2"
    old = [json.loads(line) for line in (v1 / "qa.jsonl").read_text("utf-8").splitlines()]
    new = [json.loads(line) for line in (v2 / "qa.jsonl").read_text("utf-8").splitlines()]
    assert new[:30] == old
    assert len(new) == 40 and len({row["id"] for row in new}) == 40
    dev = {row["source_group"] for row in new if row["split"] == "dev"}
    test = {row["source_group"] for row in new if row["split"] == "test"}
    assert dev.isdisjoint(test)
