import json
from dataclasses import replace

import pytest
import yaml
from conftest import ROOT, FakeChat, FakeEmbedder

from rag_eval.config import load_config
from rag_eval.corpus import Chunk, Document, load_documents
from rag_eval.metrics import evidence_ranks, summarize
from rag_eval.ollama import OllamaError
from rag_eval.prompts import NOT_FOUND
from rag_eval.qa import EvidenceSpan, QAItem, load_qa
from rag_eval.runner import run_experiment


def _write_qa(tmp_path, rows):
    path = tmp_path / "qa.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), "utf-8")
    return path


def test_multiple_evidence_and_aliases(tmp_path):
    docs = [Document("a.md", "예약은 3일 전까지."), Document("b.md", "최대 48시간.")]
    row = {"id": "q", "question": "마감과 시간?", "answer": "3일 전, 48시간",
           "answer_aliases": ["사흘 전, 48시간"], "evidences": [
               {"doc_id": "a.md", "text": "예약은 3일 전까지."},
               {"doc_id": "b.md", "text": "최대 48시간."},
           ]}
    (item,) = load_qa(_write_qa(tmp_path, [row]), docs)
    assert len(item.supports) == 2
    assert item.answer_aliases == ("사흘 전, 48시간",)
    assert evidence_ranks([Chunk("a#0", "a.md", 0, len(docs[0].text), docs[0].text)], item, 0.5) == [1, None]


def test_repeated_evidence_selects_requested_occurrence(tmp_path):
    row = {"id": "q", "question": "두 번째?", "answer": "48시간", "evidences": [
        {"doc_id": "a.md", "text": "48시간", "occurrence": 2}]}
    (item,) = load_qa(_write_qa(tmp_path, [row]), [Document("a.md", "48시간 뒤에 48시간")])
    assert item.start == 8


@pytest.mark.parametrize("row", [
    [], None, 42,
    {"id": "q", "question": None},
    {"id": "q", "question": "q", "answerable": "false"},
    {"id": "q", "question": "q", "answerable": False, "evidences": [{"doc_id": "a", "text": "x"}]},
    {"id": "q", "question": "q", "answerable": False, "answer": "아무 답변"},
    {"id": "q", "question": "q", "answer": "a", "evidences": []},
    {"id": "q", "question": "q", "answer": "a", "evidences": ["x"]},
    {"id": "q", "question": "q", "answer": "a", "evidences": [
        {"doc_id": "a.md", "text": "본문", "occurrence": 0}]},
    {"id": "q", "question": "q", "answer": "a", "answer_aliases": "별칭",
     "doc_id": "a.md", "evidence": "본문"},
    {"id": "q", "question": "q", "answer": "a", "split": "unknown",
     "doc_id": "a.md", "evidence": "본문"},
    {"id": "q", "question": "q", "answer": "a", "doc_id": "a.md", "evidence": "본문",
     "evidences": [{"doc_id": "a.md", "text": "본문"}]},
    {"id": "q", "question": "q", "answer": "a", "evidences": [
        {"doc_id": "a.md", "text": "본문"}, {"doc_id": "a.md", "text": "본문"}]},
])
def test_malformed_qa_reports_line_instead_of_crashing(tmp_path, row):
    with pytest.raises(ValueError, match="1행"):
        load_qa(_write_qa(tmp_path, [row]), [Document("a.md", "본문")])


def test_split_leakage_is_checked_before_filtering(tmp_path):
    rows = [{"id": split, "question": "q", "answerable": False, "split": split,
             "source_group": "same-publication"} for split in ("dev", "test")]
    with pytest.raises(ValueError, match="source_group.*dev/test"):
        load_qa(_write_qa(tmp_path, rows), [Document("a.md", "본문")], split="dev")


def test_demo_splits_and_counts():
    docs = load_documents(ROOT / "data/examples/docs")
    path = ROOT / "data/examples/qa.jsonl"
    all_items = load_qa(path, docs)
    dev = load_qa(path, docs, split="dev")
    test = load_qa(path, docs, split="test")
    assert len(all_items) == 12 and len(dev) == len(test) == 6
    assert sum(not it.answerable for it in all_items) == 2
    assert {it.source_group for it in dev}.isdisjoint(it.source_group for it in test)


def test_demo_runs_with_abstention_and_multiple_evidence(tmp_path):
    class DemoChat(FakeChat):
        def generate(self, messages, json_mode=False):
            if json_mode:
                return super().generate(messages, json_mode)
            if "1시간 사용하는 요금" in messages[-1]["content"]:
                return NOT_FOUND
            return "사흘 전까지"

    cfg = replace(load_config(ROOT / "configs/schema_demo.yaml"),
                  docs_dir=ROOT / "data/examples/docs", qa_path=ROOT / "data/examples/qa.jsonl",
                  output_dir=tmp_path / "results", llm_models=["fake"], judge_model="fake-judge")
    judge = FakeChat()
    out = run_experiment(cfg, lambda spec: FakeEmbedder(),
                         lambda name: judge if name == "fake-judge" else DemoChat())
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 6 and all(r["split"] == "dev" for r in rows)
    assert rows[0]["contains"] is True  # 허용한 정답 별칭
    assert len(rows[2]["evidence_ranks"]) == 2
    assert rows[-1]["abstained"] is True and rows[-1]["judge_status"] == "not_applicable"
    assert judge.calls == [True] * 5  # 답변 불가 문항은 LLM 채점하지 않는다
    (summary,) = summarize(rows)
    assert summary["retrieval_n"] == 5 and summary["abstention_acc"] == 1.0


def _row(**overrides):
    return {"chunking": "300/50", "embed_model": "e", "query_expansion": "none",
            "reranker": "none", "llm_model": "m", "answerable": True,
            "evidence_rank": 1, "evidence_recall": 1.0, "all_evidence_found": True,
            "answer": "a", "contains": True, "latency_s": 1.0, "judge_correct": True,
            "judge_status": "ok", "evaluation_status": "ok", **overrides}


def test_judge_failure_denominators_are_visible():
    rows = [_row()] + [_row(judge_correct=None, judge_status="invalid_response") for _ in range(9)]
    (summary,) = summarize(rows)
    assert summary["judge_acc"] == 1.0
    assert summary["judge_requested_n"] == 10 and summary["judged_n"] == 1
    assert summary["judge_failed_n"] == 9
    assert summary["judge_coverage"] == summary["judge_correct_rate"] == 0.1


def test_partial_multi_evidence_does_not_count_as_all_found():
    (summary,) = summarize([_row(evidence_recall=0.5, all_evidence_found=False)])
    assert summary["recall_at_k"] == 1.0
    assert summary["evidence_recall_at_k"] == 0.5
    assert summary["all_evidence_hit_at_k"] == 0.0


def test_unanswerable_is_excluded_from_retrieval_and_judge_denominators():
    negative = _row(answerable=False, evidence_rank=None, evidence_recall=None,
                    all_evidence_found=None, judge_correct=None, judge_status="not_applicable",
                    answer=NOT_FOUND, contains=None)
    (summary,) = summarize([_row(), negative])
    assert summary["n"] == 2 and summary["retrieval_n"] == 1
    assert summary["recall_at_k"] == summary["abstention_acc"] == 1.0
    assert summary["judge_requested_n"] == 1
    (negative_only,) = summarize([negative])
    assert negative_only["recall_at_k"] is negative_only["mrr"] is None


def _config(tmp_path, sample_docs_dir, sample_qa_path):
    cfg = {"name": "regression", "docs_dir": str(sample_docs_dir), "qa_path": str(sample_qa_path),
           "output_dir": str(tmp_path / "results"), "chunking": [{"size": 200, "overlap": 30}],
           "embed_models": ["fake"], "llm_models": ["fake-llm"], "judge_model": "fake-judge"}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), "utf-8")
    return load_config(path)


def test_invalid_judge_response_survives_and_marks_manifest(tmp_path, sample_docs_dir, sample_qa_path):
    cfg = _config(tmp_path, sample_docs_dir, sample_qa_path)
    out = run_experiment(cfg, lambda spec: FakeEmbedder(), lambda name: FakeChat(judge="bad json"))
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 6 and all(r["judge_raw"] == "bad json" for r in rows)
    (summary,) = summarize(rows)
    assert summary["judge_failed_n"] == 6 and summary["judge_acc"] is None
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["status"] == "completed_with_errors" and manifest["row_count"] == 6
    assert manifest["qa_sha256"] and manifest["corpus"][0]["text_sha256"]
    report = (out / "summary.md").read_text("utf-8")
    assert "전체 채점 시도" in report and "채점 유효율" in report
    assert (out / "summary_by_type.csv").is_file()


@pytest.mark.parametrize("failure_stage", ["answer", "judge"])
def test_request_errors_preserve_rows(tmp_path, sample_docs_dir, sample_qa_path, failure_stage):
    class BrokenChat(FakeChat):
        def generate(self, messages, json_mode=False):
            if (failure_stage == "judge") == json_mode:
                raise OllamaError("timeout")
            return super().generate(messages, json_mode)

    cfg = _config(tmp_path, sample_docs_dir, sample_qa_path)
    out = run_experiment(cfg, lambda spec: FakeEmbedder(), lambda name: BrokenChat())
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text("utf-8").splitlines()]
    (summary,) = summarize(rows)
    assert len(rows) == 6 and all(r["error"] == "timeout" for r in rows)
    if failure_stage == "answer":
        assert summary["generation_error_n"] == 6 and summary["generation_success_rate"] == 0.0
    else:
        assert summary["judge_failed_n"] == 6 and summary["judge_coverage"] == 0.0


def test_unexpected_error_marks_manifest_failed(tmp_path, sample_docs_dir, sample_qa_path):
    class BrokenChat(FakeChat):
        def generate(self, messages, json_mode=False):
            raise RuntimeError("unexpected error")

    cfg = _config(tmp_path, sample_docs_dir, sample_qa_path)
    with pytest.raises(RuntimeError, match="unexpected"):
        run_experiment(cfg, lambda spec: FakeEmbedder(), lambda name: BrokenChat())
    manifests = list(cfg.output_dir.glob("*/manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text("utf-8"))["status"] == "failed"


def test_multi_evidence_item_keeps_legacy_constructor():
    item = QAItem("q", "질문", "답변", evidences=(EvidenceSpan("a", "x", 0, 1),))
    assert evidence_ranks([Chunk("a#0", "a", 0, 1, "x")], item, 0.5) == [1]
