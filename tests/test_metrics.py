from rag_eval.corpus import Chunk
from rag_eval.metrics import contains_answer, evidence_rank, summarize
from rag_eval.prompts import parse_judgement
from rag_eval.qa import QAItem

ITEM = QAItem("q", "질문", "48시간", "a.md", "근거", start=100, end=200)


def _chunk(doc, start, end):
    return Chunk(f"{doc}#{start}", doc, start, end, "x" * (end - start))


def test_evidence_rank_coverage_threshold():
    chunks = [_chunk("b.md", 100, 200), _chunk("a.md", 0, 120), _chunk("a.md", 90, 210)]
    assert evidence_rank(chunks, ITEM, min_coverage=0.5) == 3  # 20% 덮는 2번은 제외
    assert evidence_rank(chunks, ITEM, min_coverage=0.1) == 2
    assert evidence_rank(chunks[:2], ITEM, min_coverage=0.5) is None


def test_contains_answer_normalizes():
    assert contains_answer("최대 48 시간입니다.", "48시간")
    assert not contains_answer("24시간입니다", "48시간")
    assert not contains_answer("무엇이든", "")


def test_parse_judgement():
    assert parse_judgement('{"correct": true, "reason": "ok"}') is True
    assert parse_judgement('{"correct": "yes"}') is None
    assert parse_judgement("not json") is None


def test_summarize_groups_and_metrics():
    base = {"chunking": "300/50", "embed_model": "e", "reranker": "none", "llm_model": "m",
            "contains": True, "latency_s": 1.0, "answer": "a"}
    rows = [
        {**base, "evidence_rank": 1, "judge_correct": True},
        {**base, "evidence_rank": 2, "judge_correct": False},
        {**base, "evidence_rank": None, "judge_correct": None},
    ]
    (s,) = summarize(rows)
    assert s["n"] == 3
    assert abs(s["recall_at_k"] - 2 / 3) < 1e-9
    assert abs(s["mrr"] - (1 + 0.5 + 0) / 3) < 1e-9
    assert s["judge_acc"] == 0.5 and s["judged_n"] == 2
