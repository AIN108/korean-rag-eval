"""검색·답변 지표와 결과 집계."""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean
from typing import Any

from rag_eval.corpus import Chunk
from rag_eval.prompts import NOT_FOUND
from rag_eval.qa import QAItem

GROUP_KEYS = ("chunking", "embed_model", "query_expansion", "reranker", "llm_model")
_STRIP = re.compile(r"[\s\.,·!?\"'()\[\]~\-]+")


def evidence_rank(chunks: list[Chunk], item: QAItem, min_coverage: float) -> int | None:
    """필요한 근거 중 하나를 포함한 첫 청크 순위. 기존 단일 근거 동작을 유지한다."""
    found = [rank for rank in evidence_ranks(chunks, item, min_coverage) if rank is not None]
    return min(found) if found else None


def evidence_ranks(chunks: list[Chunk], item: QAItem, min_coverage: float) -> list[int | None]:
    """각 필수 근거의 첫 hit 순위. 답변 불가 질문에는 검색 정답 근거가 없어 빈 리스트."""
    ranks = []
    for support in item.supports:
        length = support.end - support.start
        found = None
        for rank, chunk in enumerate(chunks, start=1):
            if chunk.doc_id != support.doc_id or length <= 0:
                continue
            overlap = min(chunk.end, support.end) - max(chunk.start, support.start)
            if overlap > 0 and overlap / length >= min_coverage:
                found = rank
                break
        ranks.append(found)
    return ranks


def _compact(text: str) -> str:
    return _STRIP.sub("", text).lower()


def contains_answer(prediction: str, reference: str) -> bool:
    """공백·문장부호를 무시한 포함 여부. 짧은 사실형 정답에 대한 보조 지표."""
    ref = _compact(reference)
    return bool(ref) and ref in _compact(prediction)


def is_abstention(prediction: str) -> bool:
    """프롬프트가 요구하는 정해진 답변 불가 문구와의 일치 여부."""
    return _compact(prediction) == _compact(NOT_FOUND)


def summarize(
    rows: list[dict[str, Any]], extra_group_keys: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    group_keys = GROUP_KEYS + extra_group_keys
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        # 이전 버전 결과(rows.jsonl)에는 없는 키가 있을 수 있어 "none"으로 채운다
        groups[tuple(r.get(k, "none") for k in group_keys)].append(r)

    summary = []
    for key, rs in groups.items():
        retrieval = [r for r in rs if r.get("answerable", True)]
        ranks = [r["evidence_rank"] for r in retrieval]
        judged = [r["judge_correct"] for r in rs if r.get("judge_correct") is not None]
        answered = [r for r in rs if r.get("answer") is not None]
        positives = [r for r in answered if r.get("answerable", True)]
        negatives = [r for r in answered if not r.get("answerable", True)]
        # 이전 결과에 status가 없고 채점된 행이 하나라도 있으면 같은 조건의 나머지도 분모에 포함한다.
        requested = [
            r for r in positives
            if r.get("judge_status", "legacy" if judged else "not_requested")
            not in ("not_requested", "not_applicable")
        ]
        generation_n = sum(
            r.get("evaluation_status") != "retrieval_only" and r.get("llm_model") != "none" for r in rs
        )
        contains = [r["contains"] for r in positives if r.get("contains") is not None]
        summary.append(
            {
                **dict(zip(group_keys, key, strict=True)),
                "n": len(rs),
                "retrieval_n": len(retrieval),
                "unanswerable_n": len(rs) - len(retrieval),
                "recall_at_k": float(mean(rk is not None for rk in ranks)) if ranks else None,
                "mrr": mean(1 / rk if rk else 0.0 for rk in ranks) if ranks else None,
                "evidence_recall_at_k": mean(
                    r.get("evidence_recall", float(r["evidence_rank"] is not None)) for r in retrieval
                ) if retrieval else None,
                "all_evidence_hit_at_k": float(mean(
                    r.get("all_evidence_found", r["evidence_rank"] is not None) for r in retrieval
                )) if retrieval else None,
                "contains_rate": float(mean(contains)) if contains else None,
                "judge_acc": float(mean(judged)) if judged else None,
                "judged_n": len(judged),
                "judge_requested_n": len(requested),
                "judge_failed_n": len(requested) - len(judged),
                "judge_coverage": len(judged) / len(requested) if requested else None,
                "judge_correct_rate": sum(judged) / len(requested) if requested else None,
                "answer_n": len(answered),
                "generation_error_n": sum(r.get("evaluation_status") == "generation_error" for r in rs),
                "generation_success_rate": len(answered) / generation_n if generation_n else None,
                "abstention_n": len(negatives),
                "abstention_acc": float(mean(is_abstention(r["answer"]) for r in negatives))
                if negatives else None,
                "false_abstention_rate": float(mean(is_abstention(r["answer"]) for r in positives))
                if positives else None,
                "mean_latency_s": mean(r["latency_s"] for r in answered) if answered else None,
                "mean_search_s": mean(r["search_s"] for r in rs if r.get("search_s") is not None)
                if any(r.get("search_s") is not None for r in rs) else None,
                "mean_rerank_s": mean(r["rerank_s"] for r in rs if r.get("rerank_s") is not None)
                if any(r.get("rerank_s") is not None for r in rs) else None,
            }
        )
    summary.sort(key=lambda s: (
        -(s["judge_correct_rate"] or 0), -(s["all_evidence_hit_at_k"] or 0), -(s["mrr"] or 0)
    ))
    return summary
