"""검색·답변 지표와 결과 집계."""

from __future__ import annotations

import re
from collections import defaultdict
from statistics import mean
from typing import Any

from rag_eval.corpus import Chunk
from rag_eval.qa import QAItem

GROUP_KEYS = ("chunking", "embed_model", "reranker", "llm_model")
_STRIP = re.compile(r"[\s\.,·!?\"'()\[\]~\-]+")


def evidence_rank(chunks: list[Chunk], item: QAItem, min_coverage: float) -> int | None:
    """근거 구간을 min_coverage 이상 덮는 첫 청크의 순위(1부터). 없으면 None."""
    length = item.end - item.start
    for rank, c in enumerate(chunks, start=1):
        if c.doc_id != item.doc_id:
            continue
        overlap = min(c.end, item.end) - max(c.start, item.start)
        if overlap > 0 and overlap / length >= min_coverage:
            return rank
    return None


def _compact(text: str) -> str:
    return _STRIP.sub("", text).lower()


def contains_answer(prediction: str, reference: str) -> bool:
    """공백·문장부호를 무시한 포함 여부. 짧은 사실형 정답에 대한 보조 지표."""
    ref = _compact(reference)
    return bool(ref) and ref in _compact(prediction)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in GROUP_KEYS)].append(r)

    summary = []
    for key, rs in groups.items():
        ranks = [r["evidence_rank"] for r in rs]
        judged = [r["judge_correct"] for r in rs if r.get("judge_correct") is not None]
        answered = [r for r in rs if r.get("answer") is not None]
        summary.append(
            {
                **dict(zip(GROUP_KEYS, key, strict=True)),
                "n": len(rs),
                "recall_at_k": float(mean(rk is not None for rk in ranks)),
                "mrr": mean(1 / rk if rk else 0.0 for rk in ranks),
                "contains_rate": float(mean(r["contains"] for r in answered)) if answered else None,
                "judge_acc": float(mean(judged)) if judged else None,
                "judged_n": len(judged),
                "mean_latency_s": mean(r["latency_s"] for r in answered) if answered else None,
            }
        )
    summary.sort(key=lambda s: (-(s["judge_acc"] or 0), -s["recall_at_k"], -s["mrr"]))
    return summary
