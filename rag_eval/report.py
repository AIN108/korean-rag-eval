"""결과를 CSV·Markdown으로 저장."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

CONDITIONS = [
    ("chunking", "청크(size/overlap)"),
    ("embed_model", "임베딩"),
    ("query_expansion", "질의 확장"),
    ("reranker", "reranker"),
    ("llm_model", "LLM"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _table(summary: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    header = "| 조건 | " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in range(len(columns) + 1)) + "|"
    lines = [header, sep]
    for number, row in enumerate(summary, start=1):
        lines.append(f"| C{number} | " + " | ".join(_fmt(row[key]) for key, _ in columns) + " |")
    return lines


def write_summary(
    summary: list[dict[str, Any]], out_dir: Path, top_k: int, stem: str = "summary"
) -> None:
    with (out_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        writer.writeheader()
        writer.writerows(summary)

    conditions = CONDITIONS + ([("question_type", "질문 유형")] if summary and
                               "question_type" in summary[0] else [])
    lines = [f"# 결과 요약 (k={top_k})", "", "비율은 0~1이며, `-`는 평가 대상이 없음을 뜻합니다.",
             "", "## 실험 조건", "", *_table(summary, conditions)]
    sections = [
        ("검색 — 답변 가능한 질문만 집계", [
            ("retrieval_n", "질문 수"), ("recall_at_k", "근거 하나 이상 hit@k"),
            ("evidence_recall_at_k", "필수 근거 평균 회수율"),
            ("all_evidence_hit_at_k", "모든 근거 hit@k"), ("mrr", "MRR"),
        ]),
        ("답변 생성 및 채점", [
            ("answer_n", "생성 성공"), ("generation_error_n", "생성 오류"),
            ("contains_rate", "정답 포함률(보조)"), ("judge_acc", "유효 채점 정확도"),
            ("judged_n", "유효 채점 수"), ("judge_requested_n", "전체 채점 시도"),
            ("judge_failed_n", "채점 실패"), ("judge_coverage", "채점 유효율"),
            ("judge_correct_rate", "확인된 정답/전체 채점 시도"),
        ]),
        ("답변 불가 및 잘못된 답변 거부", [
            ("unanswerable_n", "답변 불가 질문 수"), ("abstention_n", "그중 생성 성공"),
            ("abstention_acc", "답변 불가 정답률"),
            ("false_abstention_rate", "답변 가능한 질문의 거부율"),
        ]),
        ("단계별 평균 시간 — 초", [
            ("mean_search_s", "벡터 검색"), ("mean_rerank_s", "재정렬"),
            ("mean_latency_s", "답변 생성"),
        ]),
    ]
    for title, columns in sections:
        lines.extend(["", f"## {title}", "", *_table(summary, columns)])
    lines.extend([
        "", "유효 채점 정확도의 분모는 해석 가능한 채점 수입니다. 채점 실패를 오답으로 확정하지 않습니다.",
        "전체 시도 대비 확인된 정답 비율과 채점 유효율을 함께 확인하세요.",
        "답변 불가 질문은 검색 회수율과 LLM 채점에서 제외하고, 정해진 거부 문구의 일치율로 평가합니다.",
        "검색/재정렬 시간에는 임베딩 계산·모델 적재가 포함되지 않습니다. 전체 응답 시간으로 해석하지 마세요.",
    ])
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
