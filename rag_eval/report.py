"""결과를 CSV·Markdown으로 저장."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

COLUMNS = [
    ("chunking", "청크(size/overlap)"),
    ("embed_model", "임베딩"),
    ("reranker", "reranker"),
    ("llm_model", "LLM"),
    ("n", "n"),
    ("recall_at_k", "Recall@k"),
    ("mrr", "MRR"),
    ("contains_rate", "정답 포함률"),
    ("judge_acc", "채점 정확도"),
    ("mean_latency_s", "평균 지연(s)"),
]


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_summary(summary: list[dict[str, Any]], out_dir: Path, top_k: int) -> None:
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        writer.writeheader()
        writer.writerows(summary)

    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [f"# 결과 요약 (k={top_k})", "", header, sep]
    for s in summary:
        lines.append("| " + " | ".join(_fmt(s[key]) for key, _ in COLUMNS) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
