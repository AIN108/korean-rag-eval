"""답변 생성·채점 프롬프트."""

from __future__ import annotations

import json

from rag_eval.corpus import Chunk

NOT_FOUND = "문서에서 찾을 수 없습니다"

ANSWER_SYSTEM = (
    "당신은 주어진 문맥만 근거로 한국어로 답하는 도우미입니다. "
    f"문맥에 답이 없으면 정확히 '{NOT_FOUND}'라고만 답하세요. "
    "답은 1~2문장으로 간결하게 쓰세요."
)

JUDGE_SYSTEM = (
    "당신은 질의응답 채점자입니다. 모델 답변이 정답과 의미상 일치하는지 판단하세요. "
    "표현이 달라도 핵심 사실(수치, 기간, 대상, 조건)이 같으면 정답입니다. "
    '반드시 JSON으로만 답하세요: {"correct": true 또는 false, "reason": "한 문장"}'
)


def build_answer_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    context = "\n\n".join(f"[문맥 {i + 1}] ({c.doc_id})\n{c.text.strip()}" for i, c in enumerate(chunks))
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"{context}\n\n질문: {question}"},
    ]


def build_judge_messages(question: str, reference: str, prediction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": f"질문: {question}\n정답: {reference}\n모델 답변: {prediction}",
        },
    ]


def parse_judgement(raw: str) -> bool | None:
    """채점 모델 출력을 bool로 변환한다. 해석 불가하면 None (집계에서 제외)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    value = data.get("correct") if isinstance(data, dict) else None
    return value if isinstance(value, bool) else None
