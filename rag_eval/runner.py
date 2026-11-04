"""설정 조합(청크 × 임베딩 × reranker × LLM)을 순회하며 평가한다."""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from rag_eval.config import ExperimentConfig
from rag_eval.corpus import Chunk, chunk_corpus, load_documents
from rag_eval.metrics import contains_answer, evidence_rank, summarize
from rag_eval.prompts import build_answer_messages, build_judge_messages, parse_judgement
from rag_eval.qa import QAItem, load_qa
from rag_eval.report import write_summary
from rag_eval.retrieval import Embedder, Reranker, VectorIndex, embed_cached, rerank

log = logging.getLogger(__name__)


class ChatModel(Protocol):
    def generate(self, messages: list[dict[str, str]], json_mode: bool = False) -> str: ...


EmbedderFactory = Callable[[str], Embedder]
ChatFactory = Callable[[str], ChatModel]
RerankerFactory = Callable[[str], Reranker]


def _retrieve_all(
    cfg: ExperimentConfig,
    items: list[QAItem],
    index: VectorIndex,
    q_vecs,
    reranker: Reranker | None,
) -> list[list[Chunk]]:
    k = cfg.rerank_candidates if reranker else cfg.top_k
    results = []
    for item, vec in zip(items, q_vecs, strict=True):
        hits = index.search(vec, k)
        if reranker:
            hits = rerank(reranker, item.question, hits, cfg.top_k)
        results.append(hits)
    return results


def run_experiment(
    cfg: ExperimentConfig,
    embedder_factory: EmbedderFactory,
    chat_factory: ChatFactory,
    reranker_factory: RerankerFactory | None = None,
    config_path: Path | None = None,
) -> Path:
    docs = load_documents(cfg.docs_dir)
    items = load_qa(cfg.qa_path, docs)
    out_dir = _new_run_dir(cfg.output_dir, cfg.name)
    cache_dir = cfg.output_dir / ".cache"
    if config_path is not None:
        shutil.copy(config_path, out_dir / "config.yaml")
    log.info("문서 %d개, 질문 %d개 → %s", len(docs), len(items), out_dir)

    judge = chat_factory(cfg.judge_model) if cfg.judge_model else None
    rerankers = {
        name: (reranker_factory or _missing_reranker)(name) for name in cfg.rerankers if name
    }
    rows: list[dict[str, Any]] = []
    questions = [it.question for it in items]

    with (out_dir / "rows.jsonl").open("w", encoding="utf-8") as fout:
        for ch in cfg.chunking:
            chunks = chunk_corpus(docs, ch.size, ch.overlap)
            for embed_model in cfg.embed_models:
                embedder = embedder_factory(embed_model)
                vecs = embed_cached(embedder, embed_model, [c.text for c in chunks], cache_dir)
                q_vecs = embed_cached(embedder, embed_model, questions, cache_dir)
                index = VectorIndex(chunks, vecs)
                for rr_name in cfg.rerankers:
                    retrieved = _retrieve_all(cfg, items, index, q_vecs, rerankers.get(rr_name))
                    for llm_name in cfg.llm_models or [None]:
                        llm = chat_factory(llm_name) if llm_name else None
                        log.info("chunk=%s embed=%s rerank=%s llm=%s",
                                 ch.label, embed_model, rr_name, llm_name)
                        for item, hits in zip(items, retrieved, strict=True):
                            row = _evaluate_item(cfg, item, hits, llm, judge)
                            row.update(chunking=ch.label, embed_model=embed_model,
                                       reranker=rr_name or "none", llm_model=llm_name or "none")
                            rows.append(row)
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_summary(summarize(rows), out_dir, cfg.top_k)
    return out_dir


def _evaluate_item(
    cfg: ExperimentConfig,
    item: QAItem,
    hits: list[Chunk],
    llm: ChatModel | None,
    judge: ChatModel | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "qid": item.qid,
        "evidence_rank": evidence_rank(hits, item, cfg.min_evidence_coverage),
        "retrieved": [c.chunk_id for c in hits],
        "answer": None,
        "contains": None,
        "judge_correct": None,
        "latency_s": None,
    }
    if llm is None:
        return row
    t0 = time.perf_counter()
    answer = llm.generate(build_answer_messages(item.question, hits))
    row["latency_s"] = round(time.perf_counter() - t0, 3)
    row["answer"] = answer
    row["contains"] = contains_answer(answer, item.answer)
    if judge is not None:
        raw = judge.generate(build_judge_messages(item.question, item.answer, answer), json_mode=True)
        row["judge_correct"] = parse_judgement(raw)
    return row


def _new_run_dir(output_dir: Path, name: str) -> Path:
    """이전 결과를 덮어쓰지 않도록 항상 새 폴더를 만든다 (같은 초에 재실행하면 접미 번호)."""
    base = f"{name}-{datetime.now():%Y%m%d-%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1000):
        path = output_dir / (base if i == 0 else f"{base}-{i}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"결과 폴더를 만들 수 없습니다: {output_dir / base}")


def _missing_reranker(name: str) -> Reranker:
    raise ValueError(f"reranker '{name}'가 설정됐지만 reranker_factory가 없습니다")
