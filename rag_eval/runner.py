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

from rag_eval.config import EmbedSpec, ExperimentConfig
from rag_eval.corpus import Chunk, chunk_corpus, load_documents
from rag_eval.expansion import expand_query, expansion_label, load_glossary
from rag_eval.metrics import contains_answer, evidence_rank, evidence_ranks, is_abstention, summarize
from rag_eval.ollama import OllamaError
from rag_eval.prompts import build_answer_messages, build_judge_messages, parse_judgement
from rag_eval.provenance import record_run
from rag_eval.qa import QAItem, load_qa
from rag_eval.report import write_summary
from rag_eval.retrieval import Embedder, Reranker, VectorIndex, embed_cached, rerank

log = logging.getLogger(__name__)


class ChatModel(Protocol):
    def generate(self, messages: list[dict[str, str]], json_mode: bool = False) -> str: ...


EmbedderFactory = Callable[[EmbedSpec], Embedder]
ChatFactory = Callable[[str], ChatModel]
RerankerFactory = Callable[[str], Reranker]


def _retrieve_all(
    cfg: ExperimentConfig,
    queries: list[str],
    index: VectorIndex,
    q_vecs,
    reranker: Reranker | None,
) -> list[tuple[list[Chunk], float, float]]:
    k = cfg.rerank_candidates if reranker else cfg.top_k
    results = []
    for query, vec in zip(queries, q_vecs, strict=True):
        started = time.perf_counter()
        hits = index.search(vec, k)
        search_s = time.perf_counter() - started
        rerank_s = 0.0
        if reranker:
            started = time.perf_counter()
            hits = rerank(reranker, query, hits, cfg.top_k)
            rerank_s = time.perf_counter() - started
        results.append((hits, search_s, rerank_s))
    return results


def run_experiment(
    cfg: ExperimentConfig,
    embedder_factory: EmbedderFactory,
    chat_factory: ChatFactory,
    reranker_factory: RerankerFactory | None = None,
    config_path: Path | None = None,
) -> Path:
    cache_dir = cfg.output_dir / ".cache"
    docs = load_documents(cfg.docs_dir, text_cache_dir=cache_dir / "text")
    items = load_qa(cfg.qa_path, docs, split=cfg.qa_split)
    out_dir = _new_run_dir(cfg.output_dir, cfg.name)
    if config_path is not None:
        shutil.copy(config_path, out_dir / "config.yaml")
    log.info("문서 %d개, 질문 %d개 → %s", len(docs), len(items), out_dir)

    judge = chat_factory(cfg.judge_model) if cfg.judge_model else None
    if cfg.judge_model and cfg.judge_model in cfg.llm_models:
        log.warning("답변과 채점에 같은 모델 '%s'가 포함되어 있습니다", cfg.judge_model)
    rerankers = {
        name: (reranker_factory or _missing_reranker)(name) for name in cfg.rerankers if name
    }
    # 확장은 검색 질의에만 적용하고, LLM에는 원래 질문을 준다 (답변 프롬프트를 조건 간 동일하게 유지).
    expanded = {
        path: [expand_query(it.question, load_glossary(path) if path else None) for it in items]
        for path in cfg.query_expansions
    }
    rows: list[dict[str, Any]] = []
    embedders: dict[str, Embedder] = {}

    with record_run(cfg, docs, items, out_dir) as manifest, (out_dir / "rows.jsonl").open(
        "w", encoding="utf-8"
    ) as fout:
        for ch in cfg.chunking:
            chunks = chunk_corpus(docs, ch.size, ch.overlap)
            for spec in cfg.embed_models:
                if spec.cache_id not in embedders:  # 접두어만 다른 조건은 모델을 다시 로드하지 않는다
                    embedders[spec.cache_id] = embedder_factory(spec)
                embedder = embedders[spec.cache_id]
                passages = [spec.passage_prefix + c.text for c in chunks]
                index = VectorIndex(chunks, embed_cached(embedder, spec.cache_id, passages, cache_dir))
                for exp_path, queries in expanded.items():
                    prefixed = [spec.query_prefix + q for q in queries]
                    q_vecs = embed_cached(embedder, spec.cache_id, prefixed, cache_dir)
                    for rr_name in cfg.rerankers:
                        retrieved = _retrieve_all(cfg, queries, index, q_vecs, rerankers.get(rr_name))
                        for llm_name in cfg.llm_models or [None]:
                            llm = chat_factory(llm_name) if llm_name else None
                            labels = {
                                "chunking": ch.label,
                                "embed_model": spec.name,
                                "query_expansion": expansion_label(exp_path),
                                "reranker": rr_name or "none",
                                "llm_model": llm_name or "none",
                            }
                            log.info("%s", labels)
                            for item, (hits, search_s, rerank_s) in zip(items, retrieved, strict=True):
                                row = {**_evaluate_item(cfg, item, hits, llm, judge), **labels}
                                row.update(search_s=search_s, rerank_s=rerank_s)
                                rows.append(row)
                                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                                fout.flush()
                                manifest["row_count"] = len(rows)

        write_summary(summarize(rows), out_dir, cfg.top_k)
        write_summary(summarize(rows, ("question_type",)), out_dir, cfg.top_k, stem="summary_by_type")
        if any(r["evaluation_status"] == "generation_error" or r["judge_status"] in
               ("invalid_response", "request_error") for r in rows):
            manifest["status"] = "completed_with_errors"
    return out_dir


def _evaluate_item(
    cfg: ExperimentConfig,
    item: QAItem,
    hits: list[Chunk],
    llm: ChatModel | None,
    judge: ChatModel | None,
) -> dict[str, Any]:
    ranks = evidence_ranks(hits, item, cfg.min_evidence_coverage)
    found = sum(rank is not None for rank in ranks)
    row: dict[str, Any] = {
        "qid": item.qid,
        "answerable": item.answerable,
        "question_type": item.question_type,
        "split": item.split,
        "source_group": item.source_group,
        "evidence_rank": evidence_rank(hits, item, cfg.min_evidence_coverage),
        "evidence_ranks": ranks,
        "evidence_recall": found / len(ranks) if ranks else None,
        "all_evidence_found": found == len(ranks) if ranks else None,
        "retrieved": [c.chunk_id for c in hits],
        "answer": None,
        "contains": None,
        "judge_correct": None,
        "judge_status": "not_requested" if item.answerable else "not_applicable",
        "judge_raw": None,
        "evaluation_status": "retrieval_only" if llm is None else "ok",
        "error": None,
        "abstained": None,
        "latency_s": None,
    }
    if llm is None:
        return row
    t0 = time.perf_counter()
    try:
        answer = llm.generate(build_answer_messages(item.question, hits))
    except OllamaError as e:
        row.update(evaluation_status="generation_error", error=str(e))
        log.warning("답변 생성 실패 (%s): %s", item.qid, e)
        return row
    row["latency_s"] = round(time.perf_counter() - t0, 3)
    row["answer"] = answer
    row["abstained"] = is_abstention(answer)
    if item.answerable:
        row["contains"] = any(contains_answer(answer, ref) for ref in (item.answer, *item.answer_aliases))
    if judge is not None and item.answerable:
        try:
            raw = judge.generate(build_judge_messages(item.question, item.answer, answer), json_mode=True)
        except OllamaError as e:
            row.update(judge_status="request_error", error=str(e))
            log.warning("채점 요청 실패 (%s): %s", item.qid, e)
            return row
        row["judge_raw"] = raw
        row["judge_correct"] = parse_judgement(raw)
        row["judge_status"] = "ok" if row["judge_correct"] is not None else "invalid_response"
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
