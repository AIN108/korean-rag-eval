"""명령행 진입점: `python -m rag_eval {validate,run} --config ...`"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rag_eval.config import ConfigError, EmbedSpec, load_config
from rag_eval.corpus import chunk_corpus, load_documents
from rag_eval.expansion import load_glossary
from rag_eval.ollama import OllamaChat, OllamaClient, OllamaEmbedder, OllamaError
from rag_eval.qa import load_qa


def cmd_validate(config_path: Path) -> int:
    cfg = load_config(config_path)
    docs = load_documents(cfg.docs_dir, text_cache_dir=cfg.output_dir / ".cache" / "text")
    items = load_qa(cfg.qa_path, docs)
    for path in cfg.query_expansions:
        if path:
            print(f"  용어집 {path.name}: {len(load_glossary(path))}개 항목")
    print(f"문서 {len(docs)}개, 질문 {len(items)}개 — 검증 통과")
    for ch in cfg.chunking:
        chunks = chunk_corpus(docs, ch.size, ch.overlap)
        avg = sum(len(c.text) for c in chunks) / len(chunks)
        print(f"  청크 {ch.label}: {len(chunks)}개, 평균 {avg:.0f}자")
    return 0


def cmd_run(config_path: Path) -> int:
    from rag_eval.runner import run_experiment

    cfg = load_config(config_path)
    client = OllamaClient(cfg.ollama_url)
    try:
        out = run_experiment(
            cfg,
            embedder_factory=lambda spec: _load_embedder(spec, client),
            chat_factory=lambda m: OllamaChat(client, m, cfg.temperature, cfg.seed),
            reranker_factory=_load_reranker,
            config_path=config_path,
        )
    finally:
        client.close()
    print(f"완료: {out / 'summary.md'}")
    return 0


def _load_embedder(spec: EmbedSpec, client: OllamaClient):
    if spec.backend == "hf":
        from rag_eval.retrieval import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(spec.model)
    return OllamaEmbedder(client, spec.model)


def _load_reranker(name: str):
    from rag_eval.retrieval import CrossEncoderReranker

    return CrossEncoderReranker(name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("validate", "문서·QA셋 검증"), ("run", "평가 실행")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return cmd_validate(args.config) if args.command == "validate" else cmd_run(args.config)
    except (ConfigError, FileNotFoundError, ValueError, OllamaError, ImportError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
