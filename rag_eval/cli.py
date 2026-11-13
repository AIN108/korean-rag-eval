"""명령행 진입점: `python -m rag_eval {validate,run} --config ...`"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
from collections import Counter
from pathlib import Path

from rag_eval.config import ConfigError, EmbedSpec, load_config
from rag_eval.corpus import chunk_corpus, load_documents
from rag_eval.expansion import load_glossary
from rag_eval.ollama import OllamaChat, OllamaClient, OllamaEmbedder, OllamaError
from rag_eval.qa import load_qa


def cmd_validate(config_path: Path) -> int:
    cfg = load_config(config_path)
    docs = load_documents(cfg.docs_dir, text_cache_dir=cfg.output_dir / ".cache" / "text")
    items = load_qa(cfg.qa_path, docs, split=cfg.qa_split)
    for path in cfg.query_expansions:
        if path:
            print(f"  용어집 {path.name}: {len(load_glossary(path))}개 항목")
    print(f"문서 {len(docs)}개, 질문 {len(items)}개 — 검증 통과")
    print(f"  유형: {dict(Counter(it.question_type for it in items))}")
    print(f"  split: {dict(Counter(it.split for it in items))}")
    for ch in cfg.chunking:
        chunks = chunk_corpus(docs, ch.size, ch.overlap)
        avg = sum(len(c.text) for c in chunks) / len(chunks)
        print(f"  청크 {ch.label}: {len(chunks)}개, 평균 {avg:.0f}자")
    return 0


def cmd_export_text(config_path: Path) -> int:
    """QA 작성에 사용할, 평가기가 실제 읽는 정규화 텍스트를 내보낸다."""
    cfg = load_config(config_path)
    docs = load_documents(cfg.docs_dir, text_cache_dir=cfg.output_dir / ".cache" / "text")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="annotation-text-", dir=cfg.output_dir)).resolve()
    index = []
    for doc in docs:
        target = out_dir / f"{doc.doc_id}.txt"
        if not target.resolve().is_relative_to(out_dir):
            raise ValueError(f"내보내기 경로가 결과 폴더를 벗어납니다: {doc.doc_id}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.text, encoding="utf-8")
        index.append({"doc_id": doc.doc_id, "file": target.relative_to(out_dir).as_posix(),
                      "text_sha256": hashlib.sha256(doc.text.encode()).hexdigest()})
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"문서 {len(docs)}개 추출 완료: {out_dir}")
    print("QA의 doc_id는 index.json의 원래 문서 ID를 사용하세요. 내보낸 .txt 파일명이 아닙니다.")
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
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] == "completed_with_errors":
        print(f"일부 요청 실패 — 결과와 실패 건수를 확인하세요: {out / 'summary.md'}", file=sys.stderr)
        return 1
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
    for name, help_text in (("validate", "문서·QA셋 검증"), ("run", "평가 실행"),
                            ("export-text", "QA 작성용 정규화 텍스트 내보내기")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, required=True)
        # README처럼 하위 명령 뒤에 둔 -v도 허용하며, 앞에서 설정한 값을 덮어쓰지 않는다.
        p.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        commands = {"validate": cmd_validate, "run": cmd_run, "export-text": cmd_export_text}
        return commands[args.command](args.config)
    except (ConfigError, FileNotFoundError, ValueError, OllamaError, ImportError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
