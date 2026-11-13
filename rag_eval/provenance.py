"""실험 설정과 데이터 지문을 기록한다. 원문과 질문셋 자체는 복사하지 않는다."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rag_eval.config import ExperimentConfig
from rag_eval.corpus import Document
from rag_eval.qa import QAItem


def _versions() -> dict[str, str]:
    packages = {}
    for name in ("numpy", "requests", "pyyaml", "pypdf", "sentence-transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return packages


@contextmanager
def record_run(cfg: ExperimentConfig, docs: list[Document], items: list[QAItem], out_dir: Path):
    """예외/중단도 status에 남긴다. 모델 태그는 가중치 digest를 보장하지 않는다."""
    config = asdict(cfg)
    # 원본 설정 대신 환경변수 등도 적용된 유효 설정을 남긴다. 접속 URL은 저장하지 않는다.
    config.pop("raw")
    config.pop("ollama_url")
    code_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    manifest = {
        "schema_version": 2,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "effective_config": config,
        "corpus": [
            {"doc_id": d.doc_id, "text_sha256": hashlib.sha256(d.text.encode()).hexdigest(),
             "characters": len(d.text)} for d in docs
        ],
        "qa_sha256": hashlib.sha256(cfg.qa_path.read_bytes()).hexdigest(),
        "question_count": len(items),
        "question_types": dict(Counter(item.question_type for item in items)),
        "splits": dict(Counter(item.split for item in items)),
        "code_sha256": code_hashes,
        "runtime": {"python": platform.python_version(), "packages": _versions()},
        "row_count": 0,
        "limitations": ["모델 태그만 기록합니다. 가중치 digest, Ollama 버전, 하드웨어는 별도 기록하세요."],
    }
    path = out_dir / "manifest.json"

    def save():
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", "utf-8")

    save()
    try:
        yield manifest
    except BaseException:
        manifest["status"] = "failed"
        raise
    else:
        if manifest["status"] == "running":
            manifest["status"] = "completed"
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
