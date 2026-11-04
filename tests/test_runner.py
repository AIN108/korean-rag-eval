import json

import numpy as np
import pytest
import yaml
from conftest import FakeChat, FakeEmbedder

from rag_eval.config import ConfigError, load_config
from rag_eval.corpus import Chunk
from rag_eval.retrieval import VectorIndex
from rag_eval.runner import run_experiment


def _write_config(tmp_path, sample_docs_dir, sample_qa_path, **overrides):
    data = {
        "name": "t",
        "docs_dir": str(sample_docs_dir),
        "qa_path": str(sample_qa_path),
        "output_dir": str(tmp_path / "results"),
        "chunking": [{"size": 150, "overlap": 30}, {"size": 400, "overlap": 50}],
        "embed_models": ["fake"],
        "rerankers": [None],
        "llm_models": ["fake-llm"],
        "judge_model": "fake-judge",
        "top_k": 2,
        **overrides,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_vector_index_orders_by_similarity():
    chunks = [Chunk(str(i), "d", 0, 1, "x") for i in range(3)]
    index = VectorIndex(chunks, np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    assert [c.chunk_id for c in index.search(np.array([1.0, 0.1]), 2)] == ["0", "2"]


def test_end_to_end_with_fakes(tmp_path, sample_docs_dir, sample_qa_path):
    cfg_path = _write_config(tmp_path, sample_docs_dir, sample_qa_path)
    cfg = load_config(cfg_path)
    embedder, chat = FakeEmbedder(), FakeChat()
    out = run_experiment(cfg, lambda m: embedder, lambda m: chat, config_path=cfg_path)

    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text("utf-8").splitlines()]
    assert len(rows) == 2 * 6  # 청크 설정 2개 × 질문 6개
    assert {r["chunking"] for r in rows} == {"150/30", "400/50"}
    assert all(len(r["retrieved"]) == 2 for r in rows)
    assert all(r["judge_correct"] is True for r in rows)
    assert (out / "summary.md").is_file() and (out / "config.yaml").is_file()
    # 단순 해싱 임베딩으로도 근거 청크 대부분은 찾아야 정상
    assert sum(r["evidence_rank"] is not None for r in rows) >= 8

    # 두 번째 실행은 임베딩 캐시를 써서 embed 호출이 늘지 않는다
    calls = embedder.calls
    run_experiment(cfg, lambda m: embedder, lambda m: chat)
    assert embedder.calls == calls


def test_retrieval_only_mode(tmp_path, sample_docs_dir, sample_qa_path):
    cfg = load_config(
        _write_config(tmp_path, sample_docs_dir, sample_qa_path, llm_models=[], judge_model=None)
    )
    chat = FakeChat()
    out = run_experiment(cfg, lambda m: FakeEmbedder(), lambda m: chat)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text("utf-8").splitlines()]
    assert chat.calls == []
    assert all(r["answer"] is None and r["llm_model"] == "none" for r in rows)


@pytest.mark.parametrize(
    "override",
    [
        {"chunking": [{"size": 100, "overlap": 100}]},
        {"embed_models": []},
        {"top_k": 0},
        {"rerankers": ["x"], "rerank_candidates": 1},
    ],
)
def test_invalid_config(tmp_path, sample_docs_dir, sample_qa_path, override):
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, sample_docs_dir, sample_qa_path, **override))
