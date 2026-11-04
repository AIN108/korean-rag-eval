import json

import pytest
import yaml
from conftest import FakeChat, FakeEmbedder

from rag_eval.config import ConfigError, EmbedSpec, load_config
from rag_eval.corpus import load_documents
from rag_eval.expansion import expand_query, load_glossary
from rag_eval.runner import run_experiment


def _minimal_pdf(text: str) -> bytes:
    """텍스트 한 줄이 든 1쪽짜리 PDF (테스트용, 외부 라이브러리 없이 작성)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_pdf_loading_uses_text_cache(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.pdf").write_bytes(_minimal_pdf("The army has 120000 personnel."))
    cache = tmp_path / "cache"
    (doc,) = load_documents(docs_dir, text_cache_dir=cache)
    assert "120000 personnel" in doc.text
    assert len(list(cache.glob("*.txt"))) == 1
    # 캐시가 있으면 PDF를 다시 파싱하지 않는다
    next(cache.glob("*.txt")).write_text("cached text", encoding="utf-8")
    assert load_documents(docs_dir, text_cache_dir=cache)[0].text == "cached text"


def test_glossary_and_expansion(tmp_path):
    path = tmp_path / "g.yaml"
    path.write_text("러시아: Russia\n병력: [manpower, troops]\n", encoding="utf-8")
    glossary = load_glossary(path)
    assert expand_query("러시아 병력은?", glossary) == "러시아 병력은? Russia manpower troops"
    assert expand_query("무관한 질문", glossary) == "무관한 질문"
    assert expand_query("러시아", None) == "러시아"


def _config(tmp_path, sample_docs_dir, sample_qa_path, **overrides):
    data = {
        "name": "ext",
        "docs_dir": str(sample_docs_dir),
        "qa_path": str(sample_qa_path),
        "output_dir": str(tmp_path / "results"),
        "chunking": [{"size": 200, "overlap": 20}],
        "embed_models": ["plain"],
        "llm_models": [],
        "top_k": 2,
        **overrides,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_embed_spec_parsing(tmp_path, sample_docs_dir, sample_qa_path):
    cfg = load_config(
        _config(
            tmp_path, sample_docs_dir, sample_qa_path,
            embed_models=["bge-m3", {"name": "e5+p", "model": "e5", "backend": "hf",
                                     "query_prefix": "query: ", "passage_prefix": "passage: "}],
        )
    )
    assert cfg.embed_models[0] == EmbedSpec(name="bge-m3", model="bge-m3")
    assert cfg.embed_models[1].cache_id == "hf:e5"
    assert cfg.embed_models[1].query_prefix == "query: "


@pytest.mark.parametrize(
    "override",
    [
        {"embed_models": [{"model": "x", "backend": "openai"}]},
        {"embed_models": ["a", {"name": "a", "model": "b"}]},
        {"embed_models": [{"name": "no-model"}]},
        {"query_expansions": ["missing.yaml"]},
    ],
)
def test_invalid_extension_config(tmp_path, sample_docs_dir, sample_qa_path, override):
    with pytest.raises(ConfigError):
        load_config(_config(tmp_path, sample_docs_dir, sample_qa_path, **override))


class RecordingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[str] = []

    def embed(self, texts):
        self.inputs.extend(texts)
        return super().embed(texts)


def test_prefixes_and_expansion_grid(tmp_path, sample_docs_dir, sample_qa_path):
    glossary = tmp_path / "g.yaml"
    glossary.write_text("GPU: graphics accelerator\n", encoding="utf-8")
    cfg = load_config(
        _config(
            tmp_path, sample_docs_dir, sample_qa_path,
            embed_models=[
                {"name": "raw", "model": "m"},
                {"name": "prefixed", "model": "m", "query_prefix": "query: ", "passage_prefix": "passage: "},
            ],
            query_expansions=[None, str(glossary)],
            llm_models=["llm"],
        )
    )
    embedder, created = RecordingEmbedder(), []

    def factory(spec):
        created.append(spec.name)
        return embedder

    chat = FakeChat()
    out = run_experiment(cfg, factory, lambda m: chat)
    rows = [json.loads(x) for x in (out / "rows.jsonl").read_text("utf-8").splitlines()]

    assert created == ["raw"]  # 같은 모델은 한 번만 로드
    assert len(rows) == 2 * 2 * 6  # 임베딩 조건 2 × 확장 2 × 질문 6
    assert {r["query_expansion"] for r in rows} == {"none", "g"}
    assert any(t.startswith("passage: ") for t in embedder.inputs)
    assert "query: GPU 서버는 언제까지 예약해야 하나요? graphics accelerator" in embedder.inputs
    # LLM 답변 프롬프트에는 확장되지 않은 원래 질문이 들어간다
    assert not any("graphics accelerator" in r.get("answer", "") for r in rows)
    summary = (out / "summary.md").read_text("utf-8")
    assert "질의 확장" in summary and "prefixed" in summary
