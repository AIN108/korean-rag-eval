import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from rag_eval import cli
from rag_eval.ollama import OllamaClient, OllamaError


@pytest.mark.parametrize("args", [
    ["-v", "validate", "--config", "configs/default.yaml"],
    ["validate", "--config", "configs/default.yaml", "-v"],
])
def test_verbose_flag_before_and_after_subcommand(monkeypatch, args):
    calls = []
    monkeypatch.setattr(cli, "cmd_validate", lambda path: calls.append(path) or 0)
    assert cli.main(args) == 0 and len(calls) == 1


def test_export_text_does_not_require_a_qa_file(tmp_path, monkeypatch):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("정규화할 원문\r\n본문", encoding="utf-8")
    cfg = SimpleNamespace(docs_dir=docs, output_dir=tmp_path / "results")
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    assert cli.main(["export-text", "--config", "not-yet-annotated.yaml"]) == 0
    (out,) = cfg.output_dir.glob("annotation-text-*")
    index = json.loads((out / "index.json").read_text("utf-8"))
    assert index[0]["doc_id"] == "a.md" and index[0]["file"] == "a.md.txt"
    assert (out / "a.md.txt").read_text("utf-8") == "정규화할 원문\n본문"


def test_cli_returns_failure_code_when_run_has_request_errors(tmp_path, monkeypatch):
    from rag_eval import runner

    (tmp_path / "manifest.json").write_text('{"status":"completed_with_errors"}', "utf-8")
    cfg = SimpleNamespace(ollama_url="http://localhost:11434", temperature=0, seed=42)
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setattr(runner, "run_experiment", lambda *args, **kwargs: tmp_path)
    assert cli.cmd_run(tmp_path / "cfg.yaml") == 1


@pytest.mark.parametrize("response", [[], "not-json"])
def test_invalid_http_response_becomes_domain_error_and_closes(response):
    client = OllamaClient("http://localhost:11434")
    resp = Mock(ok=True, status_code=200)
    if response == "not-json":
        resp.json.side_effect = ValueError("bad json")
    else:
        resp.json.return_value = response
    client._session.post = Mock(return_value=resp)
    try:
        with pytest.raises(OllamaError, match="JSON"):
            client.post("/api/chat", {"model": "fake"})
        resp.close.assert_called_once()
    finally:
        client.close()


def test_timeout_is_reported_without_uncaught_requests_exception():
    client = OllamaClient("http://localhost:11434", timeout=1)
    client._session.post = Mock(side_effect=requests.Timeout())
    try:
        with pytest.raises(OllamaError, match="1초"):
            client.post("/api/chat", {})
    finally:
        client.close()
