import json
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import ClassVar

import pytest

from crag.answer import OllamaClient, build_prompt, require_local_url
from crag.search import Hit


def hit(number: int, text: str) -> Hit:
    return Hit(
        chunk_id=number,
        source_key=f"TRACK-{number}",
        source_type="issue",
        issue_key=f"TRACK-{number}",
        title=f"TRACK-{number} Acme Corp",
        url=f"https://tracker.example.com/issue/TRACK-{number}",
        document_date=date(2026, 3, 1),
        text=text,
        score=0.5,
    )


def test_prompt_numbers_passages_and_restricts_the_answer_to_them() -> None:
    prompt = build_prompt("Which roles failed?", [hit(24, "No Quuxdb skills."), hit(9, "Too far.")])
    assert "[1] TRACK-24 Acme Corp\nNo Quuxdb skills." in prompt
    assert "[2] TRACK-9 Acme Corp\nToo far." in prompt
    assert "only the passages" in prompt
    assert prompt.rstrip().endswith("Question: Which roles failed?")


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://[::1]:11434",
    ],
)
def test_local_llm_urls_are_accepted(url: str) -> None:
    assert require_local_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "https://api.example.com",
        "http://10.0.0.5:11434",
        "http://localhost.example.com",
        "http://ollama:11434",
        "",
    ],
)
def test_remote_llm_urls_are_refused(url: str) -> None:
    with pytest.raises(ValueError, match="local"):
        require_local_url(url)


class FakeOllama(BaseHTTPRequestHandler):
    received: ClassVar[list[dict[str, object]]] = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.received.append(body)
        payload = json.dumps({"response": "TRACK-24 lacked Quuxdb [1]."}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        return None


def test_ollama_client_sends_the_prompt_and_returns_the_answer() -> None:
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = OllamaClient(f"http://127.0.0.1:{server.server_port}", "test-model")
        assert client.generate("prompt text") == "TRACK-24 lacked Quuxdb [1]."
        assert FakeOllama.received[-1] == {
            "model": "test-model",
            "prompt": "prompt text",
            "stream": False,
        }
    finally:
        server.shutdown()


def test_ollama_client_refuses_a_remote_url() -> None:
    with pytest.raises(ValueError, match="local"):
        OllamaClient("https://api.example.com", "test-model")


def test_ollama_client_ignores_proxy_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = OllamaClient(f"http://127.0.0.1:{server.server_port}", "test-model")
        assert client.generate("prompt text") == "TRACK-24 lacked Quuxdb [1]."
    finally:
        server.shutdown()
