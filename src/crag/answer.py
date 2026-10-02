"""Answers from retrieved passages, written by a language model on this machine only."""

import json
import urllib.parse
import urllib.request
from typing import Protocol

from crag.search import Hit

LOCAL_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "::1", "ollama"}
)  # "ollama": the Compose service
GENERATE_TIMEOUT_SECONDS = 120  # a 7B model on a laptop CPU can take a minute for a short answer


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str: ...


def require_local_url(url: str) -> str:
    """Private passages go into the prompt, so the model must not run on another machine."""
    host = urllib.parse.urlparse(url).hostname
    if host not in LOCAL_HOSTS:
        raise ValueError(f"the language model must be local ({sorted(LOCAL_HOSTS)}), got {url!r}")
    return url


class OllamaClient:
    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = require_local_url(base_url).rstrip("/")
        self._model = model

    def generate(self, prompt: str) -> str:
        request = urllib.request.Request(  # noqa: S310 - URL is checked to be local
            self._base_url + "/api/generate",
            data=json.dumps({"model": self._model, "prompt": prompt, "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=GENERATE_TIMEOUT_SECONDS) as response:  # noqa: S310
            body = json.loads(response.read())
        answer = body.get("response") if isinstance(body, dict) else None
        if not isinstance(answer, str):
            raise ValueError("language model returned no answer text")
        return answer.strip()


def build_prompt(question: str, hits: list[Hit]) -> str:
    passages = "\n\n".join(
        f"[{number}] {hit.title}\n{hit.text}" for number, hit in enumerate(hits, start=1)
    )
    return (
        "Answer the question using only the passages below. Cite each claim with its "
        "passage number in square brackets, like [1]. If the passages do not contain the "
        "answer, say so.\n\n"
        f"{passages}\n\nQuestion: {question}\n"
    )
