"""Ollama LLM provider (OpenAI-compatible-ish HTTP interface to local Ollama)."""
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Iterator

from mullet.core.env import ollama_url
from mullet.core.providers.base import LLMResponse, Provider, StreamChunk


@dataclass
class OllamaConfig:
    """Configuration for an Ollama local LLM endpoint."""
    base_url: str = field(default_factory=ollama_url)
    model: str = "llama2"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 4096
    request_timeout: int = 300


class OllamaProvider(Provider):
    """Ollama local-LLM provider using the /api/chat REST endpoint."""

    name = "ollama"

    def __init__(self, config: OllamaConfig):
        self.config = config

    def create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = False,
        reasoning_config: dict | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to the Ollama API."""
        url = f"{self.config.base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": self.config.top_p,
                "num_predict": max_tokens,
            },
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
                result = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Ollama unreachable at {self.config.base_url}: {exc}") from exc

        content = result.get("message", {}).get("content", "")
        finish = "stop" if result.get("done") else "length"
        return LLMResponse(content=content, finish_reason=finish, provider="ollama", model=self.config.model)

    def create_stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Iterator[StreamChunk]:
        """Stream a chat completion from the Ollama API."""
        url = f"{self.config.base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": self.config.top_p,
                "num_predict": max_tokens,
            },
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:
                for raw_line in resp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode())
                    except json.JSONDecodeError:
                        continue
                    text = chunk.get("message", {}).get("content", "")
                    done = chunk.get("done", False)
                    yield StreamChunk(content=text, finish_reason="stop" if done else None)
                    if done:
                        return
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Ollama unreachable at {self.config.base_url}: {exc}") from exc

    @property
    def supports_tool_calling(self) -> bool:
        """Return whether this provider supports tool calling."""
        return False
