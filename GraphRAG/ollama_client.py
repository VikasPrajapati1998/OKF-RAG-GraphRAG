"""Thin Ollama HTTP client for embeddings and chat (streaming and JSON)."""

import json
from collections.abc import Iterator

import numpy as np
import requests

from config import Settings
from exception import EmbeddingError, LLMError
from logger import logger

_EMBED_BATCH = 32


class OllamaClient:
    """Client for the local Ollama server."""

    def __init__(self, settings: Settings) -> None:
        """Create a client.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._session = requests.Session()

    def check_ready(self) -> None:
        """Verify the server is reachable and both models are installed.

        Raises:
            LLMError: If Ollama is unreachable or a model is missing.
        """
        try:
            response = self._session.get(f"{self._settings.ollama_host}/api/tags", timeout=10)
            response.raise_for_status()
            installed = {m["name"] for m in response.json().get("models", [])}
        except (requests.RequestException, ValueError, KeyError) as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self._settings.ollama_host}. Is 'ollama serve' running?"
            ) from exc
        for model in (self._settings.chat_model, self._settings.embed_model):
            if model not in installed and f"{model}:latest" not in installed:
                raise LLMError(f"Model '{model}' is not installed. Run: ollama pull {model}")
        logger.info("Ollama ready; models available")

    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed texts into L2-normalised float32 vectors.

        Args:
            texts: Texts to embed.
            is_query: Use the query prefix instead of the document prefix.

        Returns:
            Array of shape ``(len(texts), dim)``.

        Raises:
            EmbeddingError: If the request fails or the response is malformed.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        prefix = self._settings.embed_query_prefix if is_query else self._settings.embed_doc_prefix
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = [prefix + t for t in texts[start : start + _EMBED_BATCH]]
            try:
                response = self._session.post(
                    f"{self._settings.ollama_host}/api/embed",
                    json={"model": self._settings.embed_model, "input": batch, "keep_alive": "30m"},
                    timeout=self._settings.request_timeout,
                )
                response.raise_for_status()
                embeddings = response.json()["embeddings"]
            except (requests.RequestException, ValueError, KeyError) as exc:
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc
            if len(embeddings) != len(batch):
                raise EmbeddingError("Embedding count does not match input count")
            vectors.extend(embeddings)
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _payload(self, messages: list[dict[str, str]], stream: bool, json_mode: bool) -> dict:
        """Build a chat request payload."""
        payload: dict = {
            "model": self._settings.chat_model,
            "messages": messages,
            "stream": stream,
            "keep_alive": "30m",
            "options": {
                "temperature": self._settings.temperature,
                "num_ctx": self._settings.num_ctx,
                "num_predict": self._settings.max_answer_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
            payload["options"]["temperature"] = 0.0
            payload["options"]["num_predict"] = 1024
        return payload

    def chat_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream a chat completion token by token.

        Args:
            messages: Chat messages in Ollama format.

        Yields:
            Text fragments as they are generated.

        Raises:
            LLMError: If the request fails or the stream is malformed.
        """
        try:
            with self._session.post(
                f"{self._settings.ollama_host}/api/chat",
                json=self._payload(messages, stream=True, json_mode=False),
                timeout=self._settings.request_timeout,
                stream=True,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise LLMError(f"Model error: {chunk['error']}")
                    yield chunk.get("message", {}).get("content", "")
                    if chunk.get("done"):
                        break
        except (requests.RequestException, ValueError) as exc:
            raise LLMError(f"Chat request failed: {exc}") from exc

    def chat_json(self, messages: list[dict[str, str]]) -> dict:
        """Run a non-streaming chat completion constrained to JSON output.

        Args:
            messages: Chat messages in Ollama format.

        Returns:
            The parsed JSON object.

        Raises:
            LLMError: If the request fails or the output is not valid JSON.
        """
        try:
            response = self._session.post(
                f"{self._settings.ollama_host}/api/chat",
                json=self._payload(messages, stream=False, json_mode=True),
                timeout=self._settings.request_timeout,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
            parsed = json.loads(content)
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMError(f"JSON chat request failed: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("Model returned JSON that is not an object")
        return parsed
