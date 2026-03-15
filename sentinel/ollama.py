# Copyright 2026 Jason Gagne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ollama API client using stdlib only (no external dependencies).

Wraps the Ollama REST API for chat completion and model info.
"""

import json
import time
import urllib.request
import urllib.error
from typing import Optional


class OllamaClient:
    """Minimal Ollama API client."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def _request(self, path: str, data: Optional[dict] = None, timeout: int = 300) -> dict:
        url = f"{self.base_url}{path}"
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
        else:
            req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        num_predict: int = 256,
    ) -> dict:
        """Send a chat completion request. Returns the full API response."""
        start_ms = time.monotonic_ns() // 1_000_000
        resp = self._request("/api/chat", {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        })
        elapsed_ms = (time.monotonic_ns() // 1_000_000) - start_ms

        return {
            "content": resp.get("message", {}).get("content", ""),
            "model": resp.get("model", model),
            "inference_ms": elapsed_ms,
            "prompt_tokens": resp.get("prompt_eval_count", 0),
            "completion_tokens": resp.get("eval_count", 0),
        }

    def get_model_digest(self, model: str) -> str:
        """Get the SHA-256 digest for a model."""
        resp = self._request("/api/show", {"model": model})
        # The digest is in the modelinfo or we fall back to model details
        if "details" in resp and "digest" in resp["details"]:
            return resp["details"]["digest"]
        # Try to get it from the model list
        tags = self._request("/api/tags")
        for m in tags.get("models", []):
            if m["name"] == model or m["model"] == model:
                return m["digest"]
        return "unknown"

    def list_models(self) -> list[dict]:
        """List available models."""
        resp = self._request("/api/tags")
        return resp.get("models", [])

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            self._request("/api/tags")
            return True
        except (urllib.error.URLError, ConnectionError, OSError):
            return False
