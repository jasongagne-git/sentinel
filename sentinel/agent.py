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

"""SENTINEL Agent — wraps an LLM via Ollama with persona and conversation history."""

from dataclasses import dataclass, field
from typing import Optional

from .ollama import OllamaClient


@dataclass
class AgentConfig:
    """Configuration for a SENTINEL agent."""
    name: str
    system_prompt: str
    model: str = "llama3.2:3b"
    temperature: float = 0.7
    max_history: int = 50
    response_limit: int = 256
    is_control: bool = False
    traits_json: Optional[str] = None
    trait_fingerprint: Optional[str] = None


def serialize_agent_config(config: AgentConfig) -> dict:
    """Serialize AgentConfig for transmission to workers over HTTP."""
    return {
        "name": config.name,
        "system_prompt": config.system_prompt,
        "model": config.model,
        "temperature": config.temperature,
        "max_history": config.max_history,
        "response_limit": config.response_limit,
    }


class Agent:
    """A SENTINEL agent that generates responses via Ollama.

    Each agent has a persona (system prompt), a model, and a sliding
    window of conversation history. The agent observes messages from
    the interaction layer, builds a prompt, and generates a response.
    """

    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        client: OllamaClient,
        model_digest: str,
    ):
        self.agent_id = agent_id
        self.config = config
        self.client = client
        self.model_digest = model_digest

    def build_prompt(self, visible_messages: list[dict]) -> list[dict]:
        """Build the chat messages array from system prompt + visible history.

        Args:
            visible_messages: Recent messages from the interaction layer,
                each with 'agent_name', 'content', and 'agent_id' fields.

        Returns:
            List of chat messages in Ollama format.
        """
        messages = [{"role": "system", "content": self.config.system_prompt}]

        # Sliding window of recent messages
        history = visible_messages[-self.config.max_history:]

        for msg in history:
            if msg["agent_id"] == self.agent_id:
                messages.append({"role": "assistant", "content": msg["content"]})
            else:
                messages.append({
                    "role": "user",
                    "content": f"[{msg['agent_name']}]: {msg['content']}",
                })

        # If no messages yet (first turn), add a nudge
        if not history:
            messages.append({
                "role": "user",
                "content": "The conversation is starting. Please introduce yourself and share your perspective.",
            })

        return messages

    async def generate(self, visible_messages: list[dict]) -> dict:
        """Generate a response given visible messages.

        Returns dict with: content, full_prompt, inference_ms, prompt_tokens, completion_tokens
        """
        import asyncio
        import json

        prompt_messages = self.build_prompt(visible_messages)

        # Run the synchronous Ollama call in a thread to not block the event loop
        result = await asyncio.to_thread(
            self.client.chat,
            model=self.config.model,
            messages=prompt_messages,
            temperature=self.config.temperature,
            num_predict=self.config.response_limit,
        )

        return {
            "content": result["content"],
            "full_prompt": json.dumps(prompt_messages),
            "inference_ms": result["inference_ms"],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "model_digest": self.model_digest,
        }
