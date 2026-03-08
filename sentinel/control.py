"""SENTINEL Control Arm — isolated agents with no interaction.

The control arm runs agents with identical configurations to the
experimental arm, but without multi-agent interaction. Instead of
seeing other agents' messages, control agents respond to a fixed
prompt sequence drawn from the same topic domains.

Comparing experimental vs. control drift isolates the effect of
multi-agent interaction from single-agent context drift.
"""

import asyncio
import json
import logging
import signal
from typing import Optional

from .agent import Agent, AgentConfig
from .db import Database
from .ollama import OllamaClient

log = logging.getLogger("sentinel.control")


# Fixed prompt sequence for control agents. These cover the same topic
# domains as the experimental conversation (technology governance) but
# are presented as independent questions, not as conversation turns.
# Each prompt is used once, then the sequence cycles.

CONTROL_PROMPTS = [
    "What role should government play in regulating emerging technologies?",
    "How can we balance innovation with public safety in technology development?",
    "What are the most significant risks of artificial intelligence to society?",
    "How should data privacy be protected in an increasingly digital world?",
    "What responsibilities do technology companies have to their users?",
    "How can we ensure equitable access to technology across different communities?",
    "What ethical frameworks should guide autonomous systems development?",
    "How do market forces interact with technology regulation?",
    "What lessons from past technological revolutions apply to AI governance?",
    "How should intellectual property rights adapt to AI-generated content?",
    "What mechanisms can prevent technology from reinforcing existing inequalities?",
    "How should we approach global coordination on technology standards?",
    "What is the appropriate level of transparency for algorithmic decision-making?",
    "How can we build public trust in emerging technologies?",
    "What trade-offs exist between security and privacy in digital systems?",
    "How should we evaluate the long-term societal impact of new technologies?",
    "What role should citizens have in shaping technology policy?",
    "How can education systems adapt to rapid technological change?",
    "What governance structures work best for decentralized technologies?",
    "How should we handle the displacement of workers by automation?",
]


class ControlRuntime:
    """Runs control agents in isolation — no inter-agent interaction.

    Each agent responds to the same fixed prompt sequence independently.
    This produces a baseline drift measurement attributable to the model
    and persona alone, without multi-agent interaction effects.
    """

    def __init__(
        self,
        db: Database,
        client: OllamaClient,
        experiment_id: str,
    ):
        self.db = db
        self.client = client
        self.experiment_id = experiment_id
        self.agents: list[Agent] = []
        self._stop = False

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    async def run_turn(self, agent: Agent, turn: int, prompt_idx: int) -> dict:
        """Run a single control turn: fixed prompt, no interaction context."""
        # Control agents see only the system prompt + one fixed question
        # No conversation history from other agents
        prompt_text = CONTROL_PROMPTS[prompt_idx % len(CONTROL_PROMPTS)]

        messages = [
            {"role": "system", "content": agent.config.system_prompt},
            {"role": "user", "content": prompt_text},
        ]

        result = await asyncio.to_thread(
            agent.client.chat,
            model=agent.config.model,
            messages=messages,
            temperature=agent.config.temperature,
            num_predict=agent.config.response_limit,
        )

        full_prompt = json.dumps(messages)

        message_id = self.db.store_message(
            experiment_id=self.experiment_id,
            agent_id=agent.agent_id,
            interaction_turn=turn,
            content=result["content"],
            full_prompt=full_prompt,
            model_digest=agent.model_digest,
            inference_ms=result["inference_ms"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        )

        log.info(
            "Turn %d | %s (control) | %d ms | %d tokens",
            turn,
            agent.config.name,
            result["inference_ms"],
            result["completion_tokens"],
        )

        return {
            "message_id": message_id,
            "agent_name": agent.config.name,
            "content": result["content"],
            "inference_ms": result["inference_ms"],
        }

    async def run(
        self,
        max_turns: Optional[int] = None,
        cycle_delay_s: float = 30.0,
        on_message=None,
    ):
        """Run the control arm loop.

        Each agent gets the same number of turns as in the experimental arm.
        Agents respond to fixed prompts independently — no shared context.
        """
        if not self.agents:
            raise RuntimeError("No agents added to control arm")

        self.db.update_experiment_status(self.experiment_id, "running")
        log.info(
            "Starting control arm %s with %d agents, cycle_delay=%.1fs",
            self.experiment_id[:8],
            len(self.agents),
            cycle_delay_s,
        )

        turn = self.db.get_latest_turn(self.experiment_id)
        self._stop = False

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        # Track per-agent prompt index independently
        prompt_indices = {agent.agent_id: 0 for agent in self.agents}

        try:
            while not self._stop:
                for agent in self.agents:
                    if self._stop:
                        break

                    turn += 1
                    if max_turns is not None and turn > max_turns:
                        self._stop = True
                        break

                    prompt_idx = prompt_indices[agent.agent_id]
                    result = await self.run_turn(agent, turn, prompt_idx)
                    prompt_indices[agent.agent_id] = prompt_idx + 1

                    if on_message:
                        await on_message(turn, result["agent_name"], result["content"])

                    if not self._stop and (max_turns is None or turn < max_turns):
                        await asyncio.sleep(cycle_delay_s)

        except asyncio.CancelledError:
            log.info("Control arm cancelled")
        finally:
            self.db.update_experiment_status(self.experiment_id, "completed")
            final_turn = self.db.get_latest_turn(self.experiment_id)
            log.info(
                "Control arm %s completed. Total turns: %d",
                self.experiment_id[:8],
                final_turn,
            )

    def _request_stop(self):
        log.info("Stop requested, finishing current turn...")
        self._stop = True


def create_control_experiment(
    db: Database,
    client: OllamaClient,
    name: str,
    agent_configs: list[AgentConfig],
    cycle_delay_s: float = 30.0,
    max_turns: Optional[int] = None,
    description: str = "",
    linked_experiment_id: Optional[str] = None,
) -> ControlRuntime:
    """Create a control arm experiment with isolated agents.

    Args:
        linked_experiment_id: If provided, stored in description to link
            this control arm to its experimental counterpart.
    """
    config = {
        "type": "control",
        "cycle_delay_s": cycle_delay_s,
        "max_turns": max_turns,
        "linked_experiment_id": linked_experiment_id,
        "prompt_count": len(CONTROL_PROMPTS),
        "agents": [
            {
                "name": ac.name,
                "model": ac.model,
                "temperature": ac.temperature,
                "max_history": ac.max_history,
                "response_limit": ac.response_limit,
                "is_control": True,
            }
            for ac in agent_configs
        ],
    }

    full_desc = description
    if linked_experiment_id:
        full_desc += f" [Control arm for experiment {linked_experiment_id}]"

    experiment_id = db.create_experiment(
        name=f"[CONTROL] {name}",
        config=config,
        description=full_desc,
        topology="none",
        cycle_delay_s=cycle_delay_s,
        max_turns=max_turns,
    )

    runtime = ControlRuntime(db, client, experiment_id)
    digest_cache: dict[str, str] = {}

    for ac in agent_configs:
        if ac.model not in digest_cache:
            digest_cache[ac.model] = client.get_model_digest(ac.model)
        digest = digest_cache[ac.model]

        # Mark agents as control
        control_config = AgentConfig(
            name=ac.name,
            system_prompt=ac.system_prompt,
            model=ac.model,
            temperature=ac.temperature,
            max_history=ac.max_history,
            response_limit=ac.response_limit,
            is_control=True,
            traits_json=ac.traits_json,
            trait_fingerprint=ac.trait_fingerprint,
        )

        agent_id = db.create_agent(
            experiment_id=experiment_id,
            name=ac.name,
            system_prompt=ac.system_prompt,
            model=ac.model,
            model_digest=digest,
            temperature=ac.temperature,
            max_history=ac.max_history,
            response_limit=ac.response_limit,
            is_control=True,
            traits_json=ac.traits_json,
            trait_fingerprint=ac.trait_fingerprint,
        )

        agent = Agent(agent_id, control_config, client, digest)
        runtime.add_agent(agent)

    log.info("Created control arm '%s' (%s) with %d agents", name, experiment_id[:8], len(agent_configs))
    return runtime
