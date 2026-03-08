"""SENTINEL Experiment Runtime — orchestrates multi-agent interactions."""

import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from .agent import Agent, AgentConfig
from .db import Database
from .ollama import OllamaClient

log = logging.getLogger("sentinel.runtime")


class ExperimentRuntime:
    """Runs a multi-agent experiment with configurable topology and timing.

    Phase 1: Single-node, full mesh topology, sequential inference.
    Agents take turns in round-robin order. Each agent sees all public
    messages (full mesh). One agent generates at a time to avoid GPU
    contention on memory-constrained hardware.
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
        self.agent_names: dict[str, str] = {}  # agent_id -> name
        self._stop = False

    def add_agent(self, agent: Agent):
        self.agents.append(agent)
        self.agent_names[agent.agent_id] = agent.config.name

    def _get_visible_messages(self, agent: Agent) -> list[dict]:
        """Get messages visible to an agent. Full mesh = all public messages."""
        messages = self.db.get_messages(
            self.experiment_id,
            limit=agent.config.max_history,
        )
        # Enrich with agent names for prompt building
        for msg in messages:
            msg["agent_name"] = self.agent_names.get(msg["agent_id"], "Unknown")
        return messages

    async def run_turn(self, agent: Agent, turn: int) -> dict:
        """Run a single agent turn: observe, generate, store."""
        visible = self._get_visible_messages(agent)
        result = await agent.generate(visible)

        message_id = self.db.store_message(
            experiment_id=self.experiment_id,
            agent_id=agent.agent_id,
            interaction_turn=turn,
            content=result["content"],
            full_prompt=result["full_prompt"],
            model_digest=result["model_digest"],
            inference_ms=result["inference_ms"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
        )

        log.info(
            "Turn %d | %s | %d ms | %d tokens",
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
        """Run the experiment loop.

        Args:
            max_turns: Stop after this many total turns (None = run until stopped).
            cycle_delay_s: Delay in seconds between agent turns.
            on_message: Optional async callback(turn, agent_name, content) for live output.
        """
        if not self.agents:
            raise RuntimeError("No agents added to experiment")

        self.db.update_experiment_status(self.experiment_id, "running")
        log.info(
            "Starting experiment %s with %d agents, cycle_delay=%.1fs",
            self.experiment_id[:8],
            len(self.agents),
            cycle_delay_s,
        )

        turn = self.db.get_latest_turn(self.experiment_id)
        self._stop = False

        # Handle graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        try:
            while not self._stop:
                for agent in self.agents:
                    if self._stop:
                        break

                    turn += 1
                    if max_turns is not None and turn > max_turns:
                        self._stop = True
                        break

                    result = await self.run_turn(agent, turn)

                    if on_message:
                        await on_message(turn, result["agent_name"], result["content"])

                    # Delay between turns (but not after the last one if stopping)
                    if not self._stop and (max_turns is None or turn < max_turns):
                        await asyncio.sleep(cycle_delay_s)

        except asyncio.CancelledError:
            log.info("Experiment cancelled")
        finally:
            self.db.update_experiment_status(self.experiment_id, "completed")
            final_turn = self.db.get_latest_turn(self.experiment_id)
            log.info(
                "Experiment %s completed. Total turns: %d",
                self.experiment_id[:8],
                final_turn,
            )

    def _request_stop(self):
        log.info("Stop requested, finishing current turn...")
        self._stop = True


def create_experiment(
    db: Database,
    client: OllamaClient,
    name: str,
    agent_configs: list[AgentConfig],
    topology: str = "full_mesh",
    cycle_delay_s: float = 30.0,
    max_turns: Optional[int] = None,
    description: str = "",
) -> ExperimentRuntime:
    """Create an experiment with agents and return a ready-to-run runtime.

    Handles model digest lookup, DB registration, and agent instantiation.
    """
    # Build full config for storage
    config = {
        "topology": topology,
        "cycle_delay_s": cycle_delay_s,
        "max_turns": max_turns,
        "agents": [
            {
                "name": ac.name,
                "model": ac.model,
                "temperature": ac.temperature,
                "max_history": ac.max_history,
                "response_limit": ac.response_limit,
                "is_control": ac.is_control,
            }
            for ac in agent_configs
        ],
    }

    experiment_id = db.create_experiment(
        name=name,
        config=config,
        description=description,
        topology=topology,
        cycle_delay_s=cycle_delay_s,
        max_turns=max_turns,
    )

    runtime = ExperimentRuntime(db, client, experiment_id)

    # Cache model digests to avoid repeated API calls
    digest_cache: dict[str, str] = {}

    for ac in agent_configs:
        if ac.model not in digest_cache:
            digest_cache[ac.model] = client.get_model_digest(ac.model)
        digest = digest_cache[ac.model]

        agent_id = db.create_agent(
            experiment_id=experiment_id,
            name=ac.name,
            system_prompt=ac.system_prompt,
            model=ac.model,
            model_digest=digest,
            temperature=ac.temperature,
            max_history=ac.max_history,
            response_limit=ac.response_limit,
            is_control=ac.is_control,
            traits_json=ac.traits_json,
            trait_fingerprint=ac.trait_fingerprint,
        )

        agent = Agent(agent_id, ac, client, digest)
        runtime.add_agent(agent)

    log.info("Created experiment '%s' (%s) with %d agents", name, experiment_id[:8], len(agent_configs))
    return runtime
