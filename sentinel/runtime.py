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

"""SENTINEL Experiment Runtime — orchestrates multi-agent interactions."""

import asyncio
import json
import logging
import signal
import sys
import urllib.error
from pathlib import Path
from typing import Optional

from .agent import Agent, AgentConfig
from .db import Database
from .ollama import OllamaClient
from .probes import ProbeRunner, TriggerConfig
from .thermal import ThermalGuard

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
        self.probe_runner: Optional[ProbeRunner] = None
        self.thermal = ThermalGuard()
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

        turn, last_agent_id = self.db.get_resume_position(self.experiment_id)

        # Determine where to resume in the agent rotation.
        # If the last cycle was incomplete (crashed mid-cycle), skip agents
        # that already spoke and resume from the next one.
        start_agent_idx = 0
        if last_agent_id:
            for idx, agent in enumerate(self.agents):
                if agent.agent_id == last_agent_id:
                    start_agent_idx = idx + 1  # next agent after the last one that spoke
                    break
            if start_agent_idx >= len(self.agents):
                start_agent_idx = 0  # last cycle was complete, start fresh
            elif start_agent_idx > 0:
                log.info(
                    "Resuming mid-cycle at agent index %d/%d (after %s, turn %d)",
                    start_agent_idx, len(self.agents),
                    next((a.config.name for a in self.agents if a.agent_id == last_agent_id), "?"),
                    turn,
                )

        self._stop = False

        # Ignore SIGHUP so experiments survive SSH disconnects
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGHUP, lambda: None)

        # Handle graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        consecutive_errors = 0
        max_consecutive_errors = 3
        final_status = "completed"

        first_cycle = True
        try:
            while not self._stop:
                if first_cycle and start_agent_idx > 0:
                    agents_this_cycle = self.agents[start_agent_idx:]
                    first_cycle = False
                else:
                    agents_this_cycle = self.agents
                    first_cycle = False

                for agent in agents_this_cycle:
                    if self._stop:
                        break

                    turn += 1
                    if max_turns is not None and turn > max_turns:
                        self._stop = True
                        break

                    # Pre-inference thermal check
                    pre_extra = await self.thermal.check(f"pre {agent.config.name}")
                    if pre_extra > 0:
                        await asyncio.sleep(pre_extra)

                    try:
                        result = await self.run_turn(agent, turn)
                        consecutive_errors = 0
                    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError) as exc:
                        consecutive_errors += 1
                        log.error(
                            "Turn %d | %s | inference failed (%d/%d): %s",
                            turn, agent.config.name,
                            consecutive_errors, max_consecutive_errors, exc,
                        )
                        if consecutive_errors >= max_consecutive_errors:
                            log.error(
                                "Aborting experiment %s after %d consecutive errors",
                                self.experiment_id[:8], max_consecutive_errors,
                            )
                            final_status = "failed"
                            self._stop = True
                            break
                        turn -= 1  # Retry this turn slot
                        await asyncio.sleep(min(cycle_delay_s * 2, 30))
                        continue
                    except Exception as exc:
                        log.error(
                            "Turn %d | %s | unexpected error: %s",
                            turn, agent.config.name, exc,
                        )
                        final_status = "failed"
                        self._stop = True
                        break

                    if on_message:
                        await on_message(turn, result["agent_name"], result["content"])

                    # Feed message to drift monitor for threshold tracking
                    if self.probe_runner:
                        self.probe_runner.record_turn(
                            agent.agent_id, result["content"], turn,
                        )

                    # Run probes if configured
                    if self.probe_runner:
                        try:
                            visible = self._get_visible_messages(agent)
                            probe_results = await self.probe_runner.run_probes(agent, turn, visible)
                            if probe_results and on_message:
                                for pr in probe_results:
                                    if pr.get("drift_score") is not None:
                                        await on_message(
                                            turn,
                                            f"{agent.config.name} [probe:{pr['mode']}]",
                                            f"[{pr['category']}] drift={pr['drift_score']:.3f}",
                                        )
                        except Exception as exc:
                            log.warning("Probe failed at turn %d for %s: %s", turn, agent.config.name, exc)

                    # Post-inference thermal check
                    post_extra = await self.thermal.check(f"post {agent.config.name}")
                    if post_extra > 0:
                        await asyncio.sleep(post_extra)

                    # Delay between turns (but not after the last one if stopping)
                    if not self._stop and (max_turns is None or turn < max_turns):
                        await asyncio.sleep(cycle_delay_s)

        except asyncio.CancelledError:
            log.info("Experiment cancelled")
        finally:
            self.db.update_experiment_status(self.experiment_id, final_status)
            final_turn = self.db.get_latest_turn(self.experiment_id)
            thermal_stats = self.thermal.stats
            log.info(
                "Experiment %s %s. Total turns: %d | Thermal: pauses=%d (%.0fs), max=%.1f°C, checks=%d",
                self.experiment_id[:8],
                final_status,
                final_turn,
                thermal_stats["pause_count"],
                thermal_stats["total_pause_seconds"],
                thermal_stats["max_temp_c"],
                thermal_stats["checks"],
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
    probe_mode: Optional[str] = None,
    probe_interval: int = 20,
    probe_strategy: str = "scheduled",
    trigger_config: Optional[TriggerConfig] = None,
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

        cal_id = db.find_calibration_id(ac.name, ac.model, digest)
        if cal_id:
            log.info("Linked agent '%s' to calibration %s", ac.name, cal_id[:8])

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
            calibration_id=cal_id,
        )

        agent = Agent(agent_id, ac, client, digest)
        runtime.add_agent(agent)

    # Attach probe runner if configured
    if probe_mode:
        runtime.probe_runner = ProbeRunner(
            db=db,
            client=client,
            experiment_id=experiment_id,
            mode=probe_mode,
            interval=probe_interval,
            strategy=probe_strategy,
            trigger_config=trigger_config,
        )
        log.info(
            "Probes enabled: mode=%s, strategy=%s, interval=%d turns",
            probe_mode, probe_strategy, probe_interval,
        )

    log.info("Created experiment '%s' (%s) with %d agents", name, experiment_id[:8], len(agent_configs))
    return runtime
