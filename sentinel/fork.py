"""SENTINEL Experiment Forking — checkpoint and branch experiments.

Forking creates a new experiment branch from a checkpoint in a running
or completed experiment. The original continues uninterrupted. The fork
starts with agents reset to fresh context but identical configurations.

This answers: is drift path-dependent (contingent on conversation history)
or deterministic (inevitable given the conditions)?

Fork lineage is tracked via:
  - experiments.forked_from_experiment_id → parent experiment
  - experiments.fork_at_turn → turn number at fork point
  - agents.forked_from_agent_id → parent agent (for drift comparison)
"""

import json
import logging
from typing import Optional

from .agent import Agent, AgentConfig
from .db import Database
from .ollama import OllamaClient
from .runtime import ExperimentRuntime

log = logging.getLogger("sentinel.fork")


def fork_experiment(
    db: Database,
    client: OllamaClient,
    source_experiment_id: str,
    fork_at_turn: Optional[int] = None,
    name_suffix: str = "fork",
    trait_overrides: Optional[dict[str, dict]] = None,
) -> ExperimentRuntime:
    """Fork an experiment at a given turn.

    Creates a new experiment with:
      - Same topology, timing, and agent configurations as the source
      - Agents linked to their source counterparts via forked_from_agent_id
      - Fresh conversation context (no message history carried over)
      - Lineage recorded in the DB for auditing

    Args:
        db: Database instance
        client: Ollama client
        source_experiment_id: Experiment to fork from
        fork_at_turn: Turn number to fork at. If None, forks from the
            latest turn (current state).
        name_suffix: Appended to experiment name (e.g., "fork", "fork-contrarian")
        trait_overrides: Optional dict of {agent_name: {trait_dim: new_value}}
            for trait mutation experiments. Changes one or more trait dimensions
            for specific agents in the fork.

    Returns:
        ExperimentRuntime ready to run (call .run() to start the fork)
    """
    # Load source experiment
    source_exp = db.get_experiment(source_experiment_id)
    if not source_exp:
        raise ValueError(f"Source experiment not found: {source_experiment_id}")

    source_config = json.loads(source_exp["config_json"])
    source_agents = db.get_agents(source_experiment_id)

    if not source_agents:
        raise ValueError(f"No agents in source experiment: {source_experiment_id}")

    # Determine fork point
    max_turn = db.get_latest_turn(source_experiment_id)
    if max_turn == 0:
        raise ValueError(
            f"Source experiment {source_experiment_id[:8]} has no messages. "
            f"Cannot fork from an experiment with no recorded turns."
        )
    if fork_at_turn is None:
        fork_at_turn = max_turn
    elif fork_at_turn > max_turn:
        raise ValueError(
            f"Fork turn {fork_at_turn} exceeds max turn {max_turn} "
            f"in experiment {source_experiment_id[:8]}"
        )

    # Compute metrics snapshot at fork point for the record
    fork_state = _capture_fork_state(db, source_experiment_id, fork_at_turn, source_agents)

    # Build fork config
    fork_config = dict(source_config)
    fork_config["forked_from"] = source_experiment_id
    fork_config["fork_at_turn"] = fork_at_turn
    fork_config["fork_state"] = fork_state
    if trait_overrides:
        fork_config["trait_overrides"] = trait_overrides

    # Create fork experiment
    fork_name = f"{source_exp['name']} [{name_suffix} @t{fork_at_turn}]"
    fork_desc = (
        f"Fork of {source_experiment_id[:8]} at turn {fork_at_turn}. "
        f"{len(source_agents)} agents with fresh context."
    )
    if trait_overrides:
        overrides_desc = ", ".join(
            f"{name}: {changes}" for name, changes in trait_overrides.items()
        )
        fork_desc += f" Trait mutations: {overrides_desc}"

    fork_experiment_id = db.create_experiment(
        name=fork_name,
        config=fork_config,
        description=fork_desc,
        topology=source_exp["topology"],
        cycle_delay_s=source_exp["cycle_delay_s"],
        max_turns=source_config.get("max_turns"),
    )

    # Set fork lineage
    db.conn.execute(
        "UPDATE experiments SET forked_from_experiment_id=?, fork_at_turn=? WHERE experiment_id=?",
        (source_experiment_id, fork_at_turn, fork_experiment_id),
    )
    db.conn.commit()

    # Create forked agents
    runtime = ExperimentRuntime(db, client, fork_experiment_id)
    digest_cache: dict[str, str] = {}

    for source_agent in source_agents:
        model = source_agent["model"]
        if model not in digest_cache:
            digest_cache[model] = client.get_model_digest(model)
        digest = digest_cache[model]

        # Apply trait overrides if specified
        system_prompt = source_agent["system_prompt"]
        traits_json = source_agent["traits_json"]
        trait_fingerprint = source_agent["trait_fingerprint"]

        if trait_overrides and source_agent["name"] in trait_overrides:
            system_prompt, traits_json, trait_fingerprint = _apply_trait_overrides(
                source_agent, trait_overrides[source_agent["name"]]
            )

        # Reuse calibration from source agent if traits unchanged
        cal_id = None
        if not (trait_overrides and source_agent["name"] in trait_overrides):
            cal_id = source_agent["calibration_id"]

        agent_id = db.create_agent(
            experiment_id=fork_experiment_id,
            name=source_agent["name"],
            system_prompt=system_prompt,
            model=model,
            model_digest=digest,
            temperature=source_agent["temperature"],
            max_history=source_agent["max_history"],
            response_limit=source_agent["response_limit"],
            is_control=bool(source_agent["is_control"]),
            traits_json=traits_json,
            trait_fingerprint=trait_fingerprint,
            forked_from_agent_id=source_agent["agent_id"],
            calibration_id=cal_id,
        )

        config = AgentConfig(
            name=source_agent["name"],
            system_prompt=system_prompt,
            model=model,
            temperature=source_agent["temperature"],
            max_history=source_agent["max_history"],
            response_limit=source_agent["response_limit"],
            is_control=bool(source_agent["is_control"]),
            traits_json=traits_json,
            trait_fingerprint=trait_fingerprint,
        )
        agent = Agent(agent_id, config, client, digest)
        runtime.add_agent(agent)

    log.info(
        "Forked experiment %s at turn %d → %s (%d agents%s)",
        source_experiment_id[:8],
        fork_at_turn,
        fork_experiment_id[:8],
        len(source_agents),
        f", {len(trait_overrides)} trait mutations" if trait_overrides else "",
    )

    return runtime


def _capture_fork_state(
    db: Database,
    experiment_id: str,
    fork_at_turn: int,
    agents: list[dict],
) -> dict:
    """Capture a summary of experiment state at the fork point."""
    state = {
        "fork_at_turn": fork_at_turn,
        "agents": {},
    }

    for agent in agents:
        # Count messages up to fork point
        row = db.conn.execute(
            "SELECT COUNT(*) as n FROM messages "
            "WHERE experiment_id=? AND agent_id=? AND interaction_turn <= ?",
            (experiment_id, agent["agent_id"], fork_at_turn),
        ).fetchone()

        # Get last message content
        last_msg = db.conn.execute(
            "SELECT content, interaction_turn FROM messages "
            "WHERE experiment_id=? AND agent_id=? AND interaction_turn <= ? "
            "ORDER BY interaction_turn DESC LIMIT 1",
            (experiment_id, agent["agent_id"], fork_at_turn),
        ).fetchone()

        state["agents"][agent["name"]] = {
            "agent_id": agent["agent_id"],
            "message_count": row["n"],
            "last_turn": last_msg["interaction_turn"] if last_msg else 0,
            "last_message_preview": last_msg["content"][:100] if last_msg else "",
        }

    return state


def _apply_trait_overrides(
    source_agent: dict,
    overrides: dict[str, str],
) -> tuple[str, str, str]:
    """Apply trait overrides to a source agent, generating new prompt and fingerprint.

    Returns (new_system_prompt, new_traits_json, new_trait_fingerprint).
    """
    from .persona import PersonaTraits, compose_system_prompt

    if not source_agent["traits_json"]:
        raise ValueError(
            f"Cannot apply trait overrides to agent '{source_agent['name']}' — "
            f"no structured traits (uses legacy system_prompt)"
        )

    traits_dict = json.loads(source_agent["traits_json"])
    traits_dict.update(overrides)

    traits = PersonaTraits(**traits_dict)
    errors = traits.validate()
    if errors:
        raise ValueError(f"Invalid trait overrides:\n" + "\n".join(errors))

    system_prompt = compose_system_prompt(source_agent["name"], traits)
    traits_json = json.dumps(traits_dict)
    fingerprint = traits.fingerprint()

    return system_prompt, traits_json, fingerprint


def list_forks(db: Database, experiment_id: str) -> list[dict]:
    """List all forks of an experiment (direct children only)."""
    rows = db.conn.execute(
        "SELECT experiment_id, name, fork_at_turn, status, created_at "
        "FROM experiments WHERE forked_from_experiment_id=? "
        "ORDER BY fork_at_turn, created_at",
        (experiment_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_fork_lineage(db: Database, experiment_id: str) -> list[dict]:
    """Trace the full lineage of an experiment back to the root."""
    lineage = []
    current_id = experiment_id

    while current_id:
        exp = db.get_experiment(current_id)
        if not exp:
            break
        lineage.append({
            "experiment_id": exp["experiment_id"],
            "name": exp["name"],
            "fork_at_turn": exp["fork_at_turn"],
            "status": exp["status"],
        })
        current_id = exp["forked_from_experiment_id"]

    lineage.reverse()
    return lineage
