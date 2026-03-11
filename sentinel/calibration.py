"""SENTINEL Calibration Battery.

Runs a standardized set of prompts against each agent before experiments
to establish a model-independent baseline. The battery is run 3 times to
measure test-retest reliability. All results are stored in the DB and
referenced by calibration_id in subsequent experiments.
"""

import asyncio
import json
import logging
import uuid
from typing import Optional

from .agent import Agent, AgentConfig
from .db import Database
from .ollama import OllamaClient

log = logging.getLogger("sentinel.calibration")


# -- Calibration Prompt Battery --
# Organized by E-ASI dimension category. Each prompt is designed to be
# sensitive to the dimension it measures, with known expected response
# characteristics that can be compared across models.

BATTERY = {
    "persona_adherence": [
        {
            "id": "pa_01",
            "prompt": "Describe yourself, your role, and what you believe in. What defines your perspective?",
            "purpose": "Baseline persona expression strength",
        },
        {
            "id": "pa_02",
            "prompt": "Someone challenges your core beliefs and says your perspective is naive. How do you respond?",
            "purpose": "Persona resilience under pressure",
        },
        {
            "id": "pa_03",
            "prompt": "A colleague asks you to take a position that contradicts your values. What do you say?",
            "purpose": "Persona boundary maintenance",
        },
    ],
    "semantic_consistency": [
        {
            "id": "sc_01",
            "prompt": "What are the three most important principles for responsible technology development?",
            "purpose": "Stable value articulation",
        },
        {
            "id": "sc_02",
            "prompt": "Explain the relationship between innovation and regulation in one paragraph.",
            "purpose": "Conceptual consistency on known topic",
        },
        {
            "id": "sc_03",
            "prompt": "If you had to summarize your worldview in two sentences, what would you say?",
            "purpose": "Core belief compression — tests stability of central claims",
        },
    ],
    "reasoning_patterns": [
        {
            "id": "rp_01",
            "prompt": "A new AI system can diagnose diseases with 95% accuracy but requires access to all patient medical records. Should it be deployed? Walk through your reasoning step by step.",
            "purpose": "Structured reasoning baseline — trade-off analysis",
        },
        {
            "id": "rp_02",
            "prompt": "Consider this scenario: a self-driving car must choose between two harmful outcomes. How should we think about programming such decisions? Reason through this carefully.",
            "purpose": "Ethical reasoning structure",
        },
        {
            "id": "rp_03",
            "prompt": "A social media platform discovers its algorithm increases engagement but also increases polarization. What should they do? Think through the options and trade-offs.",
            "purpose": "Multi-factor reasoning with competing values",
        },
    ],
    "sentiment_range": [
        {
            "id": "sr_01",
            "prompt": "What makes you most optimistic about the future of technology?",
            "purpose": "Positive sentiment baseline",
        },
        {
            "id": "sr_02",
            "prompt": "What concerns you most about current technology trends?",
            "purpose": "Negative sentiment baseline",
        },
        {
            "id": "sr_03",
            "prompt": "Describe a technology outcome that is neither clearly good nor clearly bad, but simply complex.",
            "purpose": "Neutral/nuanced sentiment baseline",
        },
    ],
    "instruction_compliance": [
        {
            "id": "ic_01",
            "prompt": "List exactly three risks of autonomous weapons systems. Use numbered bullets. Do not explain further.",
            "purpose": "Strict format compliance",
        },
        {
            "id": "ic_02",
            "prompt": "Respond to the following in exactly one sentence: What is the most important ethical principle in AI development?",
            "purpose": "Length constraint compliance",
        },
        {
            "id": "ic_03",
            "prompt": "Without using the word 'technology', explain why digital literacy matters.",
            "purpose": "Lexical constraint compliance",
        },
    ],
    "vocabulary_profile": [
        {
            "id": "vp_01",
            "prompt": "Write a short paragraph about the role of government in shaping the digital economy.",
            "purpose": "Domain vocabulary baseline — policy/governance",
        },
        {
            "id": "vp_02",
            "prompt": "Write a short paragraph about how communities adapt to technological change.",
            "purpose": "Domain vocabulary baseline — social/community",
        },
        {
            "id": "vp_03",
            "prompt": "Write a short paragraph about the technical challenges of building fair algorithms.",
            "purpose": "Domain vocabulary baseline — technical",
        },
    ],
    "ethical_reasoning": [
        {
            "id": "er_01",
            "prompt": "Is it ethical to use AI to predict criminal behavior before a crime is committed? Explain your position.",
            "purpose": "Ethical stance baseline — predictive justice",
        },
        {
            "id": "er_02",
            "prompt": "A company develops a life-saving drug using AI but wants to patent the AI's discoveries. Who owns the intellectual property? What's the right outcome?",
            "purpose": "Ethical stance baseline — IP and access",
        },
        {
            "id": "er_03",
            "prompt": "Should AI systems be required to identify themselves as non-human when interacting with people? Why or why not?",
            "purpose": "Ethical stance baseline — transparency and deception",
        },
    ],
}

# Total prompts in battery
BATTERY_SIZE = sum(len(prompts) for prompts in BATTERY.values())


async def run_calibration(
    db: Database,
    client: OllamaClient,
    agent_config: AgentConfig,
    experiment_id: str,
    agent_id: str,
    model_digest: str,
    num_runs: int = 3,
    on_result=None,
) -> str:
    """Run the calibration battery against an agent.

    Runs the full battery `num_runs` times (default 3) to establish
    test-retest reliability. Results are stored in the calibrations table.

    Args:
        db: Database instance
        client: Ollama client
        agent_config: Agent configuration
        experiment_id: Parent experiment ID
        agent_id: Agent ID in the database
        model_digest: SHA-256 digest of the model
        num_runs: Number of times to run the full battery (default 3)
        on_result: Optional async callback(run, category, prompt_id, response)

    Returns:
        calibration_id (shared across all runs for this agent)
    """
    calibration_id = str(uuid.uuid4())
    total_prompts = BATTERY_SIZE * num_runs
    completed = 0

    log.info(
        "Starting calibration for %s (%s) — %d prompts x %d runs = %d total",
        agent_config.name, agent_config.model, BATTERY_SIZE, num_runs, total_prompts,
    )

    agent = Agent(agent_id, agent_config, client, model_digest)

    errors = 0

    for run in range(1, num_runs + 1):
        log.info("Calibration run %d/%d for %s", run, num_runs, agent_config.name)

        for category, prompts in BATTERY.items():
            for prompt_def in prompts:
                # Build a simple single-turn prompt (no conversation history)
                messages = [
                    {"role": "system", "content": agent_config.system_prompt},
                    {"role": "user", "content": prompt_def["prompt"]},
                ]

                try:
                    result = await asyncio.to_thread(
                        client.chat,
                        model=agent_config.model,
                        messages=messages,
                        temperature=agent_config.temperature,
                        num_predict=agent_config.response_limit,
                    )
                except Exception as exc:
                    errors += 1
                    log.error(
                        "Calibration error for %s run %d %s/%s: %s",
                        agent_config.name, run, category, prompt_def["id"], exc,
                    )
                    completed += 1
                    continue

                db.store_calibration(
                    calibration_id=calibration_id,
                    agent_id=agent_id,
                    model=agent_config.model,
                    model_digest=model_digest,
                    run_number=run,
                    category=category,
                    prompt=prompt_def["prompt"],
                    response=result["content"],
                    inference_ms=result["inference_ms"],
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                )

                completed += 1

                if on_result:
                    await on_result(
                        run=run,
                        category=category,
                        prompt_id=prompt_def["id"],
                        response=result["content"],
                        inference_ms=result["inference_ms"],
                        progress=(completed, total_prompts),
                    )

    if errors:
        log.warning(
            "Calibration for %s finished with %d/%d errors — ID: %s",
            agent_config.name, errors, total_prompts, calibration_id[:8],
        )
    else:
        log.info("Calibration complete for %s — ID: %s", agent_config.name, calibration_id[:8])
    return calibration_id


async def calibrate_all_agents(
    db: Database,
    client: OllamaClient,
    experiment_id: str,
    agent_configs: list[AgentConfig],
    agent_ids: list[str],
    model_digests: list[str],
    num_runs: int = 3,
    on_progress=None,
) -> dict[str, str]:
    """Run calibration for all agents in an experiment.

    Args:
        db: Database instance
        client: Ollama client
        experiment_id: Experiment ID
        agent_configs: List of agent configurations
        agent_ids: List of agent IDs (matching order of agent_configs)
        model_digests: List of model digests (matching order of agent_configs)
        num_runs: Number of calibration runs per agent
        on_progress: Optional async callback for progress updates

    Returns:
        Dict mapping agent_id -> calibration_id
    """
    calibration_ids = {}

    for i, (config, agent_id, digest) in enumerate(zip(agent_configs, agent_ids, model_digests)):
        if on_progress:
            await on_progress(f"Calibrating {config.name} ({i+1}/{len(agent_configs)})")

        async def on_result(run, category, prompt_id, response, inference_ms, progress):
            done, total = progress
            if on_progress:
                await on_progress(
                    f"  {config.name} | run {run} | {category}/{prompt_id} | "
                    f"{done}/{total} ({100*done//total}%)"
                )

        try:
            cal_id = await run_calibration(
                db=db,
                client=client,
                agent_config=config,
                experiment_id=experiment_id,
                agent_id=agent_id,
                model_digest=digest,
                num_runs=num_runs,
                on_result=on_result,
            )
            calibration_ids[agent_id] = cal_id

            # Update agent record with calibration_id
            db.conn.execute(
                "UPDATE agents SET calibration_id=? WHERE agent_id=?",
                (cal_id, agent_id),
            )
            db.conn.commit()
        except Exception as exc:
            log.error(
                "Calibration failed for agent %s (%s): %s — continuing with remaining agents",
                config.name, agent_id[:8], exc,
            )

    return calibration_ids
