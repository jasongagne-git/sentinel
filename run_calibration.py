#!/usr/bin/env python3
"""Run SENTINEL calibration battery for agents defined in an experiment config.

Usage:
    python3 run_calibration.py config/three_agent_mesh.json
    python3 run_calibration.py config/three_agent_mesh.json --runs 1  # quick test
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from sentinel.agent import AgentConfig
from sentinel.calibration import calibrate_all_agents, BATTERY_SIZE
from sentinel.db import Database
from sentinel.ollama import OllamaClient
from sentinel.persona import load_persona_config


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def main():
    parser = argparse.ArgumentParser(description="Run SENTINEL calibration battery")
    parser.add_argument("config", help="Path to experiment config JSON")
    parser.add_argument("--runs", type=int, default=3, help="Number of calibration runs (default 3)")
    parser.add_argument("--db", default="experiments/sentinel.db", help="Database path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    client = OllamaClient()
    if not client.is_available():
        print("Error: Ollama is not running.", file=sys.stderr)
        sys.exit(1)

    db = Database(args.db)

    # Build agent configs
    agent_configs = []
    for agent_def in config["agents"]:
        name, traits, system_prompt = load_persona_config(agent_def)
        agent_configs.append(AgentConfig(
            name=name,
            system_prompt=system_prompt,
            model=agent_def.get("model", config.get("default_model", "llama3:latest")),
            temperature=agent_def.get("temperature", config.get("default_temperature", 0.7)),
            max_history=agent_def.get("max_history", config.get("default_max_history", 50)),
            response_limit=agent_def.get("response_limit", config.get("default_response_limit", 256)),
        ))

    # Create experiment record for calibration
    experiment_id = db.create_experiment(
        name=f"Calibration: {config['name']}",
        config=config,
        description=f"Calibration battery for {len(agent_configs)} agents, {args.runs} runs each",
    )

    # Register agents and get digests
    agent_ids = []
    model_digests = []
    digest_cache = {}
    for ac in agent_configs:
        if ac.model not in digest_cache:
            digest_cache[ac.model] = client.get_model_digest(ac.model)
        digest = digest_cache[ac.model]
        model_digests.append(digest)

        agent_id = db.create_agent(
            experiment_id=experiment_id,
            name=ac.name,
            system_prompt=ac.system_prompt,
            model=ac.model,
            model_digest=digest,
            temperature=ac.temperature,
            max_history=ac.max_history,
            response_limit=ac.response_limit,
        )
        agent_ids.append(agent_id)

    total = BATTERY_SIZE * args.runs * len(agent_configs)
    print(f"\nSENTINEL Calibration Battery")
    print(f"Agents: {len(agent_configs)} | Prompts per agent: {BATTERY_SIZE} | Runs: {args.runs}")
    print(f"Total inferences: {total}")
    print(f"Database: {args.db}\n")

    async def on_progress(msg):
        print(msg)

    calibration_ids = await calibrate_all_agents(
        db=db,
        client=client,
        experiment_id=experiment_id,
        agent_configs=agent_configs,
        agent_ids=agent_ids,
        model_digests=model_digests,
        num_runs=args.runs,
        on_progress=on_progress,
    )

    print(f"\nCalibration complete.")
    for agent_id, cal_id in calibration_ids.items():
        name = db.conn.execute("SELECT name FROM agents WHERE agent_id=?", (agent_id,)).fetchone()[0]
        count = db.conn.execute(
            "SELECT COUNT(*) as n FROM calibrations WHERE calibration_id=?", (cal_id,)
        ).fetchone()[0]
        print(f"  {name}: {cal_id[:8]}... ({count} responses)")

    db.close()


if __name__ == "__main__":
    asyncio.run(main())
