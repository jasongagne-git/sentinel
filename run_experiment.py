#!/usr/bin/env python3
"""Run a SENTINEL multi-agent experiment.

Usage:
    python3 run_experiment.py config/three_agent_mesh.json
    python3 run_experiment.py config/three_agent_mesh.json --max-turns 30 --delay 10
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from sentinel.agent import AgentConfig
from sentinel.db import Database
from sentinel.ollama import OllamaClient
from sentinel.persona import load_persona_config
from sentinel.probes import TriggerConfig
from sentinel.runtime import create_experiment


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def print_message(turn: int, agent_name: str, content: str):
    """Live output callback — prints each message as it's generated."""
    print(f"\n{'='*60}")
    print(f"Turn {turn} | {agent_name}")
    print(f"{'='*60}")
    # Truncate long messages for display
    display = content[:500] + "..." if len(content) > 500 else content
    print(display)


async def main():
    parser = argparse.ArgumentParser(description="Run a SENTINEL experiment")
    parser.add_argument("config", help="Path to experiment config JSON")
    parser.add_argument("--max-turns", type=int, default=None, help="Max interaction turns")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between turns")
    parser.add_argument("--probe", default=None, choices=["shadow", "injected", "both"],
                        help="Enable probes: shadow (no contamination), injected (enters conversation), both")
    parser.add_argument("--probe-interval", type=int, default=20, help="Probe every N turns (default 20)")
    parser.add_argument("--probe-strategy", default="scheduled",
                        choices=["scheduled", "triggered", "hybrid"],
                        help="Probe trigger strategy (default: scheduled)")
    parser.add_argument("--vocab-threshold", type=float, default=0.15,
                        help="Vocabulary JSD threshold for triggered probes (default 0.15)")
    parser.add_argument("--sentiment-threshold", type=float, default=0.3,
                        help="Sentiment delta threshold for triggered probes (default 0.3)")
    parser.add_argument("--db", default="experiments/sentinel.db", help="Database path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    # Override from CLI
    max_turns = args.max_turns or config.get("max_turns")
    cycle_delay = args.delay if args.delay is not None else config.get("cycle_delay_s", 30.0)

    # Connect
    client = OllamaClient()
    if not client.is_available():
        print("Error: Ollama is not running. Start it with: systemctl start ollama", file=sys.stderr)
        sys.exit(1)

    db = Database(args.db)

    # Build agent configs
    agent_configs = []
    for agent_def in config["agents"]:
        name, traits, system_prompt = load_persona_config(agent_def)
        fingerprint = traits.fingerprint() if traits else None
        traits_json_str = json.dumps(agent_def["traits"]) if traits else None
        if traits:
            print(f"  {name}: traits={fingerprint} ({traits.role}/{traits.disposition}/{traits.values})")
        agent_configs.append(AgentConfig(
            name=name,
            system_prompt=system_prompt,
            model=agent_def.get("model", config.get("default_model", "llama3:latest")),
            temperature=agent_def.get("temperature", config.get("default_temperature", 0.7)),
            max_history=agent_def.get("max_history", config.get("default_max_history", 50)),
            response_limit=agent_def.get("response_limit", config.get("default_response_limit", 256)),
            is_control=agent_def.get("is_control", False),
            traits_json=traits_json_str,
            trait_fingerprint=fingerprint,
        ))

    # Build trigger config if using triggered/hybrid strategy
    trigger_config = None
    if args.probe_strategy in ("triggered", "hybrid"):
        trigger_config = TriggerConfig(
            vocab_jsd_threshold=args.vocab_threshold,
            sentiment_threshold=args.sentiment_threshold,
        )

    # Create and run
    runtime = create_experiment(
        db=db,
        client=client,
        name=config["name"],
        agent_configs=agent_configs,
        topology=config.get("topology", "full_mesh"),
        cycle_delay_s=cycle_delay,
        max_turns=max_turns,
        description=config.get("description", ""),
        probe_mode=args.probe,
        probe_interval=args.probe_interval,
        probe_strategy=args.probe_strategy,
        trigger_config=trigger_config,
    )

    print(f"\nSENTINEL Experiment: {config['name']}")
    print(f"Agents: {len(agent_configs)} | Topology: {config.get('topology', 'full_mesh')}")
    print(f"Max turns: {max_turns or 'unlimited'} | Delay: {cycle_delay}s")
    if args.probe:
        print(f"Probes: {args.probe} | strategy: {args.probe_strategy} | interval: {args.probe_interval} turns")
        if args.probe_strategy in ("triggered", "hybrid"):
            print(f"  Thresholds: vocab_jsd={args.vocab_threshold}, sentiment={args.sentiment_threshold}")
    print(f"Database: {args.db}")
    print(f"\nPress Ctrl+C to stop gracefully.\n")

    try:
        await runtime.run(
            max_turns=max_turns,
            cycle_delay_s=cycle_delay,
            on_message=print_message,
        )
    finally:
        db.close()

    print("\nExperiment complete.")


if __name__ == "__main__":
    asyncio.run(main())
