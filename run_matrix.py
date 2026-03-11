#!/usr/bin/env python3
"""Run a model matrix — same experiment across multiple model configurations.

Usage:
    python3 run_matrix.py config/three_agent_mesh.json
    python3 run_matrix.py config/three_agent_mesh.json --mode asymmetric --max-turns 30
    python3 run_matrix.py config/three_agent_mesh.json --models gemma2:2b,llama3.2:3b --mode homogeneous
    python3 run_matrix.py config/three_agent_mesh.json --preview  # dry run
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

from sentinel.agent import AgentConfig
from sentinel.db import Database
from sentinel.metrics import MetricsConfig, run_metrics_pipeline, print_metrics_summary
from sentinel.models import discover_models, generate_model_matrix, print_model_matrix_preview
from sentinel.ollama import OllamaClient
from sentinel.persona import load_persona_config
from sentinel.runtime import create_experiment


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def run_single_experiment(db, client, config, max_turns, cycle_delay):
    """Run a single experiment from a config dict. Returns experiment_id."""
    agent_configs = []
    for agent_def in config["agents"]:
        name, traits, system_prompt = load_persona_config(agent_def)
        fingerprint = traits.fingerprint() if traits else None
        traits_json_str = json.dumps(agent_def["traits"]) if traits else None
        agent_configs.append(AgentConfig(
            name=name,
            system_prompt=system_prompt,
            model=agent_def.get("model", config.get("default_model", "llama3.2:3b")),
            temperature=agent_def.get("temperature", config.get("default_temperature", 0.7)),
            max_history=agent_def.get("max_history", config.get("default_max_history", 50)),
            response_limit=agent_def.get("response_limit", config.get("default_response_limit", 256)),
            is_control=agent_def.get("is_control", False),
            traits_json=traits_json_str,
            trait_fingerprint=fingerprint,
        ))

    _max_turns = max_turns or config.get("max_turns")
    _cycle_delay = cycle_delay if cycle_delay is not None else config.get("cycle_delay_s", 30.0)

    runtime = create_experiment(
        db=db,
        client=client,
        name=config["name"],
        agent_configs=agent_configs,
        topology=config.get("topology", "full_mesh"),
        cycle_delay_s=_cycle_delay,
        max_turns=_max_turns,
        description=config.get("description", ""),
    )

    async def on_message(turn, agent_name, content):
        preview = content[:80].replace("\n", " ")
        print(f"    turn {turn:3d} | {agent_name}: {preview}...")

    await runtime.run(max_turns=_max_turns, cycle_delay_s=_cycle_delay, on_message=on_message)
    return runtime.experiment_id


async def main():
    parser = argparse.ArgumentParser(description="Run a SENTINEL model matrix")
    parser.add_argument("config", help="Path to base experiment config JSON")
    parser.add_argument("--mode", default="homogeneous",
                        choices=["homogeneous", "asymmetric", "mixed"])
    parser.add_argument("--models", help="Comma-separated model names (default: all available)")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument("--preview", action="store_true", help="Preview only, don't run")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip metrics computation")
    parser.add_argument("--db", default="experiments/sentinel.db")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        base_config = json.load(f)

    client = OllamaClient()
    if not client.is_available():
        print("Error: Ollama is not running.", file=sys.stderr)
        sys.exit(1)

    available = discover_models(client)
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        available_names = {m.name for m in available}
        for name in model_names:
            if name not in available_names:
                print(f"Model not found: {name}. Available: {', '.join(available_names)}", file=sys.stderr)
                sys.exit(1)
    else:
        model_names = [m.name for m in available]

    configs = generate_model_matrix(base_config, model_names, args.mode)
    print_model_matrix_preview(configs)

    if args.preview:
        return

    print(f"\nRunning {len(configs)} experiment(s)...\n")
    db = Database(args.db)
    experiment_ids = []

    log = logging.getLogger("sentinel.matrix")

    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(configs)}: {config['name']}")
        print(f"{'='*60}")

        try:
            experiment_id = await run_single_experiment(
                db, client, config,
                max_turns=args.max_turns,
                cycle_delay=args.delay,
            )
            experiment_ids.append((config["name"], experiment_id))
        except Exception as exc:
            log.error("Experiment %d/%d (%s) failed: %s", i, len(configs), config["name"], exc)
            print(f"\nERROR: {config['name']} failed: {exc}. Continuing...", file=sys.stderr)

    # Run metrics on all experiments
    if not args.skip_metrics:
        print(f"\n{'='*60}")
        print("Running metrics on all experiments...")
        print(f"{'='*60}")

        metrics_config = MetricsConfig(window_size=5)
        for name, eid in experiment_ids:
            print(f"\nMetrics for: {name}")
            try:
                summary = run_metrics_pipeline(
                    db, client, eid, metrics_config,
                    on_progress=lambda msg: print(msg),
                )
                print_metrics_summary(summary)
            except Exception as exc:
                log.error("Metrics for %s failed: %s", name, exc)

    print(f"\n{'='*60}")
    print(f"Model Matrix Complete — {len(configs)} experiments")
    for name, eid in experiment_ids:
        print(f"  {eid[:8]}  {name}")
    print(f"Database: {args.db}")

    db.close()


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    asyncio.run(main())
