#!/usr/bin/env python3
"""Run a paired SENTINEL experiment — experimental arm + control arm.

Runs both arms sequentially with identical agent configurations.
Experimental agents interact; control agents respond in isolation.
Metrics are computed for both and compared.

Usage:
    python3 run_paired.py config/three_agent_mesh.json --max-turns 30 --delay 5
    python3 run_paired.py config/three_agent_mesh.json --max-turns 100 --delay 5 --skip-metrics
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

from sentinel.agent import AgentConfig
from sentinel.control import create_control_experiment
from sentinel.db import Database
from sentinel.metrics import MetricsConfig, run_metrics_pipeline, print_metrics_summary
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


def build_agent_configs(config: dict) -> list[AgentConfig]:
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
            is_control=False,
            traits_json=traits_json_str,
            trait_fingerprint=fingerprint,
        ))
    return agent_configs


async def print_message(turn: int, agent_name: str, content: str):
    preview = content[:80].replace("\n", " ")
    print(f"    turn {turn:3d} | {agent_name}: {preview}...")


def print_comparison(exp_summary: dict, ctrl_summary: dict):
    """Print side-by-side comparison of experimental vs. control metrics."""
    print(f"\n{'='*70}")
    print("EXPERIMENTAL vs. CONTROL COMPARISON")
    print(f"{'='*70}")

    all_agents = set(exp_summary.keys()) | set(ctrl_summary.keys())

    for agent_name in sorted(all_agents):
        print(f"\n  Agent: {agent_name}")
        print(f"  {'Dimension':<25s} {'Experimental':>14s} {'Control':>14s} {'Δ (exp-ctrl)':>14s}")
        print(f"  {'-'*67}")

        exp = exp_summary.get(agent_name, {})
        ctrl = ctrl_summary.get(agent_name, {})

        for dim in ["vocabulary_drift", "sentiment_trajectory", "semantic_coherence", "persona_adherence"]:
            exp_data = exp.get(dim, [])
            ctrl_data = ctrl.get(dim, [])

            if not exp_data and not ctrl_data:
                continue

            # Get final window value for each
            if dim == "vocabulary_drift":
                exp_val = exp_data[-1]["jsd"] if exp_data else None
                ctrl_val = ctrl_data[-1]["jsd"] if ctrl_data else None
                label = "vocab drift (JSD)"
            elif dim == "sentiment_trajectory":
                exp_val = exp_data[-1]["sentiment_shift"] if exp_data else None
                ctrl_val = ctrl_data[-1]["sentiment_shift"] if ctrl_data else None
                label = "sentiment shift"
            elif dim == "semantic_coherence":
                exp_val = exp_data[-1]["mean_similarity"] if exp_data else None
                ctrl_val = ctrl_data[-1]["mean_similarity"] if ctrl_data else None
                label = "semantic coherence"
            elif dim == "persona_adherence":
                exp_val = exp_data[-1]["score"] if exp_data else None
                ctrl_val = ctrl_data[-1]["score"] if ctrl_data else None
                label = "persona (1-10)"

            exp_str = f"{exp_val:+.4f}" if exp_val is not None else "n/a"
            ctrl_str = f"{ctrl_val:+.4f}" if ctrl_val is not None else "n/a"

            if exp_val is not None and ctrl_val is not None:
                delta = exp_val - ctrl_val
                delta_str = f"{delta:+.4f}"
                # Flag significant differences
                flag = " ***" if abs(delta) > 0.1 else ""
            else:
                delta_str = "n/a"
                flag = ""

            print(f"  {label:<25s} {exp_str:>14s} {ctrl_str:>14s} {delta_str:>14s}{flag}")

    print(f"\n  *** = difference > 0.1 (potentially significant)")


async def main():
    parser = argparse.ArgumentParser(description="Run paired SENTINEL experiment")
    parser.add_argument("config", help="Path to experiment config JSON")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--probe", default=None, choices=["shadow", "injected", "both"],
                        help="Enable probes: shadow, injected, or both")
    parser.add_argument("--probe-interval", type=int, default=20, help="Probe every N turns")
    parser.add_argument("--probe-strategy", default="scheduled",
                        choices=["scheduled", "triggered", "hybrid"],
                        help="Probe trigger strategy")
    parser.add_argument("--vocab-threshold", type=float, default=0.15,
                        help="Vocabulary JSD threshold for triggered probes")
    parser.add_argument("--sentiment-threshold", type=float, default=0.3,
                        help="Sentiment delta threshold for triggered probes")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--fast-metrics", action="store_true", help="Skip Ollama-based metrics")
    parser.add_argument("--window-size", "-w", type=int, default=10)
    parser.add_argument("--db", default="experiments/sentinel.db")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        config = json.load(f)

    client = OllamaClient()
    if not client.is_available():
        print("Error: Ollama is not running.", file=sys.stderr)
        sys.exit(1)

    db = Database(args.db)
    agent_configs = build_agent_configs(config)

    # Build trigger config if using triggered/hybrid strategy
    trigger_config = None
    if args.probe_strategy in ("triggered", "hybrid"):
        trigger_config = TriggerConfig(
            vocab_jsd_threshold=args.vocab_threshold,
            sentiment_threshold=args.sentiment_threshold,
        )

    print(f"\nSENTINEL Paired Experiment")
    print(f"Config: {config['name']}")
    print(f"Agents: {len(agent_configs)} | Turns: {args.max_turns} | Delay: {args.delay}s")
    if args.probe:
        print(f"Probes: {args.probe} | strategy: {args.probe_strategy} | interval: {args.probe_interval} turns")
    print(f"Each arm runs independently with identical agent configurations.\n")

    log = logging.getLogger("sentinel.paired")
    exp_id = None
    ctrl_id = None

    # -- Experimental Arm --
    print(f"{'='*60}")
    print(f"EXPERIMENTAL ARM (agents interact)")
    print(f"{'='*60}")

    try:
        exp_runtime = create_experiment(
            db=db, client=client,
            name=config["name"],
            agent_configs=agent_configs,
            topology=config.get("topology", "full_mesh"),
            cycle_delay_s=args.delay,
            max_turns=args.max_turns,
            description=config.get("description", ""),
            probe_mode=args.probe,
            probe_interval=args.probe_interval,
            probe_strategy=args.probe_strategy,
            trigger_config=trigger_config,
        )

        await exp_runtime.run(
            max_turns=args.max_turns,
            cycle_delay_s=args.delay,
            on_message=print_message,
        )
        exp_id = exp_runtime.experiment_id
    except Exception as exc:
        log.error("Experimental arm failed: %s", exc)
        print(f"\nERROR: Experimental arm failed: {exc}", file=sys.stderr)
        print("Continuing to control arm...\n")

    # -- Control Arm --
    print(f"\n{'='*60}")
    print(f"CONTROL ARM (agents isolated)")
    print(f"{'='*60}")

    try:
        ctrl_runtime = create_control_experiment(
            db=db, client=client,
            name=config["name"],
            agent_configs=agent_configs,
            cycle_delay_s=args.delay,
            max_turns=args.max_turns,
            description=config.get("description", ""),
            linked_experiment_id=exp_id,
        )

        await ctrl_runtime.run(
            max_turns=args.max_turns,
            cycle_delay_s=args.delay,
            on_message=print_message,
        )
        ctrl_id = ctrl_runtime.experiment_id
    except Exception as exc:
        log.error("Control arm failed: %s", exc)
        print(f"\nERROR: Control arm failed: {exc}", file=sys.stderr)

    # -- Metrics --
    if not args.skip_metrics and (exp_id or ctrl_id):
        print(f"\n{'='*60}")
        print(f"COMPUTING METRICS")
        print(f"{'='*60}")

        metrics_config = MetricsConfig(
            window_size=args.window_size,
            compute_semantic=not args.fast_metrics,
            compute_persona=not args.fast_metrics,
        )

        exp_summary = {}
        ctrl_summary = {}

        if exp_id:
            print(f"\nExperimental arm metrics:")
            try:
                exp_summary = run_metrics_pipeline(
                    db, client, exp_id, metrics_config,
                    on_progress=lambda msg: print(msg),
                )
                print_metrics_summary(exp_summary)
            except Exception as exc:
                log.error("Experimental metrics failed: %s", exc)

        if ctrl_id:
            print(f"\nControl arm metrics:")
            try:
                ctrl_summary = run_metrics_pipeline(
                    db, client, ctrl_id, metrics_config,
                    on_progress=lambda msg: print(msg),
                )
                print_metrics_summary(ctrl_summary)
            except Exception as exc:
                log.error("Control metrics failed: %s", exc)

        # -- Comparison --
        if exp_summary and ctrl_summary:
            print_comparison(exp_summary, ctrl_summary)

    print(f"\n{'='*60}")
    print(f"Paired Experiment Complete")
    print(f"  Experimental: {exp_id[:8] if exp_id else 'FAILED'}")
    print(f"  Control:      {ctrl_id[:8] if ctrl_id else 'FAILED'}")
    print(f"  Database:     {args.db}")

    db.close()


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    asyncio.run(main())
