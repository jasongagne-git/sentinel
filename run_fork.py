#!/usr/bin/env python3
"""Fork a SENTINEL experiment and run the fork.

Usage:
    # Fork the most recent experiment at its last turn
    python3 run_fork.py --max-turns 30 --delay 5

    # Fork a specific experiment at turn 50
    python3 run_fork.py -e 3f9dd450 --at-turn 50 --max-turns 30

    # Fork with a trait mutation — make Aria contrarian
    python3 run_fork.py -e 3f9dd450 --mutate "Aria:disposition=contrarian" --max-turns 30

    # List forks of an experiment
    python3 run_fork.py -e 3f9dd450 --list-forks

    # Show lineage of an experiment
    python3 run_fork.py -e 3f9dd450 --lineage
"""

import argparse
import asyncio
import json
import logging
import sys

from sentinel.db import Database
from sentinel.fork import fork_experiment, list_forks, get_fork_lineage
from sentinel.metrics import MetricsConfig, run_metrics_pipeline, print_metrics_summary
from sentinel.ollama import OllamaClient


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_experiment(db, experiment_id_prefix=None):
    """Find an experiment by prefix or return the most recent."""
    if experiment_id_prefix:
        row = db.conn.execute(
            "SELECT experiment_id, name FROM experiments WHERE experiment_id LIKE ?",
            (experiment_id_prefix + "%",),
        ).fetchone()
    else:
        row = db.conn.execute(
            "SELECT experiment_id, name FROM experiments "
            "WHERE status='completed' AND experiment_id IN "
            "(SELECT DISTINCT experiment_id FROM messages) "
            "AND topology != 'none' "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return row


def parse_mutations(mutation_strs: list[str]) -> dict[str, dict]:
    """Parse mutation strings like 'Aria:disposition=contrarian'."""
    overrides = {}
    for s in mutation_strs:
        if ":" not in s:
            raise ValueError(f"Invalid mutation format: '{s}'. Use 'AgentName:trait=value'")
        agent_name, trait_spec = s.split(":", 1)
        if "=" not in trait_spec:
            raise ValueError(f"Invalid trait spec: '{trait_spec}'. Use 'trait=value'")
        trait_dim, trait_val = trait_spec.split("=", 1)
        if agent_name not in overrides:
            overrides[agent_name] = {}
        overrides[agent_name][trait_dim] = trait_val
    return overrides


async def main():
    parser = argparse.ArgumentParser(description="Fork a SENTINEL experiment")
    parser.add_argument("-e", "--experiment-id", help="Source experiment ID (prefix match)")
    parser.add_argument("--at-turn", type=int, default=None, help="Fork at this turn (default: latest)")
    parser.add_argument("--mutate", action="append", default=[],
                        help="Trait mutation: 'AgentName:trait=value' (repeatable)")
    parser.add_argument("--max-turns", type=int, default=None, help="Max turns for the fork")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between turns")
    parser.add_argument("--suffix", default=None, help="Fork name suffix")
    parser.add_argument("--list-forks", action="store_true", help="List forks of an experiment")
    parser.add_argument("--lineage", action="store_true", help="Show fork lineage")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--db", default="experiments/sentinel.db")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    db = Database(args.db)

    source = resolve_experiment(db, args.experiment_id)
    if not source:
        print("No suitable experiment found.", file=sys.stderr)
        db.close()
        sys.exit(1)

    source_id = source["experiment_id"]

    # List forks
    if args.list_forks:
        forks = list_forks(db, source_id)
        print(f"\nForks of {source['name']} ({source_id[:8]}):")
        if not forks:
            print("  (none)")
        for f in forks:
            print(f"  {f['experiment_id'][:8]}  @t{f['fork_at_turn']}  {f['status']:10s}  {f['name']}")
        db.close()
        return

    # Show lineage
    if args.lineage:
        lineage = get_fork_lineage(db, source_id)
        print(f"\nLineage of {source_id[:8]}:")
        for i, exp in enumerate(lineage):
            prefix = "  └─ " if i > 0 else "  "
            fork_info = f" (forked @t{exp['fork_at_turn']})" if exp['fork_at_turn'] else " (root)"
            print(f"{prefix}{exp['experiment_id'][:8]}  {exp['status']:10s}  {exp['name']}{fork_info}")
        db.close()
        return

    # Parse mutations
    trait_overrides = None
    if args.mutate:
        trait_overrides = parse_mutations(args.mutate)

    # Determine suffix
    suffix = args.suffix
    if not suffix:
        if trait_overrides:
            parts = []
            for name, changes in trait_overrides.items():
                for dim, val in changes.items():
                    parts.append(f"{name}-{val}")
            suffix = "mutate-" + "-".join(parts)
        else:
            suffix = "fork"

    client = OllamaClient()
    if not client.is_available():
        print("Error: Ollama is not running.", file=sys.stderr)
        db.close()
        sys.exit(1)

    # Fork
    at_turn = args.at_turn or db.get_latest_turn(source_id)
    print(f"\nForking experiment: {source['name']}")
    print(f"  Source:    {source_id[:8]}")
    print(f"  At turn:   {at_turn}")
    if trait_overrides:
        for name, changes in trait_overrides.items():
            print(f"  Mutation:  {name} → {changes}")
    print()

    fork_runtime = fork_experiment(
        db=db,
        client=client,
        source_experiment_id=source_id,
        fork_at_turn=at_turn,
        name_suffix=suffix,
        trait_overrides=trait_overrides,
    )

    max_turns = args.max_turns or at_turn  # Default: run same number of turns as source

    async def on_message(turn, agent_name, content):
        preview = content[:80].replace("\n", " ")
        print(f"    turn {turn:3d} | {agent_name}: {preview}...")

    print(f"Running fork for {max_turns} turns...\n")
    await fork_runtime.run(
        max_turns=max_turns,
        cycle_delay_s=args.delay,
        on_message=on_message,
    )

    fork_id = fork_runtime.experiment_id

    if not args.skip_metrics:
        print(f"\nComputing metrics...")
        metrics_config = MetricsConfig(window_size=10)
        summary = run_metrics_pipeline(
            db, client, fork_id, metrics_config,
            on_progress=lambda msg: print(msg),
        )
        print_metrics_summary(summary)

    # Show lineage
    lineage = get_fork_lineage(db, fork_id)
    print(f"\nFork complete.")
    print(f"  Fork ID:  {fork_id[:8]}")
    print(f"  Lineage:  {' → '.join(e['experiment_id'][:8] for e in lineage)}")

    db.close()


if __name__ == "__main__":
    asyncio.run(main())
