#!/usr/bin/env python3

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

"""Resume a SENTINEL experiment that was interrupted mid-run.

Usage:
    python3 run_resume.py -e 387fb9a3 --max-turns 500 --delay 5
"""

import argparse
import asyncio
import json
import logging
import signal
import sys

from sentinel.agent import Agent, AgentConfig
from sentinel.db import Database
from sentinel.metrics import MetricsConfig, run_metrics_pipeline, print_metrics_summary
from sentinel.ollama import OllamaClient
from sentinel.runtime import ExperimentRuntime


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def main():
    parser = argparse.ArgumentParser(description="Resume an interrupted SENTINEL experiment")
    parser.add_argument("-e", "--experiment-id", required=True,
                        help="Experiment ID to resume (prefix match)")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Max turns (default: use experiment's configured max)")
    parser.add_argument("--delay", type=float, default=5.0, help="Delay between turns")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--db", default="experiments/sentinel.db")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger("sentinel.resume")

    db = Database(args.db)

    # Find the experiment
    row = db.conn.execute(
        "SELECT experiment_id, name, status, max_turns FROM experiments "
        "WHERE experiment_id LIKE ?",
        (args.experiment_id + "%",),
    ).fetchone()
    if not row:
        print(f"No experiment found matching '{args.experiment_id}'", file=sys.stderr)
        db.close()
        sys.exit(1)

    exp_id = row["experiment_id"]
    current_turn, last_agent_id = db.get_resume_position(exp_id)
    max_turns = args.max_turns or row["max_turns"]

    # Show which agent spoke last for mid-cycle awareness
    last_agent_name = None
    if last_agent_id:
        arow = db.conn.execute(
            "SELECT name FROM agents WHERE agent_id=?", (last_agent_id,),
        ).fetchone()
        last_agent_name = arow["name"] if arow else last_agent_id[:8]

    print(f"Resuming experiment: {row['name']}")
    print(f"  ID:           {exp_id[:8]}")
    print(f"  Status:       {row['status']}")
    print(f"  Current turn: {current_turn}")
    if last_agent_name:
        print(f"  Last agent:   {last_agent_name}")
    print(f"  Target turns: {max_turns}")
    print(f"  Remaining:    {max_turns - current_turn}")
    print()

    if current_turn >= max_turns:
        print("Experiment already at or past max turns. Nothing to resume.")
        db.close()
        return

    client = OllamaClient()
    if not client.is_available():
        print("Error: Ollama is not running.", file=sys.stderr)
        db.close()
        sys.exit(1)

    # Reconstruct runtime with existing agents
    runtime = ExperimentRuntime(db, client, exp_id)
    agents_rows = db.get_agents(exp_id)
    if not agents_rows:
        print("No agents found for this experiment.", file=sys.stderr)
        db.close()
        sys.exit(1)

    digest_cache = {}
    for agent_row in agents_rows:
        model = agent_row["model"]
        if model not in digest_cache:
            digest_cache[model] = client.get_model_digest(model)

        config = AgentConfig(
            name=agent_row["name"],
            system_prompt=agent_row["system_prompt"],
            model=model,
            temperature=agent_row["temperature"],
            max_history=agent_row["max_history"],
            response_limit=agent_row["response_limit"],
            is_control=bool(agent_row["is_control"]),
            traits_json=agent_row["traits_json"],
            trait_fingerprint=agent_row["trait_fingerprint"],
        )
        agent = Agent(agent_row["agent_id"], config, client, digest_cache[model])
        runtime.add_agent(agent)

    log.info("Reconstructed %d agents, resuming from turn %d", len(agents_rows), current_turn)

    async def on_message(turn, agent_name, content):
        preview = content[:80].replace("\n", " ")
        print(f"    turn {turn:3d} | {agent_name}: {preview}...")

    try:
        await runtime.run(
            max_turns=max_turns,
            cycle_delay_s=args.delay,
            on_message=on_message,
        )
    except Exception as exc:
        log.error("Resume run failed: %s", exc)
        print(f"\nERROR: Resume failed: {exc}", file=sys.stderr)

    if not args.skip_metrics:
        print(f"\nComputing metrics...")
        try:
            metrics_config = MetricsConfig(window_size=10)
            summary = run_metrics_pipeline(
                db, client, exp_id, metrics_config,
                on_progress=lambda msg: print(msg),
            )
            print_metrics_summary(summary)
        except Exception as exc:
            log.error("Metrics failed: %s", exc)

    final_turn = db.get_latest_turn(exp_id)
    print(f"\nResume complete. Final turn: {final_turn}")
    db.close()


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    asyncio.run(main())
