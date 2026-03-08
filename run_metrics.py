#!/usr/bin/env python3
"""Run SENTINEL metrics pipeline on a completed experiment.

Usage:
    python3 run_metrics.py                           # analyze most recent experiment
    python3 run_metrics.py --experiment-id <uuid>    # analyze specific experiment
    python3 run_metrics.py --list                    # list all experiments
    python3 run_metrics.py --fast                    # skip Ollama-based metrics (vocab + sentiment only)
"""

import argparse
import logging
import sys

from sentinel.db import Database
from sentinel.metrics import MetricsConfig, run_metrics_pipeline, print_metrics_summary
from sentinel.ollama import OllamaClient


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def list_experiments(db: Database):
    rows = db.conn.execute(
        "SELECT experiment_id, name, status, created_at, "
        "(SELECT COUNT(*) FROM messages WHERE messages.experiment_id = experiments.experiment_id) as msg_count, "
        "(SELECT COUNT(*) FROM agents WHERE agents.experiment_id = experiments.experiment_id) as agent_count "
        "FROM experiments ORDER BY created_at DESC"
    ).fetchall()
    if not rows:
        print("No experiments found.")
        return
    print(f"\n{'ID':10s} {'Status':10s} {'Agents':>6s} {'Messages':>8s}  {'Name'}")
    print("-" * 70)
    for r in rows:
        print(f"{r['experiment_id'][:8]:10s} {r['status']:10s} {r['agent_count']:6d} {r['msg_count']:8d}  {r['name']}")


def main():
    parser = argparse.ArgumentParser(description="Run SENTINEL metrics pipeline")
    parser.add_argument("--experiment-id", "-e", help="Experiment ID (prefix match)")
    parser.add_argument("--list", "-l", action="store_true", help="List all experiments")
    parser.add_argument("--fast", action="store_true", help="Skip Ollama-based metrics")
    parser.add_argument("--window-size", "-w", type=int, default=10, help="Analysis window size")
    parser.add_argument("--db", default="experiments/sentinel.db", help="Database path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    db = Database(args.db)

    if args.list:
        list_experiments(db)
        db.close()
        return

    # Find experiment
    if args.experiment_id:
        # Prefix match
        row = db.conn.execute(
            "SELECT experiment_id, name FROM experiments WHERE experiment_id LIKE ?",
            (args.experiment_id + "%",)
        ).fetchone()
        if not row:
            print(f"No experiment found matching: {args.experiment_id}", file=sys.stderr)
            db.close()
            sys.exit(1)
    else:
        # Most recent with messages
        row = db.conn.execute(
            "SELECT experiment_id, name FROM experiments "
            "WHERE status='completed' AND experiment_id IN "
            "(SELECT DISTINCT experiment_id FROM messages) "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            print("No completed experiments with messages found.", file=sys.stderr)
            db.close()
            sys.exit(1)

    experiment_id = row["experiment_id"]
    experiment_name = row["name"]

    client = OllamaClient()
    if not args.fast and not client.is_available():
        print("Ollama not running. Use --fast for vocab/sentiment only.", file=sys.stderr)
        db.close()
        sys.exit(1)

    config = MetricsConfig(
        window_size=args.window_size,
        compute_semantic=not args.fast,
        compute_persona=not args.fast,
    )

    msg_count = db.conn.execute(
        "SELECT COUNT(*) as n FROM messages WHERE experiment_id=?",
        (experiment_id,)
    ).fetchone()["n"]

    print(f"\nSENTINEL Metrics Pipeline")
    print(f"Experiment: {experiment_name}")
    print(f"ID: {experiment_id[:8]}...")
    print(f"Messages: {msg_count} | Window size: {config.window_size}")
    print(f"Dimensions: vocabulary_drift, sentiment_trajectory"
          + (", semantic_coherence, persona_adherence" if not args.fast else " (fast mode)"))
    print()

    def on_progress(msg):
        print(msg)

    summary = run_metrics_pipeline(db, client, experiment_id, config, on_progress)
    print_metrics_summary(summary)

    db.close()


if __name__ == "__main__":
    main()
