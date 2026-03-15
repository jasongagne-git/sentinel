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

"""Compare two SENTINEL experiments side by side.

Usage:
    # Diff two experiments by ID prefix
    python3 run_diff.py -a 3f9dd450 -b c6793fd1

    # Diff most recent experiment against its parent fork
    python3 run_diff.py --fork-parent

    # Diff most recent paired experiment (experimental vs control)
    python3 run_diff.py --paired

    # List experiments to find IDs
    python3 run_diff.py --list
"""

import argparse
import json
import sys

from sentinel.db import Database
from sentinel.diff import diff_experiments, print_diff


def resolve_experiment(db, prefix):
    """Find an experiment by prefix."""
    row = db.conn.execute(
        "SELECT experiment_id, name FROM experiments WHERE experiment_id LIKE ?",
        (prefix + "%",),
    ).fetchone()
    return row


def find_latest_with_messages(db):
    """Find the most recent completed experiment with messages."""
    return db.conn.execute(
        "SELECT experiment_id, name FROM experiments "
        "WHERE experiment_id IN (SELECT DISTINCT experiment_id FROM messages) "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()


def find_fork_parent(db, experiment_id):
    """Find the parent experiment of a fork."""
    exp = db.get_experiment(experiment_id)
    if not exp or not exp["forked_from_experiment_id"]:
        return None
    return db.get_experiment(exp["forked_from_experiment_id"])


def find_paired_control(db, experiment_id):
    """Find the control experiment linked to an experimental run."""
    # Control experiments reference their experimental counterpart in config
    exp = db.get_experiment(experiment_id)
    if not exp:
        return None

    config = json.loads(exp["config_json"]) if exp["config_json"] else {}

    # Check if this is a control experiment pointing to its experimental arm
    linked = config.get("linked_experiment_id")
    if linked:
        return db.get_experiment(linked)

    # Check if any control experiment points to this one
    rows = db.conn.execute(
        "SELECT experiment_id FROM experiments WHERE topology='none'"
    ).fetchall()
    for row in rows:
        ctrl = db.get_experiment(row["experiment_id"])
        if ctrl:
            ctrl_config = json.loads(ctrl["config_json"]) if ctrl["config_json"] else {}
            if ctrl_config.get("linked_experiment_id") == experiment_id:
                return ctrl

    return None


def list_experiments(db):
    """List all experiments."""
    rows = db.conn.execute(
        "SELECT e.experiment_id, e.name, e.status, e.topology, "
        "e.forked_from_experiment_id, e.fork_at_turn, "
        "(SELECT COUNT(*) FROM messages m WHERE m.experiment_id=e.experiment_id) as msg_count, "
        "(SELECT MAX(interaction_turn) FROM messages m WHERE m.experiment_id=e.experiment_id) as max_turn "
        "FROM experiments e ORDER BY e.created_at DESC"
    ).fetchall()

    if not rows:
        print("No experiments found.")
        return

    print(f"\n{'ID':<10s}  {'Status':<10s}  {'Turns':>5s}  {'Msgs':>5s}  {'Topo':<10s}  {'Fork':>8s}  Name")
    print("-" * 85)
    for r in rows:
        fork = f"@t{r['fork_at_turn']}" if r["fork_at_turn"] else ""
        parent = r["forked_from_experiment_id"][:8] if r["forked_from_experiment_id"] else ""
        fork_str = f"{parent}{fork}" if parent else ""
        print(f"{r['experiment_id'][:8]:<10s}  {r['status']:<10s}  "
              f"{r['max_turn'] or 0:5d}  {r['msg_count']:5d}  "
              f"{r['topology']:<10s}  {fork_str:>8s}  {r['name']}")


def main():
    parser = argparse.ArgumentParser(description="Compare two SENTINEL experiments")
    parser.add_argument("-a", "--experiment-a", help="First experiment ID (prefix)")
    parser.add_argument("-b", "--experiment-b", help="Second experiment ID (prefix)")
    parser.add_argument("--fork-parent", action="store_true",
                        help="Auto-detect: diff most recent experiment against its fork parent")
    parser.add_argument("--paired", action="store_true",
                        help="Auto-detect: diff most recent experimental run against its control arm")
    parser.add_argument("--match-by", default="name", choices=["name", "position"],
                        help="How to pair agents across experiments (default: name)")
    parser.add_argument("--list", "-l", action="store_true", help="List all experiments")
    parser.add_argument("--db", default="experiments/sentinel.db")
    args = parser.parse_args()

    db = Database(args.db)

    if args.list:
        list_experiments(db)
        db.close()
        return

    # Resolve experiment IDs
    exp_a_id = None
    exp_b_id = None

    if args.fork_parent:
        latest = find_latest_with_messages(db)
        if not latest:
            print("No experiments with messages found.", file=sys.stderr)
            db.close()
            sys.exit(1)
        exp = db.get_experiment(latest["experiment_id"])
        if not exp or not exp["forked_from_experiment_id"]:
            print(f"Most recent experiment ({latest['experiment_id'][:8]}) is not a fork.", file=sys.stderr)
            db.close()
            sys.exit(1)
        exp_a_id = exp["forked_from_experiment_id"]
        exp_b_id = latest["experiment_id"]
        print(f"Comparing fork parent → fork:")
        print(f"  A (parent): {exp_a_id[:8]}")
        print(f"  B (fork):   {exp_b_id[:8]}")

    elif args.paired:
        latest = find_latest_with_messages(db)
        if not latest:
            print("No experiments with messages found.", file=sys.stderr)
            db.close()
            sys.exit(1)
        control = find_paired_control(db, latest["experiment_id"])
        if not control:
            # Maybe the latest IS the control — check its link
            exp = db.get_experiment(latest["experiment_id"])
            config = json.loads(exp["config_json"]) if exp and exp["config_json"] else {}
            linked = config.get("linked_experiment_id")
            if linked:
                exp_a_id = linked
                exp_b_id = latest["experiment_id"]
            else:
                print(f"No paired control found for {latest['experiment_id'][:8]}.", file=sys.stderr)
                db.close()
                sys.exit(1)
        else:
            exp_a_id = latest["experiment_id"]
            exp_b_id = control["experiment_id"]
        print(f"Comparing experimental → control:")
        print(f"  A (experimental): {exp_a_id[:8]}")
        print(f"  B (control):      {exp_b_id[:8]}")

    elif args.experiment_a and args.experiment_b:
        row_a = resolve_experiment(db, args.experiment_a)
        row_b = resolve_experiment(db, args.experiment_b)
        if not row_a:
            print(f"Experiment not found: {args.experiment_a}", file=sys.stderr)
            db.close()
            sys.exit(1)
        if not row_b:
            print(f"Experiment not found: {args.experiment_b}", file=sys.stderr)
            db.close()
            sys.exit(1)
        exp_a_id = row_a["experiment_id"]
        exp_b_id = row_b["experiment_id"]
    else:
        print("Specify -a and -b, or use --fork-parent or --paired.", file=sys.stderr)
        parser.print_help()
        db.close()
        sys.exit(1)

    # Run diff
    result = diff_experiments(db, exp_a_id, exp_b_id, match_by=args.match_by)
    print_diff(result)

    db.close()


if __name__ == "__main__":
    main()
