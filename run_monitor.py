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

"""Live-monitor a running or completed SENTINEL experiment.

Usage:
    # Monitor the most recent experiment
    python3 run_monitor.py

    # Monitor a specific experiment
    python3 run_monitor.py -e 3f9dd450

    # Faster refresh rate
    python3 run_monitor.py --refresh 1

Controls:
    q       Quit
    +/-     Increase/decrease refresh rate
"""

import argparse
import sys

from sentinel.db import Database
from sentinel.monitor import run_monitor


def main():
    parser = argparse.ArgumentParser(description="Live-monitor a SENTINEL experiment")
    parser.add_argument("-e", "--experiment-id", help="Experiment ID (prefix match)")
    parser.add_argument("--refresh", type=float, default=3.0,
                        help="Refresh interval in seconds (default 3)")
    parser.add_argument("--db", default="experiments/sentinel.db")
    args = parser.parse_args()

    db = Database(args.db)

    if args.experiment_id:
        row = db.conn.execute(
            "SELECT experiment_id FROM experiments WHERE experiment_id LIKE ?",
            (args.experiment_id + "%",),
        ).fetchone()
        if not row:
            print(f"Experiment not found: {args.experiment_id}", file=sys.stderr)
            db.close()
            sys.exit(1)
        experiment_id = row["experiment_id"]
    else:
        row = db.conn.execute(
            "SELECT experiment_id FROM experiments "
            "WHERE experiment_id IN (SELECT DISTINCT experiment_id FROM messages) "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            print("No experiments with messages found.", file=sys.stderr)
            db.close()
            sys.exit(1)
        experiment_id = row["experiment_id"]

    print(f"Monitoring experiment {experiment_id[:8]}...")
    print(f"Press 'q' to quit, '+'/'-' to adjust refresh rate.\n")

    try:
        run_monitor(db, experiment_id, refresh_interval=args.refresh)
    finally:
        db.close()


if __name__ == "__main__":
    main()
