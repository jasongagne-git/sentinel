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

"""SENTINEL Findings Knowledge Base — structured store for experiment analysis.

Usage:
    python3 run_findings.py --list                        # list all findings
    python3 run_findings.py --show [ref]                 # show one finding
    python3 run_findings.py --search "beck collapse"      # search titles, text, tags
    python3 run_findings.py --tags                         # list all tags with counts
    python3 run_findings.py --related [ref]              # show related findings
    python3 run_findings.py --add --title "..." --type cross-model \\
        --exp-a abc123 --exp-b def456 --tags "tag1,tag2"  # create new finding
"""

import argparse
import glob
import json
import os
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

FINDINGS_DIR = Path("findings")


def load_all() -> list[dict]:
    """Load all finding JSON files, sorted by ID."""
    findings = []
    for path in sorted(glob.glob(str(FINDINGS_DIR / "F-*.json"))):
        try:
            with open(path) as f:
                data = json.load(f)
            data["_path"] = path
            findings.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: skipping {path}: {exc}", file=sys.stderr)
    return findings


def load_one(finding_id: str) -> dict | None:
    """Load a single finding by ID (e.g., '[ref]')."""
    for f in load_all():
        if f.get("id") == finding_id:
            return f
    return None


def next_id() -> str:
    """Generate the next sequential finding ID."""
    existing = load_all()
    if not existing:
        return "[ref]"
    last = max(f.get("id", "[ref]") for f in existing)
    num = int(last.split("-")[1]) + 1
    return f"F-{num:04d}"


def slugify(title: str) -> str:
    """Convert title to filename slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:60]


# ── Display ────────────────────────────────────────────────────────

def fmt_list(findings: list[dict]):
    """Print findings as a table."""
    if not findings:
        print("No findings.")
        return
    print(f"{'ID':<8s} {'Date':<12s} {'Type':<16s} {'IP':<6s} {'Title'}")
    print(f"{'─'*8} {'─'*12} {'─'*16} {'─'*6} {'─'*44}")
    for f in findings:
        fid = f.get("id", "?")
        date = f.get("created", "")[:10]
        ctype = f.get("comparison", {}).get("type", "?")
        ip = f.get("ip_class", "?")
        title = f.get("title", "untitled")
        if len(title) > 60:
            title = title[:57] + "..."
        print(f"{fid:<8s} {date:<12s} {ctype:<16s} {ip:<6s} {title}")
    print(f"\n{len(findings)} finding(s)")


def fmt_show(f: dict):
    """Pretty-print a single finding."""
    print(f"\n{'='*70}")
    print(f"  {f.get('id', '?')}: {f.get('title', 'untitled')}")
    print(f"{'='*70}")

    print(f"\n  Created:  {f.get('created', '?')}")
    if f.get("updated") and f["updated"] != f.get("created"):
        print(f"  Updated:  {f['updated']}")
    print(f"  Author:   {f.get('author', '?')}")
    print(f"  IP class: {f.get('ip_class', '?')}")

    comp = f.get("comparison", {})
    if comp:
        print(f"\n  Comparison:")
        print(f"    Type: {comp.get('type', '?')}")
        if comp.get("experiment_a"):
            print(f"    Exp A: {comp['experiment_a']}")
        if comp.get("experiment_b"):
            print(f"    Exp B: {comp['experiment_b']}")
        if comp.get("experiments"):
            print(f"    Experiments: {', '.join(comp['experiments'])}")
        if comp.get("fork_at_turn"):
            print(f"    Fork at: t{comp['fork_at_turn']}")
        if comp.get("mutation"):
            print(f"    Mutation: {comp['mutation']}")
        if comp.get("description"):
            print(f"    {comp['description']}")

    metrics = f.get("metrics", {})
    if metrics:
        print(f"\n  Metrics:")
        for dim, data in metrics.items():
            print(f"    {dim}:")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"      {k}: {v}")
            else:
                print(f"      {data}")

    finding_text = f.get("finding", "")
    if finding_text:
        print(f"\n  Finding:")
        for line in textwrap.wrap(finding_text, width=66):
            print(f"    {line}")

    implications = f.get("implications", [])
    if implications:
        print(f"\n  Implications:")
        for imp in implications:
            print(f"    - {imp}")

    tags = f.get("tags", [])
    if tags:
        print(f"\n  Tags: {', '.join(tags)}")

    related = f.get("related", [])
    if related:
        print(f"  Related: {', '.join(related)}")

    evidence = f.get("evidence", {})
    if evidence:
        print(f"\n  Evidence:")
        for k, v in evidence.items():
            print(f"    {k}: {v}")

    print()


def fmt_tags(findings: list[dict]):
    """Print all tags with counts."""
    tag_counts: dict[str, int] = {}
    for f in findings:
        for tag in f.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    if not tag_counts:
        print("No tags.")
        return
    print(f"{'Count':>5s}  Tag")
    print(f"{'─'*5}  {'─'*30}")
    for tag, count in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{count:5d}  {tag}")


# ── Search ─────────────────────────────────────────────────────────

def search(findings: list[dict], query: str) -> list[dict]:
    """Case-insensitive search across title, finding text, tags, comparison."""
    q = query.lower()
    results = []
    for f in findings:
        searchable = " ".join([
            f.get("title", ""),
            f.get("finding", ""),
            " ".join(f.get("tags", [])),
            f.get("comparison", {}).get("type", ""),
            f.get("comparison", {}).get("description", ""),
            " ".join(f.get("implications", [])),
        ]).lower()
        if q in searchable:
            results.append(f)
    return results


def find_related(findings: list[dict], finding_id: str) -> list[dict]:
    """Find findings related by explicit links or shared tags."""
    target = None
    for f in findings:
        if f.get("id") == finding_id:
            target = f
            break
    if not target:
        return []

    target_tags = set(target.get("tags", []))
    explicit = set(target.get("related", []))
    results = []

    for f in findings:
        if f.get("id") == finding_id:
            continue
        # Explicitly related
        if f.get("id") in explicit:
            results.append(f)
            continue
        # This finding lists the target as related
        if finding_id in f.get("related", []):
            results.append(f)
            continue
        # Shared tags (at least 2)
        shared = target_tags & set(f.get("tags", []))
        if len(shared) >= 2:
            results.append(f)

    return results


# ── Add ────────────────────────────────────────────────────────────

def create_finding(
    title: str,
    comp_type: str,
    exp_a: str = None,
    exp_b: str = None,
    tags: list[str] = None,
    ip_class: str = "open",
) -> str:
    """Create a new finding JSON file with a template. Returns the file path."""
    fid = next_id()
    slug = slugify(title)
    filename = f"{fid}_{slug}.json"
    path = FINDINGS_DIR / filename

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    comparison = {"type": comp_type}
    if exp_a:
        comparison["experiment_a"] = exp_a
    if exp_b:
        comparison["experiment_b"] = exp_b
    comparison["description"] = ""

    finding = {
        "id": fid,
        "title": title,
        "created": now,
        "updated": now,
        "author": "sentinel-auto",
        "ip_class": ip_class,
        "comparison": comparison,
        "metrics": {},
        "finding": "",
        "implications": [],
        "tags": tags or [],
        "related": [],
        "evidence": {
            "diff_command": "",
            "batch_id": "",
        },
    }

    with open(path, "w") as f:
        json.dump(finding, f, indent=2)
        f.write("\n")

    return str(path)


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SENTINEL Findings Knowledge Base")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List all findings")
    group.add_argument("--show", metavar="ID", help="Show a finding by ID")
    group.add_argument("--search", metavar="QUERY", help="Search findings")
    group.add_argument("--tags", action="store_true", help="List all tags with counts")
    group.add_argument("--related", metavar="ID", help="Show findings related to ID")
    group.add_argument("--add", action="store_true", help="Create a new finding")

    # --add options
    parser.add_argument("--title", help="Finding title (for --add)")
    parser.add_argument("--type", dest="comp_type", help="Comparison type (for --add)")
    parser.add_argument("--exp-a", help="Experiment A ID (for --add)")
    parser.add_argument("--exp-b", help="Experiment B ID (for --add)")
    parser.add_argument("--tag", dest="tag_list", help="Comma-separated tags (for --add)")
    parser.add_argument("--ip", dest="ip_class", default="open",
                        help="IP classification (for --add, default: open)")

    args = parser.parse_args()

    FINDINGS_DIR.mkdir(exist_ok=True)

    if args.list:
        fmt_list(load_all())

    elif args.show:
        f = load_one(args.show)
        if f:
            fmt_show(f)
        else:
            print(f"Finding not found: {args.show}", file=sys.stderr)
            sys.exit(1)

    elif args.search is not None:
        results = search(load_all(), args.search)
        fmt_list(results)

    elif args.tags:
        fmt_tags(load_all())

    elif args.related:
        target = load_one(args.related)
        if not target:
            print(f"Finding not found: {args.related}", file=sys.stderr)
            sys.exit(1)
        print(f"Findings related to {args.related}: {target.get('title', '')}\n")
        results = find_related(load_all(), args.related)
        fmt_list(results)

    elif args.add:
        if not args.title or not args.comp_type:
            print("--add requires --title and --type", file=sys.stderr)
            sys.exit(1)
        tags = [t.strip() for t in args.tag_list.split(",")] if args.tag_list else []
        path = create_finding(
            title=args.title,
            comp_type=args.comp_type,
            exp_a=args.exp_a,
            exp_b=args.exp_b,
            tags=tags,
            ip_class=args.ip_class,
        )
        print(f"Created: {path}")
        print(f"Edit the file to add metrics, finding text, and implications.")


if __name__ == "__main__":
    main()
