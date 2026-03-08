"""SENTINEL Experiment Diffing — side-by-side comparison of experiments.

Compares two experiments across all metrics dimensions, probe results,
and agent-level statistics. Designed for:
  - Experimental vs control arm comparison
  - Fork vs original (path-dependence analysis)
  - Mutation vs baseline (trait impact measurement)
  - Cross-model comparison (same traits, different models)

Output includes per-agent deltas, statistical summaries, and
governance threshold assessments.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Optional

from .db import Database

# Governance thresholds (standard deviations from calibrated baseline)
THRESHOLD_GREEN = 2.0   # < 2 SD = stable
THRESHOLD_YELLOW = 2.0  # >= 2 SD = drifting
THRESHOLD_RED = 3.0     # >= 3 SD = critical


@dataclass
class AgentComparison:
    """Comparison results for a single agent across two experiments."""
    name: str
    agent_id_a: str
    agent_id_b: str
    model_a: str
    model_b: str
    metrics: dict = field(default_factory=dict)
    # metrics = {dimension: {"a": [values], "b": [values], "delta_mean": float, ...}}
    probe_summary: dict = field(default_factory=dict)
    message_stats: dict = field(default_factory=dict)


@dataclass
class DiffResult:
    """Full diff between two experiments."""
    experiment_a: dict   # experiment metadata
    experiment_b: dict
    agent_comparisons: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def diff_experiments(
    db: Database,
    experiment_id_a: str,
    experiment_id_b: str,
    match_by: str = "name",
) -> DiffResult:
    """Compare two experiments across all available data.

    Args:
        db: Database instance
        experiment_id_a: First experiment (typically the baseline/original)
        experiment_id_b: Second experiment (fork, mutation, control, etc.)
        match_by: How to pair agents across experiments.
            "name": Match agents by name (default, works for forks/controls)
            "position": Match agents by position in agent list

    Returns:
        DiffResult with per-agent comparisons and summary statistics
    """
    exp_a = db.get_experiment(experiment_id_a)
    exp_b = db.get_experiment(experiment_id_b)
    if not exp_a:
        raise ValueError(f"Experiment not found: {experiment_id_a}")
    if not exp_b:
        raise ValueError(f"Experiment not found: {experiment_id_b}")

    agents_a = db.get_agents(experiment_id_a)
    agents_b = db.get_agents(experiment_id_b)

    # Pair agents
    pairs = _pair_agents(agents_a, agents_b, match_by)

    result = DiffResult(
        experiment_a=dict(exp_a),
        experiment_b=dict(exp_b),
    )

    for agent_a, agent_b in pairs:
        comparison = _compare_agents(
            db, experiment_id_a, experiment_id_b, agent_a, agent_b,
        )
        result.agent_comparisons.append(comparison)

    # Compute summary across all agents
    result.summary = _compute_summary(result.agent_comparisons)

    return result


def _pair_agents(
    agents_a: list[dict],
    agents_b: list[dict],
    match_by: str,
) -> list[tuple[dict, dict]]:
    """Pair agents from two experiments for comparison."""
    if match_by == "name":
        b_by_name = {a["name"]: a for a in agents_b}
        pairs = []
        for a in agents_a:
            b = b_by_name.get(a["name"])
            if b:
                pairs.append((a, b))
        return pairs
    elif match_by == "position":
        return list(zip(agents_a, agents_b))
    else:
        raise ValueError(f"Unknown match_by: {match_by}")


def _compare_agents(
    db: Database,
    exp_id_a: str,
    exp_id_b: str,
    agent_a: dict,
    agent_b: dict,
) -> AgentComparison:
    """Compare a single agent across two experiments."""
    comparison = AgentComparison(
        name=agent_a["name"],
        agent_id_a=agent_a["agent_id"],
        agent_id_b=agent_b["agent_id"],
        model_a=agent_a["model"],
        model_b=agent_b["model"],
    )

    # Compare metrics
    for dimension in ("vocabulary_drift", "sentiment_trajectory",
                      "semantic_coherence", "persona_adherence"):
        vals_a = _get_metric_values(db, exp_id_a, agent_a["agent_id"], dimension)
        vals_b = _get_metric_values(db, exp_id_b, agent_b["agent_id"], dimension)

        if vals_a or vals_b:
            comparison.metrics[dimension] = _compare_metric_series(vals_a, vals_b)

    # Compare probe results
    probes_a = _get_probe_stats(db, exp_id_a, agent_a["agent_id"])
    probes_b = _get_probe_stats(db, exp_id_b, agent_b["agent_id"])
    if probes_a or probes_b:
        comparison.probe_summary = _compare_probes(probes_a, probes_b)

    # Compare message statistics
    comparison.message_stats = _compare_messages(
        db, exp_id_a, exp_id_b, agent_a["agent_id"], agent_b["agent_id"],
    )

    return comparison


def _get_metric_values(
    db: Database, experiment_id: str, agent_id: str, dimension: str,
) -> list[dict]:
    """Retrieve metric values for an agent/dimension."""
    rows = db.conn.execute(
        "SELECT window_start, window_end, value, details_json FROM metrics "
        "WHERE experiment_id=? AND agent_id=? AND dimension=? "
        "ORDER BY window_start",
        (experiment_id, agent_id, dimension),
    ).fetchall()
    return [dict(r) for r in rows]


def _compare_metric_series(
    vals_a: list[dict], vals_b: list[dict],
) -> dict:
    """Compare two metric time series."""
    a_values = [v["value"] for v in vals_a]
    b_values = [v["value"] for v in vals_b]

    result = {
        "a_values": a_values,
        "b_values": b_values,
        "a_windows": [(v["window_start"], v["window_end"]) for v in vals_a],
        "b_windows": [(v["window_start"], v["window_end"]) for v in vals_b],
    }

    if a_values:
        result["a_mean"] = sum(a_values) / len(a_values)
        result["a_first"] = a_values[0]
        result["a_last"] = a_values[-1]
        result["a_trend"] = a_values[-1] - a_values[0]
    if b_values:
        result["b_mean"] = sum(b_values) / len(b_values)
        result["b_first"] = b_values[0]
        result["b_last"] = b_values[-1]
        result["b_trend"] = b_values[-1] - b_values[0]
    if a_values and b_values:
        result["delta_mean"] = result["b_mean"] - result["a_mean"]
        result["delta_last"] = result["b_last"] - result["a_last"]
        result["delta_trend"] = result["b_trend"] - result["a_trend"]

    return result


def _get_probe_stats(db: Database, experiment_id: str, agent_id: str) -> dict:
    """Get aggregate probe statistics for an agent."""
    rows = db.conn.execute(
        "SELECT category, probe_mode, trigger_reason, drift_score "
        "FROM probes WHERE experiment_id=? AND agent_id=? AND drift_score IS NOT NULL",
        (experiment_id, agent_id),
    ).fetchall()

    if not rows:
        return {}

    by_category = {}
    trigger_counts = {"scheduled": 0, "triggered": 0, "hybrid_scheduled": 0, "hybrid_triggered": 0}

    for r in rows:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r["drift_score"])
        reason = r["trigger_reason"] or "scheduled"
        if reason in trigger_counts:
            trigger_counts[reason] += 1

    stats = {}
    for cat, scores in by_category.items():
        stats[cat] = {
            "count": len(scores),
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "first": scores[0],
            "last": scores[-1],
        }

    return {"by_category": stats, "trigger_counts": trigger_counts}


def _compare_probes(probes_a: dict, probes_b: dict) -> dict:
    """Compare probe statistics between two experiments."""
    result = {"a": probes_a, "b": probes_b, "deltas": {}}

    a_cats = probes_a.get("by_category", {})
    b_cats = probes_b.get("by_category", {})

    all_cats = set(a_cats) | set(b_cats)
    for cat in all_cats:
        a_stats = a_cats.get(cat, {})
        b_stats = b_cats.get(cat, {})
        if a_stats and b_stats:
            result["deltas"][cat] = {
                "delta_mean": b_stats["mean"] - a_stats["mean"],
                "delta_last": b_stats["last"] - a_stats["last"],
            }

    return result


def _compare_messages(
    db: Database,
    exp_id_a: str, exp_id_b: str,
    agent_id_a: str, agent_id_b: str,
) -> dict:
    """Compare message-level statistics between two agents."""
    stats = {}
    for label, exp_id, agent_id in [("a", exp_id_a, agent_id_a), ("b", exp_id_b, agent_id_b)]:
        rows = db.conn.execute(
            "SELECT COUNT(*) as n, AVG(LENGTH(content)) as avg_len, "
            "AVG(inference_ms) as avg_ms, AVG(completion_tokens) as avg_tokens "
            "FROM messages WHERE experiment_id=? AND agent_id=?",
            (exp_id, agent_id),
        ).fetchone()
        stats[label] = {
            "message_count": rows["n"],
            "avg_length": round(rows["avg_len"] or 0, 1),
            "avg_inference_ms": round(rows["avg_ms"] or 0, 1),
            "avg_tokens": round(rows["avg_tokens"] or 0, 1),
        }

    if stats["a"]["avg_length"] and stats["b"]["avg_length"]:
        stats["delta_avg_length"] = round(stats["b"]["avg_length"] - stats["a"]["avg_length"], 1)
        stats["delta_avg_tokens"] = round(stats["b"]["avg_tokens"] - stats["a"]["avg_tokens"], 1)

    return stats


def _compute_summary(comparisons: list[AgentComparison]) -> dict:
    """Compute aggregate summary across all agent comparisons."""
    summary = {
        "agent_count": len(comparisons),
        "dimensions_compared": set(),
        "overall_deltas": {},
    }

    # Aggregate deltas across agents per dimension
    dim_deltas = {}
    for comp in comparisons:
        for dim, data in comp.metrics.items():
            summary["dimensions_compared"].add(dim)
            if "delta_mean" in data:
                dim_deltas.setdefault(dim, []).append(data["delta_mean"])

    for dim, deltas in dim_deltas.items():
        summary["overall_deltas"][dim] = {
            "mean_delta": sum(deltas) / len(deltas),
            "max_delta": max(deltas, key=abs),
            "agents_with_more_drift_in_b": sum(1 for d in deltas if d > 0),
        }

    summary["dimensions_compared"] = sorted(summary["dimensions_compared"])
    return summary


def print_diff(result: DiffResult):
    """Print a formatted experiment diff."""
    exp_a = result.experiment_a
    exp_b = result.experiment_b

    print(f"\n{'='*70}")
    print(f"SENTINEL Experiment Diff")
    print(f"{'='*70}")
    print(f"  A: {exp_a['name']}")
    print(f"     {exp_a['experiment_id'][:8]}  status={exp_a['status']}")
    print(f"  B: {exp_b['name']}")
    print(f"     {exp_b['experiment_id'][:8]}  status={exp_b['status']}")

    # Fork info
    if exp_b.get("forked_from_experiment_id") == exp_a["experiment_id"]:
        print(f"  Relationship: B is a fork of A at turn {exp_b.get('fork_at_turn', '?')}")
    elif exp_a.get("forked_from_experiment_id") == exp_b["experiment_id"]:
        print(f"  Relationship: A is a fork of B at turn {exp_a.get('fork_at_turn', '?')}")

    print()

    # Per-agent comparisons
    for comp in result.agent_comparisons:
        print(f"{'─'*70}")
        model_note = ""
        if comp.model_a != comp.model_b:
            model_note = f"  (A={comp.model_a}, B={comp.model_b})"
        print(f"Agent: {comp.name}{model_note}")
        print(f"{'─'*70}")

        # Metrics comparison
        dim_labels = {
            "vocabulary_drift": ("Vocabulary Drift (JSD)", "higher = more drift"),
            "sentiment_trajectory": ("Sentiment Trajectory", "shift from baseline"),
            "semantic_coherence": ("Semantic Coherence", "higher = more coherent"),
            "persona_adherence": ("Persona Adherence", "1-10 scale"),
        }

        for dim, (label, note) in dim_labels.items():
            if dim not in comp.metrics:
                continue
            data = comp.metrics[dim]
            print(f"\n  {label} ({note}):")

            # Show series side by side
            a_vals = data.get("a_values", [])
            b_vals = data.get("b_values", [])
            a_wins = data.get("a_windows", [])
            b_wins = data.get("b_windows", [])

            max_rows = max(len(a_vals), len(b_vals))
            for i in range(max_rows):
                a_str = ""
                b_str = ""
                if i < len(a_vals):
                    w = a_wins[i]
                    a_str = f"t{w[0]:3d}-{w[1]:3d}: {a_vals[i]:+.4f}"
                if i < len(b_vals):
                    w = b_wins[i]
                    b_str = f"t{w[0]:3d}-{w[1]:3d}: {b_vals[i]:+.4f}"
                print(f"    A: {a_str:<28s}  B: {b_str}")

            # Summary line
            if "delta_mean" in data:
                arrow = "↑" if data["delta_mean"] > 0.01 else "↓" if data["delta_mean"] < -0.01 else "→"
                print(f"    Δ mean: {data['delta_mean']:+.4f} {arrow}  "
                      f"Δ last: {data.get('delta_last', 0):+.4f}  "
                      f"Δ trend: {data.get('delta_trend', 0):+.4f}")

        # Probe comparison
        if comp.probe_summary and comp.probe_summary.get("deltas"):
            print(f"\n  Probe Drift Scores:")
            for cat, deltas in comp.probe_summary["deltas"].items():
                print(f"    {cat}: Δ mean={deltas['delta_mean']:+.4f}  "
                      f"Δ last={deltas['delta_last']:+.4f}")

        # Message stats
        if comp.message_stats:
            ms = comp.message_stats
            a_s = ms.get("a", {})
            b_s = ms.get("b", {})
            print(f"\n  Message Stats:")
            print(f"    Messages:  A={a_s.get('message_count', 0)}  B={b_s.get('message_count', 0)}")
            print(f"    Avg length: A={a_s.get('avg_length', 0):.0f}  B={b_s.get('avg_length', 0):.0f}"
                  f"  (Δ{ms.get('delta_avg_length', 0):+.0f})")
            print(f"    Avg tokens: A={a_s.get('avg_tokens', 0):.0f}  B={b_s.get('avg_tokens', 0):.0f}"
                  f"  (Δ{ms.get('delta_avg_tokens', 0):+.0f})")

    # Overall summary
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"{'='*70}")
    for dim, deltas in result.summary.get("overall_deltas", {}).items():
        label = dim.replace("_", " ").title()
        arrow = "↑" if deltas["mean_delta"] > 0.01 else "↓" if deltas["mean_delta"] < -0.01 else "→"
        print(f"  {label}:  mean Δ={deltas['mean_delta']:+.4f} {arrow}  "
              f"B drifts more for {deltas['agents_with_more_drift_in_b']}/{result.summary['agent_count']} agents")

    print()
