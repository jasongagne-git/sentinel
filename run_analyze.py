#!/usr/bin/env python3
"""SENTINEL Experiment Analyzer — automated pattern detection and finding generation.

Sits between run_diff.py (raw metrics) and run_findings.py (knowledge base).
Detects drift patterns, anomalies, and generates structured analysis reports
and pre-populated finding templates.

Usage:
    python3 run_analyze.py -e a1883355                          # analyze single experiment
    python3 run_analyze.py -a a1883355 -b 1b584cb6              # analyze a comparison
    python3 run_analyze.py -a a1883355 -b 1b584cb6 --finding    # also generate finding template
    python3 run_analyze.py --all                                 # analyze all experiments
"""

import argparse
import json
import math
import signal
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from sentinel.db import Database
from sentinel.diff import diff_experiments, DiffResult
from sentinel.metrics import (
    compute_vocabulary_drift,
    compute_sentiment_trajectory,
)


# ── Pattern detection thresholds ──────────────────────────────────

COLLAPSE_TOKEN_THRESHOLD = 10       # avg tokens below this = collapsed
COLLAPSE_LENGTH_THRESHOLD = 50      # avg chars below this = collapsed
VOCAB_DRIFT_HIGH = 0.15             # JSD above this is notable
VOCAB_DRIFT_EXTREME = 0.50          # JSD above this is severe
SENTIMENT_SHIFT_HIGH = 0.20         # sentiment delta above this is notable
SENTIMENT_FLATLINE = 0.01           # sentiment variance below this = flatlined
CONVERGENCE_RATIO = 0.5             # final/initial vocab ratio below this = converging
LENGTH_COLLAPSE_RATIO = 0.25        # final/initial length ratio below this = collapsing
CROSS_DRIFT_MULTIPLIER = 2.0        # one side drifts Nx more = asymmetric


# ── Data structures ───────────────────────────────────────────────

@dataclass
class Pattern:
    """A detected behavioral pattern."""
    pattern_type: str           # e.g., "agent-collapse", "vocabulary-explosion"
    severity: str               # "info", "notable", "significant", "critical"
    agent: str                  # agent name or "all"
    description: str            # human-readable description
    metrics: dict = field(default_factory=dict)
    turn_range: tuple = None    # (start, end) or None

    def __str__(self):
        sev = {"info": " ", "notable": "*", "significant": "**", "critical": "!!!"}
        marker = sev.get(self.severity, "?")
        return f"  [{marker:>3s}] {self.agent:6s} | {self.pattern_type}: {self.description}"


@dataclass
class AnalysisReport:
    """Full analysis report for one or two experiments."""
    experiment_ids: list[str]
    comparison_type: str        # "single", "cross-model", "paired", "fork"
    patterns: list[Pattern] = field(default_factory=list)
    summary: str = ""
    agent_summaries: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "experiment_ids": self.experiment_ids,
            "comparison_type": self.comparison_type,
            "patterns": [asdict(p) for p in self.patterns],
            "summary": self.summary,
            "agent_summaries": self.agent_summaries,
        }


# ── Single experiment analysis ────────────────────────────────────

def analyze_single(db: Database, experiment_id: str) -> AnalysisReport:
    """Analyze a single experiment for drift patterns."""
    exp = db.get_experiment(experiment_id)
    if not exp:
        print(f"Experiment not found: {experiment_id}", file=sys.stderr)
        sys.exit(1)

    agents = db.get_agents(experiment_id)
    report = AnalysisReport(
        experiment_ids=[experiment_id],
        comparison_type="single",
    )

    for agent in agents:
        agent_name = agent["name"]
        agent_id = agent["agent_id"]

        # Get messages for this agent
        messages = db.get_messages(experiment_id, agent_id=agent_id)
        if not messages:
            report.patterns.append(Pattern(
                pattern_type="no-data",
                severity="info",
                agent=agent_name,
                description="No messages recorded",
            ))
            continue

        # Basic message stats
        lengths = [len(m["content"]) for m in messages]
        tokens = [m.get("completion_tokens", 0) for m in messages]
        avg_length = sum(lengths) / len(lengths) if lengths else 0
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0

        # Detect agent collapse
        detect_collapse(report, agent_name, messages, lengths, tokens)

        # Detect content convergence (shrinking output)
        detect_convergence(report, agent_name, lengths)

        # Compute and analyze vocabulary drift
        vocab_results = compute_vocabulary_drift(db, experiment_id, agent_id, window_size=50)
        detect_vocab_patterns(report, agent_name, vocab_results)

        # Compute and analyze sentiment
        sentiment_results = compute_sentiment_trajectory(db, experiment_id, agent_id, window_size=50)
        detect_sentiment_patterns(report, agent_name, sentiment_results)

        # Agent summary
        report.agent_summaries[agent_name] = {
            "message_count": len(messages),
            "avg_length": round(avg_length, 1),
            "avg_tokens": round(avg_tokens, 1),
            "vocab_drift_final": vocab_results[-1]["jsd"] if vocab_results else None,
            "sentiment_shift_final": sentiment_results[-1]["sentiment_shift"] if sentiment_results else None,
        }

    report.summary = _generate_summary(report)
    return report


def detect_collapse(report, agent_name, messages, lengths, tokens):
    """Detect agent output collapse (messages shrinking to empty)."""
    if not messages:
        return

    # Check if agent went silent (last N messages very short)
    tail_size = min(50, len(messages))
    tail_lengths = lengths[-tail_size:]
    tail_avg = sum(tail_lengths) / len(tail_lengths)
    tail_tokens = tokens[-tail_size:]
    tail_token_avg = sum(tail_tokens) / len(tail_tokens)

    if tail_token_avg < COLLAPSE_TOKEN_THRESHOLD:
        # Find onset turn
        onset = None
        for i, t in enumerate(tokens):
            if all(tk < COLLAPSE_TOKEN_THRESHOLD for tk in tokens[i:min(i+10, len(tokens))]):
                onset = messages[i].get("interaction_turn", i)
                break

        report.patterns.append(Pattern(
            pattern_type="agent-collapse",
            severity="critical",
            agent=agent_name,
            description=f"Output collapsed to avg {tail_token_avg:.0f} tokens (onset ~t{onset})",
            metrics={"tail_avg_tokens": round(tail_token_avg, 1), "onset_turn": onset},
            turn_range=(onset, messages[-1].get("interaction_turn")),
        ))
    elif tail_avg < COLLAPSE_LENGTH_THRESHOLD:
        report.patterns.append(Pattern(
            pattern_type="output-thinning",
            severity="significant",
            agent=agent_name,
            description=f"Output thinned to avg {tail_avg:.0f} chars in last {tail_size} messages",
            metrics={"tail_avg_length": round(tail_avg, 1)},
        ))


def detect_convergence(report, agent_name, lengths):
    """Detect output length convergence (consistent shrinking)."""
    if len(lengths) < 20:
        return

    window = min(50, len(lengths) // 4)
    early_avg = sum(lengths[:window]) / window
    late_avg = sum(lengths[-window:]) / window

    if early_avg > 0 and late_avg / early_avg < LENGTH_COLLAPSE_RATIO:
        report.patterns.append(Pattern(
            pattern_type="length-collapse",
            severity="significant",
            agent=agent_name,
            description=f"Message length dropped {early_avg:.0f} → {late_avg:.0f} chars ({late_avg/early_avg:.0%} of original)",
            metrics={"early_avg": round(early_avg), "late_avg": round(late_avg),
                     "ratio": round(late_avg / early_avg, 3)},
        ))
    elif early_avg > 0 and late_avg / early_avg < CONVERGENCE_RATIO:
        report.patterns.append(Pattern(
            pattern_type="output-shrinking",
            severity="notable",
            agent=agent_name,
            description=f"Message length declined {early_avg:.0f} → {late_avg:.0f} chars",
            metrics={"early_avg": round(early_avg), "late_avg": round(late_avg)},
        ))


def detect_vocab_patterns(report, agent_name, vocab_results):
    """Detect vocabulary drift patterns from JSD series."""
    if len(vocab_results) < 2:
        return

    values = [v["jsd"] for v in vocab_results if v["jsd"] > 0]
    if not values:
        return

    mean_jsd = sum(values) / len(values)
    last_jsd = values[-1]
    max_jsd = max(values)

    if max_jsd > VOCAB_DRIFT_EXTREME:
        report.patterns.append(Pattern(
            pattern_type="vocabulary-explosion",
            severity="critical",
            agent=agent_name,
            description=f"Extreme vocab drift: JSD peaked at {max_jsd:.3f} (mean {mean_jsd:.3f})",
            metrics={"mean_jsd": round(mean_jsd, 4), "max_jsd": round(max_jsd, 4),
                     "last_jsd": round(last_jsd, 4)},
        ))
    elif mean_jsd > VOCAB_DRIFT_HIGH:
        report.patterns.append(Pattern(
            pattern_type="vocabulary-drift",
            severity="notable",
            agent=agent_name,
            description=f"Elevated vocab drift: mean JSD {mean_jsd:.3f}, last {last_jsd:.3f}",
            metrics={"mean_jsd": round(mean_jsd, 4), "last_jsd": round(last_jsd, 4)},
        ))

    # Detect monotonic increase (steadily worsening)
    if len(values) >= 4:
        increasing = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        if increasing / (len(values) - 1) > 0.75:
            report.patterns.append(Pattern(
                pattern_type="vocabulary-drift-accelerating",
                severity="notable",
                agent=agent_name,
                description=f"Vocab drift trending upward ({increasing}/{len(values)-1} windows increasing)",
                metrics={"increasing_windows": increasing, "total_windows": len(values) - 1},
            ))


def detect_sentiment_patterns(report, agent_name, sentiment_results):
    """Detect sentiment trajectory patterns."""
    if len(sentiment_results) < 2:
        return

    shifts = [s["sentiment_shift"] for s in sentiment_results]
    non_zero = [s for s in shifts if s != 0]
    if not non_zero:
        return

    last_shift = shifts[-1]
    mean_shift = sum(non_zero) / len(non_zero)
    variance = sum((s - mean_shift) ** 2 for s in non_zero) / len(non_zero)

    if abs(last_shift) > SENTIMENT_SHIFT_HIGH:
        direction = "positive" if last_shift > 0 else "negative"
        report.patterns.append(Pattern(
            pattern_type="sentiment-shift",
            severity="significant" if abs(last_shift) > 0.4 else "notable",
            agent=agent_name,
            description=f"Sentiment shifted {direction}: {last_shift:+.3f} from baseline",
            metrics={"last_shift": round(last_shift, 4), "mean_shift": round(mean_shift, 4)},
        ))

    if variance < SENTIMENT_FLATLINE and len(non_zero) > 3:
        report.patterns.append(Pattern(
            pattern_type="sentiment-flatline",
            severity="notable",
            agent=agent_name,
            description=f"Sentiment flatlined at {mean_shift:+.3f} (variance {variance:.4f})",
            metrics={"mean": round(mean_shift, 4), "variance": round(variance, 6)},
        ))


# ── Comparison analysis ───────────────────────────────────────────

def analyze_comparison(db: Database, exp_a: str, exp_b: str) -> AnalysisReport:
    """Analyze a comparison between two experiments."""
    diff = diff_experiments(db, exp_a, exp_b)

    # Determine comparison type
    exp_b_data = db.get_experiment(exp_b)
    if exp_b_data and exp_b_data.get("forked_from_experiment_id"):
        if exp_b_data["forked_from_experiment_id"].startswith(exp_a[:8]) or \
           exp_b_data["forked_from_experiment_id"] == exp_a:
            comp_type = "fork"
        else:
            comp_type = "cross-model"
    else:
        # Check if control
        config = json.loads(exp_b_data["config_json"]) if exp_b_data else {}
        if config.get("type") == "control":
            comp_type = "paired"
        elif diff.experiment_a.get("name") == diff.experiment_b.get("name"):
            comp_type = "paired"
        else:
            comp_type = "cross-model"

    report = AnalysisReport(
        experiment_ids=[exp_a, exp_b],
        comparison_type=comp_type,
    )

    for ac in diff.agent_comparisons:
        agent_name = ac.name

        # Analyze each metric dimension
        for dim, comparison in ac.metrics.items():
            if not comparison:
                continue

            a_mean = comparison.get("a_mean")
            b_mean = comparison.get("b_mean")
            a_last = comparison.get("a_last")
            b_last = comparison.get("b_last")
            delta_mean = comparison.get("delta_mean")
            delta_last = comparison.get("delta_last")

            if a_mean is None or b_mean is None:
                continue

            # Detect asymmetric drift
            if a_mean > 0 and b_mean / a_mean > CROSS_DRIFT_MULTIPLIER:
                report.patterns.append(Pattern(
                    pattern_type=f"{dim}-asymmetry",
                    severity="significant",
                    agent=agent_name,
                    description=f"B drifts {b_mean/a_mean:.1f}x more than A ({dim}: A={a_mean:.3f}, B={b_mean:.3f})",
                    metrics={"a_mean": round(a_mean, 4), "b_mean": round(b_mean, 4),
                             "ratio": round(b_mean / a_mean, 2)},
                ))
            elif b_mean > 0 and a_mean / b_mean > CROSS_DRIFT_MULTIPLIER:
                report.patterns.append(Pattern(
                    pattern_type=f"{dim}-asymmetry",
                    severity="significant",
                    agent=agent_name,
                    description=f"A drifts {a_mean/b_mean:.1f}x more than B ({dim}: A={a_mean:.3f}, B={b_mean:.3f})",
                    metrics={"a_mean": round(a_mean, 4), "b_mean": round(b_mean, 4),
                             "ratio": round(a_mean / b_mean, 2)},
                ))

            # Detect extreme deltas
            if delta_last is not None and abs(delta_last) > VOCAB_DRIFT_EXTREME:
                report.patterns.append(Pattern(
                    pattern_type=f"{dim}-divergence",
                    severity="critical",
                    agent=agent_name,
                    description=f"Extreme divergence in {dim}: Δ={delta_last:+.3f}",
                    metrics={"delta_last": round(delta_last, 4), "delta_mean": round(delta_mean, 4) if delta_mean else None},
                ))

        # Detect message length asymmetry
        msg_stats = ac.message_stats
        if msg_stats:
            a_len = msg_stats.get("a", {}).get("avg_length", 0)
            b_len = msg_stats.get("b", {}).get("avg_length", 0)
            if a_len > 0 and b_len > 0:
                ratio = max(a_len, b_len) / min(a_len, b_len)
                if ratio > 3:
                    shorter = "B" if b_len < a_len else "A"
                    report.patterns.append(Pattern(
                        pattern_type="length-asymmetry",
                        severity="significant",
                        agent=agent_name,
                        description=f"{shorter} messages {ratio:.1f}x shorter (A={a_len:.0f}, B={b_len:.0f} chars)",
                        metrics={"a_avg_length": round(a_len), "b_avg_length": round(b_len),
                                 "ratio": round(ratio, 1)},
                    ))

        # Agent summary
        report.agent_summaries[agent_name] = {
            "model_a": ac.model_a,
            "model_b": ac.model_b,
            "metrics": {dim: {
                "a_mean": round(c.get("a_mean", 0), 4),
                "b_mean": round(c.get("b_mean", 0), 4),
                "delta_mean": round(c["delta_mean"], 4) if c.get("delta_mean") is not None else None,
            } for dim, c in ac.metrics.items() if c},
            "message_stats": msg_stats,
        }

    # Cross-agent patterns
    detect_cascade_patterns(report, diff)

    report.summary = _generate_summary(report)
    return report


def detect_cascade_patterns(report: AnalysisReport, diff: DiffResult):
    """Detect patterns that span multiple agents."""
    # Check if all agents show the same direction of drift
    for dim in ("vocabulary_drift", "sentiment_trajectory"):
        deltas = []
        for ac in diff.agent_comparisons:
            comparison = ac.metrics.get(dim, {})
            if comparison and comparison.get("delta_mean") is not None:
                deltas.append((ac.name, comparison["delta_mean"]))

        if len(deltas) >= 2:
            all_positive = all(d > 0 for _, d in deltas)
            all_negative = all(d < 0 for _, d in deltas)
            if all_positive or all_negative:
                direction = "higher" if all_positive else "lower"
                side = "B" if all_positive else "A"
                report.patterns.append(Pattern(
                    pattern_type=f"unanimous-{dim}-direction",
                    severity="notable",
                    agent="all",
                    description=f"All agents show {direction} {dim} in {side} (cascade pattern)",
                    metrics={"deltas": {n: round(d, 4) for n, d in deltas}},
                ))


# ── Summary generation ────────────────────────────────────────────

def _generate_summary(report: AnalysisReport) -> str:
    """Generate a text summary of the analysis."""
    critical = [p for p in report.patterns if p.severity == "critical"]
    significant = [p for p in report.patterns if p.severity == "significant"]
    notable = [p for p in report.patterns if p.severity == "notable"]

    parts = []
    parts.append(f"Analysis of {report.comparison_type} comparison: "
                 f"{len(report.patterns)} patterns detected "
                 f"({len(critical)} critical, {len(significant)} significant, {len(notable)} notable)")

    if critical:
        parts.append("Critical: " + "; ".join(p.description for p in critical))
    if significant:
        parts.append("Significant: " + "; ".join(p.description for p in significant[:3]))

    return ". ".join(parts)


# ── Finding template generation ───────────────────────────────────

def generate_finding_template(report: AnalysisReport, db: Database) -> dict:
    """Generate a pre-populated finding JSON from an analysis report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Build comparison block
    comparison = {"type": report.comparison_type}
    if len(report.experiment_ids) >= 1:
        comparison["experiment_a"] = report.experiment_ids[0]
    if len(report.experiment_ids) >= 2:
        comparison["experiment_b"] = report.experiment_ids[1]

    # Get experiment names for description
    names = []
    for eid in report.experiment_ids:
        exp = db.get_experiment(eid)
        if exp:
            names.append(f"{exp['name']} ({eid[:8]})")
    comparison["description"] = " vs ".join(names)

    # Build metrics from patterns
    metrics = {}
    for p in report.patterns:
        key = f"{p.agent}_{p.pattern_type}".replace("-", "_")
        metrics[key] = {
            "severity": p.severity,
            "description": p.description,
            **p.metrics,
        }

    # Auto-generate tags from patterns
    tags = set()
    tags.add(report.comparison_type)
    for p in report.patterns:
        tags.add(p.pattern_type)
        if p.agent != "all":
            tags.add(p.agent.lower())
        if p.severity in ("critical", "significant"):
            tags.add(p.severity)

    # Determine IP class — comparison findings are open, derived thresholds are proprietary
    ip_class = "open"

    # Build related findings (search existing by shared experiment IDs)
    related = []
    findings_dir = Path("findings")
    if findings_dir.exists():
        import glob as globmod
        for path in globmod.glob(str(findings_dir / "F-*.json")):
            try:
                with open(path) as f:
                    existing = json.load(f)
                comp = existing.get("comparison", {})
                existing_exps = set()
                if comp.get("experiment_a"):
                    existing_exps.add(comp["experiment_a"][:8])
                if comp.get("experiment_b"):
                    existing_exps.add(comp["experiment_b"][:8])
                for eid in report.experiment_ids:
                    if eid[:8] in existing_exps:
                        related.append(existing["id"])
                        break
            except (json.JSONDecodeError, OSError):
                pass

    # Build evidence
    evidence = {}
    if len(report.experiment_ids) == 1:
        evidence["command"] = f"python3 run_analyze.py -e {report.experiment_ids[0][:8]}"
    elif len(report.experiment_ids) == 2:
        evidence["diff_command"] = f"python3 run_diff.py -a {report.experiment_ids[0][:8]} -b {report.experiment_ids[1][:8]}"
        evidence["analyze_command"] = f"python3 run_analyze.py -a {report.experiment_ids[0][:8]} -b {report.experiment_ids[1][:8]}"

    return {
        "id": "F-XXXX",
        "title": "[TODO: one-line title summarizing the key insight]",
        "created": now,
        "updated": now,
        "author": "jason",
        "ip_class": ip_class,
        "comparison": comparison,
        "metrics": metrics,
        "finding": report.summary,
        "implications": ["[TODO: what does this mean for governance/monitoring?]"],
        "tags": sorted(tags),
        "related": sorted(set(related)),
        "evidence": evidence,
    }


# ── Auto-finding generation ──────────────────────────────────────

# Maps pattern types to implication templates
IMPLICATION_MAP = {
    "agent-collapse": [
        "Agent collapse is a [redacted]",
        "Collapsed agents [redacted] — probe-only monitoring gives false positives",
    ],
    "vocabulary-explosion": [
        "Extreme [redacted] behavioral change, not just style drift",
        "[redacted] should trigger at lower thresholds for early warning",
    ],
    "vocabulary-drift": [
        "Elevated vocabulary drift may indicate [redacted]",
    ],
    "vocabulary-drift-accelerating": [
        "Accelerating drift suggests [redacted] should occur before saturation",
    ],
    "sentiment-shift": [
        "Sentiment changes may indicate [redacted] vocabulary is stable",
    ],
    "sentiment-flatline": [
        "Sentiment flatline suggests agent has converged to a [redacted]",
    ],
    "length-collapse": [
        "Output length collapse correlates with [redacted]",
    ],
    "length-asymmetry": [
        "Large length differences between conditions suggest one environment is [redacted]",
    ],
    "output-thinning": [
        "Gradual output thinning may [redacted] — monitor as an early warning signal",
    ],
    "unanimous-vocabulary_drift-direction": [
        "Uniform drift direction across all agents suggests a [redacted] variation",
    ],
    "unanimous-sentiment_trajectory-direction": [
        "Uniform sentiment shift across all agents suggests [redacted] dynamics",
    ],
}

# Maps comparison types to additional context implications
COMPARISON_IMPLICATIONS = {
    "fork": [
        "[redacted] — [redacted] through the group",
        "[redacted]: [redacted] others more than itself",
    ],
    "paired": [
        "Multi-agent [redacted] from single-agent context drift",
        "[redacted] of drift ([redacted] convergence)",
    ],
    "cross-model": [
        "Drift baselines must be [redacted], not just per-model",
        "Model selection is a [redacted], not just an infrastructure choice",
    ],
}


def _auto_title(report: AnalysisReport) -> str:
    """Generate a descriptive title from the most important patterns."""
    critical = [p for p in report.patterns if p.severity == "critical"]
    significant = [p for p in report.patterns if p.severity == "significant"]

    # Build title from highest-severity patterns
    parts = []
    if critical:
        agents = sorted(set(p.agent for p in critical if p.agent != "all"))
        types = sorted(set(p.pattern_type for p in critical))
        if "agent-collapse" in types:
            collapsed = [p.agent for p in critical if p.pattern_type == "agent-collapse"]
            parts.append(f"{', '.join(collapsed)} collapse detected")
        elif "vocabulary-explosion" in types:
            parts.append(f"extreme vocabulary drift in {', '.join(agents)}")
        elif any("divergence" in t for t in types):
            affected = ", ".join(agents) if agents else "all agents"
            parts.append(f"extreme divergence across {affected}")
        else:
            parts.append(f"critical drift in {', '.join(agents)}")

    if significant and not parts:
        types = sorted(set(p.pattern_type for p in significant))
        if any("asymmetry" in t for t in types):
            parts.append("asymmetric drift patterns detected")
        elif any("divergence" in t for t in types):
            parts.append("significant divergence between conditions")
        elif any("length" in t for t in types):
            parts.append("significant output length changes")
        else:
            parts.append(f"significant {types[0].replace('_', ' ')} detected")

    if not parts:
        notable = [p for p in report.patterns if p.severity == "notable"]
        if notable:
            parts.append(f"{len(notable)} notable drift patterns")
        else:
            parts.append("no significant drift detected")

    # Add comparison context
    comp_labels = {
        "fork": "fork comparison",
        "paired": "paired comparison",
        "cross-model": "cross-model comparison",
        "single": "single experiment",
    }
    context = comp_labels.get(report.comparison_type, report.comparison_type)

    title = f"{'; '.join(parts)} ({context})"
    return title[0].upper() + title[1:]


def _auto_finding_text(report: AnalysisReport) -> str:
    """Generate a comprehensive finding paragraph from patterns."""
    sentences = []

    critical = [p for p in report.patterns if p.severity == "critical"]
    significant = [p for p in report.patterns if p.severity == "significant"]
    notable = [p for p in report.patterns if p.severity == "notable"]

    if critical:
        for p in critical:
            sentences.append(p.description + ".")
    if significant:
        for p in significant[:5]:  # cap at 5 to keep readable
            sentences.append(p.description + ".")
    if notable:
        # Summarize notables rather than listing each
        notable_types = sorted(set(p.pattern_type for p in notable))
        sentences.append(
            f"Additional notable patterns: {', '.join(notable_types)} "
            f"({len(notable)} total)."
        )

    # Add cross-agent observations
    agents_affected = set(p.agent for p in report.patterns if p.agent != "all")
    if len(agents_affected) > 1 and any(p.agent == "all" for p in report.patterns):
        sentences.append("Effects span all agents, indicating systemic rather than individual drift.")

    return " ".join(sentences)


def _auto_implications(report: AnalysisReport) -> list[str]:
    """Generate implications from pattern types and comparison context."""
    implications = []
    seen = set()

    # Add pattern-specific implications
    pattern_types = set(p.pattern_type for p in report.patterns
                        if p.severity in ("critical", "significant"))
    for ptype in pattern_types:
        for imp in IMPLICATION_MAP.get(ptype, []):
            if imp not in seen:
                implications.append(imp)
                seen.add(imp)

    # Add comparison-type implications
    for imp in COMPARISON_IMPLICATIONS.get(report.comparison_type, []):
        if imp not in seen:
            implications.append(imp)
            seen.add(imp)

    # Add probe-conversation dissociation if collapse detected
    if any(p.pattern_type == "agent-collapse" for p in report.patterns):
        imp = "Probe-conversation dissociation must be monitored — agents may appear healthy under probing while conversationally collapsed"
        if imp not in seen:
            implications.append(imp)

    if not implications:
        implications.append("No significant governance implications detected in this comparison")

    return implications


def generate_auto_finding(report: AnalysisReport, db: Database) -> dict:
    """Generate a complete finding JSON from an analysis report (no TODOs)."""
    template = generate_finding_template(report, db)

    # Replace template placeholders with auto-generated content
    template["title"] = _auto_title(report)
    template["finding"] = _auto_finding_text(report)
    template["implications"] = _auto_implications(report)
    template["author"] = "sentinel-auto"  # distinguish from manual findings

    return template


# ── Output formatting ─────────────────────────────────────────────

def print_report(report: AnalysisReport):
    """Print a formatted analysis report."""
    print(f"\n{'='*70}")
    print(f"SENTINEL Analysis Report")
    print(f"{'='*70}")
    print(f"  Type: {report.comparison_type}")
    print(f"  Experiments: {', '.join(e[:8] for e in report.experiment_ids)}")
    print(f"  Patterns detected: {len(report.patterns)}")

    # Group by severity
    for severity in ("critical", "significant", "notable", "info"):
        patterns = [p for p in report.patterns if p.severity == severity]
        if patterns:
            print(f"\n  {severity.upper()} ({len(patterns)}):")
            for p in patterns:
                print(p)

    # Agent summaries
    if report.agent_summaries:
        print(f"\n  Agent Summaries:")
        for name, summary in report.agent_summaries.items():
            print(f"    {name}:")
            for k, v in summary.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        print(f"      {k}.{sk}: {sv}")
                else:
                    print(f"      {k}: {v}")

    print(f"\n  Summary: {report.summary}")
    print()


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SENTINEL Experiment Analyzer")
    parser.add_argument("-e", "--experiment", help="Analyze a single experiment (ID prefix)")
    parser.add_argument("-a", "--exp-a", help="Experiment A for comparison")
    parser.add_argument("-b", "--exp-b", help="Experiment B for comparison")
    parser.add_argument("--all", action="store_true", help="Analyze all experiments")
    parser.add_argument("--finding", action="store_true",
                        help="Generate a finding template JSON (requires manual editing)")
    parser.add_argument("--auto-finding", action="store_true",
                        help="Generate a complete finding automatically (no manual steps)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--db", default="experiments/sentinel.db")
    args = parser.parse_args()

    db = Database(args.db)

    def resolve(prefix):
        row = db.conn.execute(
            "SELECT experiment_id FROM experiments WHERE experiment_id LIKE ?",
            (prefix + "%",),
        ).fetchone()
        if not row:
            print(f"Experiment not found: {prefix}", file=sys.stderr)
            sys.exit(1)
        return row[0]

    reports = []

    if args.experiment:
        exp_id = resolve(args.experiment)
        reports.append(analyze_single(db, exp_id))

    elif args.exp_a and args.exp_b:
        exp_a = resolve(args.exp_a)
        exp_b = resolve(args.exp_b)
        reports.append(analyze_comparison(db, exp_a, exp_b))

    elif args.all:
        rows = db.conn.execute(
            "SELECT experiment_id FROM experiments WHERE status='completed' "
            "AND experiment_id IN (SELECT DISTINCT experiment_id FROM messages) "
            "ORDER BY created_at"
        ).fetchall()
        for row in rows:
            reports.append(analyze_single(db, row[0]))

    else:
        parser.print_help()
        sys.exit(1)

    for report in reports:
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print_report(report)

        if args.finding or args.auto_finding:
            if args.auto_finding:
                template = generate_auto_finding(report, db)
            else:
                template = generate_finding_template(report, db)
            # Write to file
            from run_findings import next_id, slugify, FINDINGS_DIR
            FINDINGS_DIR.mkdir(exist_ok=True)
            fid = next_id()
            template["id"] = fid
            slug = slugify(template.get("title", report.summary)[:60])
            path = FINDINGS_DIR / f"{fid}_{slug}.json"
            with open(path, "w") as f:
                json.dump(template, f, indent=2)
                f.write("\n")
            if args.auto_finding:
                print(f"Auto-finding written to: {path}")
            else:
                print(f"Finding template written to: {path}")
                print(f"Edit to add title, finding text, and implications.")

    db.close()


def auto_analyze_and_save(
    db: Database,
    experiment_ids: list[str],
    auto_finding: bool = True,
    quiet: bool = False,
) -> AnalysisReport:
    """Programmatic API: analyze experiment(s) and optionally save a finding.

    Args:
        db: Database instance
        experiment_ids: 1 or 2 experiment IDs (single or comparison)
        auto_finding: if True, auto-generate and save a complete finding
        quiet: suppress stdout output

    Returns:
        AnalysisReport
    """
    if len(experiment_ids) == 1:
        report = analyze_single(db, experiment_ids[0])
    elif len(experiment_ids) == 2:
        report = analyze_comparison(db, experiment_ids[0], experiment_ids[1])
    else:
        raise ValueError("Expected 1 or 2 experiment IDs")

    if not quiet:
        print_report(report)

    if auto_finding and report.patterns:
        finding = generate_auto_finding(report, db)
        from run_findings import next_id, slugify, FINDINGS_DIR
        FINDINGS_DIR.mkdir(exist_ok=True)
        fid = next_id()
        finding["id"] = fid
        slug = slugify(finding.get("title", "analysis")[:60])
        path = FINDINGS_DIR / f"{fid}_{slug}.json"
        with open(path, "w") as f:
            json.dump(finding, f, indent=2)
            f.write("\n")
        if not quiet:
            print(f"Auto-finding saved: {path}")

    return report


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    main()
