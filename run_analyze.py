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


# ── Pattern detection config (loaded from gitignored config) ─────

def _load_detection_config():
    """Load detection patterns config. Falls back to minimal defaults."""
    config_path = Path(__file__).parent / "config" / "detection_patterns.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None

_DETECTION_CONFIG = _load_detection_config()

def _threshold(key, fallback):
    """Get a threshold from config, or use fallback."""
    if _DETECTION_CONFIG:
        return _DETECTION_CONFIG.get("thresholds", {}).get(key, fallback)
    return fallback

COLLAPSE_TOKEN_THRESHOLD = _threshold("collapse_token", 10)
COLLAPSE_LENGTH_THRESHOLD = _threshold("collapse_length", 50)
VOCAB_DRIFT_HIGH = _threshold("vocab_drift_high", 0.15)
VOCAB_DRIFT_EXTREME = _threshold("vocab_drift_extreme", 0.50)
SENTIMENT_SHIFT_HIGH = _threshold("sentiment_shift_high", 0.20)
SENTIMENT_FLATLINE = _threshold("sentiment_flatline", 0.01)
CONVERGENCE_RATIO = _threshold("convergence_ratio", 0.5)
LENGTH_COLLAPSE_RATIO = _threshold("length_collapse_ratio", 0.25)
CROSS_DRIFT_MULTIPLIER = _threshold("cross_drift_multiplier", 2.0)
HOLLOW_VERBOSITY_LENGTH_GROWTH = _threshold("hollow_verbosity_length_growth", 1.2)
HOLLOW_VERBOSITY_VOCAB_SHRINK = _threshold("hollow_verbosity_vocab_shrink", 0.75)
PROBE_CONTAMINATION_THRESHOLD = _threshold("probe_contamination", 0.10)
DISSOCIATION_GAP_THRESHOLD = _threshold("dissociation_gap", 0.10)
CONTENT_REPETITION_THRESHOLD = _threshold("content_repetition", 0.70)
CORRELATION_REVERSAL_THRESHOLD = _threshold("correlation_reversal", 0.50)


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

        # Detect hollow verbosity (length grows while vocab shrinks)
        detect_hollow_verbosity(report, agent_name, messages, vocab_results)

        # Detect probe contamination in messages
        detect_probe_contamination(report, agent_name, messages)

        # Detect probe-conversation dissociation (shadow vs injected gap)
        detect_dissociation_gap(report, db, experiment_id, agent_id, agent_name)

        # Detect formulaic/repetitive output
        detect_content_repetition(report, agent_name, messages)

        # Agent summary
        report.agent_summaries[agent_name] = {
            "message_count": len(messages),
            "avg_length": round(avg_length, 1),
            "avg_tokens": round(avg_tokens, 1),
            "vocab_drift_final": vocab_results[-1]["jsd"] if vocab_results else None,
            "sentiment_shift_final": sentiment_results[-1]["sentiment_shift"] if sentiment_results else None,
        }

    # Experiment-level detectors (run once, not per-agent)
    detect_context_saturation(report, agents, db, experiment_id)
    detect_cascade_propagation(report, agents, db, experiment_id)
    detect_correlation_reversal(report, agents, db, experiment_id)

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


def detect_hollow_verbosity(report, agent_name, messages, vocab_results):
    """Detect hollow verbosity: messages grow longer while vocabulary shrinks."""
    if len(messages) < 40 or len(vocab_results) < 2:
        return

    # Compare early vs late message lengths
    window = min(50, len(messages) // 4)
    early_lengths = [len(m["content"]) for m in messages[:window]]
    late_lengths = [len(m["content"]) for m in messages[-window:]]
    early_avg = sum(early_lengths) / len(early_lengths)
    late_avg = sum(late_lengths) / len(late_lengths)

    if early_avg == 0:
        return

    length_ratio = late_avg / early_avg

    # Compare early vs late unique vocabulary
    early_words = set()
    for m in messages[:window]:
        early_words.update(m["content"].lower().split())
    late_words = set()
    for m in messages[-window:]:
        late_words.update(m["content"].lower().split())

    if not early_words:
        return

    vocab_ratio = len(late_words) / len(early_words)

    if length_ratio >= HOLLOW_VERBOSITY_LENGTH_GROWTH and vocab_ratio <= HOLLOW_VERBOSITY_VOCAB_SHRINK:
        severity = "significant" if length_ratio > 1.5 or vocab_ratio < 0.5 else "notable"
        report.patterns.append(Pattern(
            pattern_type="hollow-verbosity",
            severity=severity,
            agent=agent_name,
            description=(
                f"Messages grew {length_ratio:.1f}x longer while vocabulary shrank to "
                f"{vocab_ratio:.0%} — hollow verbosity detected"
            ),
            metrics={
                "length_ratio": round(length_ratio, 3),
                "vocab_ratio": round(vocab_ratio, 3),
                "early_avg_length": round(early_avg),
                "late_avg_length": round(late_avg),
                "early_unique_words": len(early_words),
                "late_unique_words": len(late_words),
            },
        ))


def detect_probe_contamination(report, agent_name, messages):
    """Detect probe identity contamination in agent messages."""
    if not messages:
        return

    # Contamination markers loaded from config
    markers = ["[Probe Response]", "SENTINEL Probe"]
    if _DETECTION_CONFIG:
        markers = _DETECTION_CONFIG.get("contamination_markers", markers)

    contaminated = 0
    pure_probe = 0
    hybrid = 0

    for m in messages:
        content = m["content"]
        hit = any(marker in content for marker in markers)

        if hit:
            contaminated += 1
            stripped = content.strip()
            if any(stripped.startswith(marker) for marker in markers):
                pure_probe += 1
            else:
                hybrid += 1

    if not messages:
        return

    pct = contaminated / len(messages)
    if pct >= PROBE_CONTAMINATION_THRESHOLD:
        severity = "critical" if pct > 0.50 else "significant" if pct > 0.25 else "notable"
        report.patterns.append(Pattern(
            pattern_type="probe-contamination",
            severity=severity,
            agent=agent_name,
            description=(
                f"{contaminated}/{len(messages)} messages ({pct:.1%}) contaminated "
                f"with probe identity ({pure_probe} pure, {hybrid} hybrid)"
            ),
            metrics={
                "contaminated": contaminated,
                "total_msgs": len(messages),
                "pct": round(pct * 100, 1),
                "pure_probe": pure_probe,
                "hybrid": hybrid,
            },
        ))


def detect_dissociation_gap(report, db, experiment_id, agent_id, agent_name):
    """Detect probe-conversation dissociation (shadow vs injected drift score gap)."""
    # Get late-stage probe results for both modes
    rows = db.conn.execute(
        "SELECT probe_mode, drift_score FROM probes "
        "WHERE experiment_id=? AND agent_id=? AND drift_score IS NOT NULL "
        "ORDER BY at_turn ASC",
        (experiment_id, agent_id),
    ).fetchall()

    if not rows:
        return

    shadow_scores = [r["drift_score"] for r in rows if r["probe_mode"] == "shadow"]
    injected_scores = [r["drift_score"] for r in rows if r["probe_mode"] == "injected"]

    if len(shadow_scores) < 3 or len(injected_scores) < 3:
        return

    # Use late-stage scores (last third)
    shadow_late = shadow_scores[-(len(shadow_scores) // 3):]
    injected_late = injected_scores[-(len(injected_scores) // 3):]

    shadow_mean = sum(shadow_late) / len(shadow_late)
    injected_mean = sum(injected_late) / len(injected_late)
    gap = abs(shadow_mean - injected_mean)

    if gap >= DISSOCIATION_GAP_THRESHOLD:
        higher = "shadow" if shadow_mean > injected_mean else "injected"
        severity = "significant" if gap > 0.25 else "notable"
        report.patterns.append(Pattern(
            pattern_type="probe-dissociation",
            severity=severity,
            agent=agent_name,
            description=(
                f"Shadow/injected drift score gap: {gap:.3f} "
                f"(shadow={shadow_mean:.3f}, injected={injected_mean:.3f}) — "
                f"{higher} probes show more drift"
            ),
            metrics={
                "gap": round(gap, 4),
                "shadow_mean": round(shadow_mean, 4),
                "injected_mean": round(injected_mean, 4),
                "shadow_n": len(shadow_late),
                "injected_n": len(injected_late),
            },
        ))


def detect_content_repetition(report, agent_name, messages):
    """Detect formulaic output — high n-gram overlap between early and late messages."""
    if len(messages) < 100:
        return

    def get_ngrams(text, n=4):
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    # Build early n-gram set from first 50 messages
    early_ngrams = set()
    for m in messages[:50]:
        early_ngrams |= get_ngrams(m["content"])

    if not early_ngrams:
        return

    # Check late messages (last 100) for overlap
    late_ngrams = set()
    for m in messages[-100:]:
        late_ngrams |= get_ngrams(m["content"])

    if not late_ngrams:
        return

    overlap = len(early_ngrams & late_ngrams) / len(late_ngrams) if late_ngrams else 0

    if overlap >= CONTENT_REPETITION_THRESHOLD:
        severity = "significant" if overlap > 0.85 else "notable"
        report.patterns.append(Pattern(
            pattern_type="formulaic-output",
            severity=severity,
            agent=agent_name,
            description=(
                f"{overlap:.0%} of late 4-grams appeared in early messages — "
                f"output has become formulaic"
            ),
            metrics={
                "ngram_overlap": round(overlap, 3),
                "early_ngrams": len(early_ngrams),
                "late_ngrams": len(late_ngrams),
            },
        ))


def _find_collapse_turns(agents, db, experiment_id):
    """Find collapse onset turn for each agent. Returns dict of {name: turn} for collapsed agents."""
    collapse_turns = {}
    for agent in agents:
        agent_name = agent["name"]
        messages = db.get_messages(experiment_id, agent_id=agent["agent_id"])
        if not messages or len(messages) < 10:
            continue

        for i in range(len(messages) - 5):
            window = messages[i:i+5]
            if all(len(m["content"].strip()) < 10 for m in window):
                collapse_turns[agent_name] = messages[i].get("interaction_turn", i)
                break
    return collapse_turns


def _find_thinning_after(agent_name, db, experiment_id, agent_id, after_turn):
    """Check if agent's output thins after a given turn, even if it doesn't fully collapse.

    Returns (pre_avg_length, post_avg_length, ratio) or None.
    """
    messages = db.get_messages(experiment_id, agent_id=agent_id)
    if not messages:
        return None

    pre = [len(m["content"]) for m in messages
           if m.get("interaction_turn", 0) < after_turn]
    post = [len(m["content"]) for m in messages
            if m.get("interaction_turn", 0) >= after_turn]

    if len(pre) < 10 or len(post) < 10:
        return None

    pre_avg = sum(pre) / len(pre)
    post_avg = sum(post) / len(post)
    if pre_avg == 0:
        return None

    return (round(pre_avg), round(post_avg), round(post_avg / pre_avg, 3))


def detect_context_saturation(report, agents, db, experiment_id):
    """Detect context window saturation collapse — multiple agents collapsing near the same turn.

    Called once per experiment (not per-agent) to detect the system-level pattern.
    """
    collapse_turns = _find_collapse_turns(agents, db, experiment_id)

    if len(collapse_turns) < 2:
        return

    # Check if collapses cluster within a narrow turn range (saturation signature)
    turns = list(collapse_turns.values())
    turn_spread = max(turns) - min(turns)

    if turn_spread <= 30:  # all collapses within 30 turns = likely saturation
        report.patterns.append(Pattern(
            pattern_type="context-saturation",
            severity="critical",
            agent="all",
            description=(
                f"Multiple agents collapsed within {turn_spread} turns "
                f"({', '.join(f'{k}@t{v}' for k, v in sorted(collapse_turns.items(), key=lambda x: x[1]))})"
                f" — possible context window saturation"
            ),
            metrics={
                "collapse_turns": collapse_turns,
                "turn_spread": turn_spread,
                "agents_collapsed": len(collapse_turns),
            },
        ))


def detect_cascade_propagation(report, agents, db, experiment_id):
    """Detect collapse cascade — one agent's collapse triggering others.

    Identifies the primary (earliest) collapse, measures lag to secondary
    collapses, and checks for output thinning in agents that didn't fully
    collapse. Reports cascade extent, timing, and resilience.

    Called once per experiment (not per-agent).
    """
    collapse_turns = _find_collapse_turns(agents, db, experiment_id)

    if not collapse_turns:
        return

    # Sort by onset turn
    ordered = sorted(collapse_turns.items(), key=lambda x: x[1])
    primary_name, primary_turn = ordered[0]

    all_agent_names = [a["name"] for a in agents]
    agent_map = {a["name"]: a["agent_id"] for a in agents}

    # Secondary collapses (agents that collapsed after the primary)
    secondary = [(name, turn) for name, turn in ordered[1:]
                 if turn > primary_turn]

    # Agents that survived — check for thinning after primary collapse
    survived = [name for name in all_agent_names if name not in collapse_turns]
    thinning = {}
    for name in survived:
        result = _find_thinning_after(name, db, experiment_id,
                                       agent_map[name], primary_turn)
        if result:
            pre_avg, post_avg, ratio = result
            if ratio < 0.70:  # >30% output reduction = thinning
                thinning[name] = {"pre_avg": pre_avg, "post_avg": post_avg,
                                  "ratio": ratio}

    # Build cascade report
    cascade_agents = len(secondary) + len(thinning)
    total_agents = len(all_agent_names)

    if cascade_agents == 0 and len(collapse_turns) == 1:
        # Single collapse, no cascade — still worth reporting
        report.patterns.append(Pattern(
            pattern_type="isolated-collapse",
            severity="notable",
            agent=primary_name,
            description=(
                f"{primary_name} collapsed at t{primary_turn} with no cascade — "
                f"{', '.join(survived) if survived else 'no'} agents remained stable"
            ),
            metrics={
                "primary": primary_name,
                "primary_turn": primary_turn,
                "cascade_count": 0,
                "survived": survived,
            },
        ))
        return

    # Build description
    parts = [f"{primary_name} collapsed at t{primary_turn}"]

    secondary_details = {}
    if secondary:
        for name, turn in secondary:
            lag = turn - primary_turn
            parts.append(f"{name} followed at t{turn} (lag={lag})")
            secondary_details[name] = {"turn": turn, "lag": lag}

    thinning_details = {}
    if thinning:
        for name, info in thinning.items():
            parts.append(f"{name} thinned to {info['ratio']:.0%} "
                        f"({info['pre_avg']}→{info['post_avg']} chars)")
            thinning_details[name] = info

    if survived and not thinning:
        resistant = [n for n in survived if n not in thinning]
        if resistant:
            parts.append(f"{', '.join(resistant)} resisted cascade")

    # Determine severity based on cascade extent
    cascade_ratio = cascade_agents / (total_agents - 1)  # exclude primary
    if cascade_ratio >= 0.75:
        severity = "critical"
    elif cascade_ratio >= 0.5 or secondary:
        severity = "significant"
    else:
        severity = "notable"

    max_lag = max((t - primary_turn for _, t in secondary), default=0)

    report.patterns.append(Pattern(
        pattern_type="cascade-propagation",
        severity=severity,
        agent="all",
        description=" → ".join(parts),
        metrics={
            "primary": primary_name,
            "primary_turn": primary_turn,
            "secondary_collapses": secondary_details,
            "thinning": thinning_details,
            "cascade_count": cascade_agents,
            "total_agents": total_agents,
            "cascade_ratio": round(cascade_ratio, 2),
            "max_lag": max_lag,
            "resistant": [n for n in survived if n not in thinning],
        },
    ))


def detect_correlation_reversal(report, agents, db, experiment_id):
    """Detect cross-agent output length correlation reversal over time.

    Called once per experiment (not per-agent). Agents speak on different
    turns within each cycle, so we index by message ordinal (1st msg,
    2nd msg, etc.) rather than turn number.
    """
    if len(agents) < 2:
        return

    # Build per-agent length series indexed by message ordinal
    agent_lengths = {}
    for agent in agents:
        messages = db.get_messages(experiment_id, agent_id=agent["agent_id"])
        if not messages:
            continue
        agent_lengths[agent["name"]] = [len(m["content"]) for m in messages]

    names = list(agent_lengths.keys())
    if len(names) < 2:
        return

    # Align by ordinal — use minimum shared length
    min_len = min(len(v) for v in agent_lengths.values())
    if min_len < 40:
        return

    mid = min_len // 2

    def correlation(xs, ys):
        if len(xs) < 10:
            return None
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        if var_x == 0 or var_y == 0:
            return None
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        return cov / (var_x * var_y) ** 0.5

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            early_r = correlation(agent_lengths[a][:mid], agent_lengths[b][:mid])
            late_r = correlation(agent_lengths[a][mid:min_len], agent_lengths[b][mid:min_len])

            if early_r is None or late_r is None:
                continue

            reversal = abs(early_r - late_r)
            if reversal >= CORRELATION_REVERSAL_THRESHOLD:
                direction = "decorrelated" if abs(late_r) < abs(early_r) else "converged"
                report.patterns.append(Pattern(
                    pattern_type="correlation-reversal",
                    severity="notable",
                    agent=f"{a}/{b}",
                    description=(
                        f"{a}-{b} output correlation shifted {early_r:+.2f} → {late_r:+.2f} "
                        f"({direction})"
                    ),
                    metrics={
                        "early_r": round(early_r, 3),
                        "late_r": round(late_r, 3),
                        "reversal": round(reversal, 3),
                    },
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
        "author": "sentinel-auto",
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

# Maps pattern types to implication templates (loaded from config)
if _DETECTION_CONFIG:
    IMPLICATION_MAP = _DETECTION_CONFIG.get("pattern_implications", {})
    COMPARISON_IMPLICATIONS = _DETECTION_CONFIG.get("comparison_implications", {})
else:
    # Minimal fallbacks — real implications live in gitignored config
    IMPLICATION_MAP = {}
    COMPARISON_IMPLICATIONS = {}


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

        # Dedup: skip if an existing finding covers the same experiments + pattern types
        from run_findings import next_id, slugify, FINDINGS_DIR
        FINDINGS_DIR.mkdir(exist_ok=True)
        if _finding_is_duplicate(finding, FINDINGS_DIR):
            if not quiet:
                print(f"Skipping duplicate finding for {', '.join(e[:8] for e in report.experiment_ids)}")
        else:
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


def _finding_is_duplicate(new_finding: dict, findings_dir: Path) -> bool:
    """Check if a finding with the same experiment IDs and pattern types exists."""
    new_comp = new_finding.get("comparison", {})
    new_exps = set()
    if new_comp.get("experiment_a"):
        new_exps.add(new_comp["experiment_a"][:8])
    if new_comp.get("experiment_b"):
        new_exps.add(new_comp["experiment_b"][:8])
    new_tags = set(new_finding.get("tags", []))

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
            # Same experiments and significant tag overlap = duplicate
            if new_exps and new_exps == existing_exps:
                existing_tags = set(existing.get("tags", []))
                overlap = len(new_tags & existing_tags)
                if overlap >= max(2, len(new_tags) // 2):
                    return True
        except (json.JSONDecodeError, OSError):
            pass
    return False


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    main()
