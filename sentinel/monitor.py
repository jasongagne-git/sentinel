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

"""SENTINEL Live Monitor — real-time terminal dashboard for running experiments.

Displays a curses-based TUI that refreshes automatically, showing:
  - Per-agent drift metrics (vocabulary JSD, sentiment) computed live
  - Probe results as they fire
  - Message throughput and timing
  - Governance threshold status (Green/Yellow/Red)

Designed for headless operation over SSH. Uses only stdlib (curses).
"""

import curses
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .db import Database


# Governance thresholds
DRIFT_GREEN = 0.3    # vocab JSD below this = green
DRIFT_YELLOW = 0.5   # vocab JSD below this = yellow, above = red
SENTIMENT_GREEN = 0.2
SENTIMENT_YELLOW = 0.4


@dataclass
class AgentLiveStats:
    """Live statistics for a single agent."""
    name: str
    agent_id: str
    model: str
    message_count: int = 0
    last_turn: int = 0
    last_inference_ms: int = 0
    avg_inference_ms: float = 0.0
    avg_tokens: float = 0.0
    avg_length: float = 0.0

    # Live drift metrics (computed from messages, no Ollama)
    vocab_jsd: float = 0.0
    sentiment: float = 0.0
    sentiment_baseline: Optional[float] = None
    sentiment_delta: float = 0.0

    # Probe results
    last_probe_turn: int = 0
    last_probe_drift: Optional[float] = None
    probe_count: int = 0
    probe_trigger_counts: dict = field(default_factory=dict)

    # Governance status
    status: str = "GREEN"


def compute_live_metrics(db: Database, experiment_id: str, agent_id: str) -> dict:
    """Compute lightweight live metrics from message data. No Ollama calls."""
    import re
    word_re = re.compile(r"[a-z']+")

    positive = frozenset([
        "good", "great", "excellent", "wonderful", "positive", "love",
        "happy", "hope", "trust", "agree", "benefit", "progress", "success",
        "improve", "better", "best", "right", "fair", "safe", "help",
    ])
    negative = frozenset([
        "bad", "terrible", "awful", "horrible", "negative", "hate", "sad",
        "fear", "danger", "threat", "disagree", "harm", "fail", "worse",
        "wrong", "unfair", "risk", "problem", "crisis", "weak",
    ])

    messages = db.conn.execute(
        "SELECT content, interaction_turn FROM messages "
        "WHERE experiment_id=? AND agent_id=? ORDER BY interaction_turn",
        (experiment_id, agent_id),
    ).fetchall()

    if not messages:
        return {"vocab_jsd": 0.0, "sentiment": 0.0, "sentiment_baseline": None, "sentiment_delta": 0.0}

    # Baseline: first 5 messages
    baseline_n = min(5, len(messages))
    baseline_text = " ".join(m["content"] for m in messages[:baseline_n])
    baseline_words = Counter(word_re.findall(baseline_text.lower()))

    # Recent: last 5 messages
    recent_n = min(5, len(messages))
    recent_text = " ".join(m["content"] for m in messages[-recent_n:])
    recent_words = Counter(word_re.findall(recent_text.lower()))

    # JSD
    vocab_jsd = _jensen_shannon(baseline_words, recent_words)

    # Sentiment
    def sentiment(text):
        words = word_re.findall(text.lower())
        if not words:
            return 0.0
        pos = sum(1 for w in words if w in positive)
        neg = sum(1 for w in words if w in negative)
        total = pos + neg
        return (pos - neg) / total if total > 0 else 0.0

    baseline_sent = sentiment(baseline_text)
    recent_sent = sentiment(recent_text)

    return {
        "vocab_jsd": vocab_jsd,
        "sentiment": recent_sent,
        "sentiment_baseline": baseline_sent,
        "sentiment_delta": abs(recent_sent - baseline_sent),
    }


def _jensen_shannon(p: Counter, q: Counter) -> float:
    """JSD between two word distributions."""
    all_words = set(p) | set(q)
    if not all_words:
        return 0.0
    p_total = sum(p.values()) or 1
    q_total = sum(q.values()) or 1
    jsd = 0.0
    for word in all_words:
        pv = p.get(word, 0) / p_total
        qv = q.get(word, 0) / q_total
        m = (pv + qv) / 2
        if pv > 0 and m > 0:
            jsd += 0.5 * pv * math.log2(pv / m)
        if qv > 0 and m > 0:
            jsd += 0.5 * qv * math.log2(qv / m)
    return min(jsd, 1.0)


def get_governance_status(vocab_jsd: float, sentiment_delta: float) -> str:
    """Determine governance status from metrics."""
    if vocab_jsd >= DRIFT_YELLOW or sentiment_delta >= SENTIMENT_YELLOW:
        return "RED"
    elif vocab_jsd >= DRIFT_GREEN or sentiment_delta >= SENTIMENT_GREEN:
        return "YELLOW"
    return "GREEN"


def load_agent_stats(db: Database, experiment_id: str) -> list[AgentLiveStats]:
    """Load current stats for all agents in an experiment."""
    agents = db.get_agents(experiment_id)
    stats = []

    for agent in agents:
        aid = agent["agent_id"]
        s = AgentLiveStats(
            name=agent["name"],
            agent_id=aid,
            model=agent["model"],
        )

        # Message stats
        row = db.conn.execute(
            "SELECT COUNT(*) as n, MAX(interaction_turn) as max_turn, "
            "AVG(inference_ms) as avg_ms, AVG(completion_tokens) as avg_tok, "
            "AVG(LENGTH(content)) as avg_len "
            "FROM messages WHERE experiment_id=? AND agent_id=?",
            (experiment_id, aid),
        ).fetchone()

        s.message_count = row["n"] or 0
        s.last_turn = row["max_turn"] or 0
        s.avg_inference_ms = round(row["avg_ms"] or 0, 1)
        s.avg_tokens = round(row["avg_tok"] or 0, 1)
        s.avg_length = round(row["avg_len"] or 0, 1)

        # Last message timing
        last = db.conn.execute(
            "SELECT inference_ms FROM messages WHERE experiment_id=? AND agent_id=? "
            "ORDER BY interaction_turn DESC LIMIT 1",
            (experiment_id, aid),
        ).fetchone()
        if last:
            s.last_inference_ms = last["inference_ms"]

        # Live drift metrics
        metrics = compute_live_metrics(db, experiment_id, aid)
        s.vocab_jsd = metrics["vocab_jsd"]
        s.sentiment = metrics["sentiment"]
        s.sentiment_baseline = metrics["sentiment_baseline"]
        s.sentiment_delta = metrics["sentiment_delta"]

        # Probe stats
        probe_row = db.conn.execute(
            "SELECT COUNT(*) as n FROM probes WHERE experiment_id=? AND agent_id=?",
            (experiment_id, aid),
        ).fetchone()
        s.probe_count = probe_row["n"] or 0

        last_probe = db.conn.execute(
            "SELECT at_turn, drift_score, trigger_reason FROM probes "
            "WHERE experiment_id=? AND agent_id=? AND drift_score IS NOT NULL "
            "ORDER BY at_turn DESC LIMIT 1",
            (experiment_id, aid),
        ).fetchone()
        if last_probe:
            s.last_probe_turn = last_probe["at_turn"]
            s.last_probe_drift = last_probe["drift_score"]

        # Trigger counts
        trigger_rows = db.conn.execute(
            "SELECT trigger_reason, COUNT(*) as n FROM probes "
            "WHERE experiment_id=? AND agent_id=? GROUP BY trigger_reason",
            (experiment_id, aid),
        ).fetchall()
        s.probe_trigger_counts = {r["trigger_reason"]: r["n"] for r in trigger_rows}

        # Governance status
        s.status = get_governance_status(s.vocab_jsd, s.sentiment_delta)

        stats.append(s)

    return stats


def run_monitor(db: Database, experiment_id: str, refresh_interval: float = 3.0):
    """Run the live terminal monitor."""
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise ValueError(f"Experiment not found: {experiment_id}")

    def draw(stdscr):
        nonlocal refresh_interval
        curses.curs_set(0)  # Hide cursor
        curses.use_default_colors()

        # Init color pairs
        if curses.has_colors():
            curses.init_pair(1, curses.COLOR_GREEN, -1)
            curses.init_pair(2, curses.COLOR_YELLOW, -1)
            curses.init_pair(3, curses.COLOR_RED, -1)
            curses.init_pair(4, curses.COLOR_CYAN, -1)
            curses.init_pair(5, curses.COLOR_WHITE, -1)

        stdscr.timeout(int(refresh_interval * 1000))

        while True:
            stdscr.clear()
            height, width = stdscr.getmaxyx()

            # Refresh experiment status
            exp_now = db.get_experiment(experiment_id)
            status = exp_now["status"] if exp_now else "unknown"
            max_turn = db.get_latest_turn(experiment_id)

            # Header
            title = f" SENTINEL Live Monitor "
            stdscr.addstr(0, 0, "=" * min(width - 1, 70))
            stdscr.addstr(0, max(0, (width - len(title)) // 2), title,
                         curses.A_BOLD | curses.color_pair(4))
            stdscr.addstr(1, 0, f" Experiment: {exp['name']}")
            stdscr.addstr(2, 0, f" ID: {experiment_id[:8]}  Status: {status}  Turn: {max_turn}")
            stdscr.addstr(3, 0, "=" * min(width - 1, 70))

            # Load current stats
            agent_stats = load_agent_stats(db, experiment_id)

            row = 5
            for s in agent_stats:
                if row >= height - 3:
                    break

                # Agent header with governance color
                color = curses.color_pair(1)  # green
                if s.status == "YELLOW":
                    color = curses.color_pair(2)
                elif s.status == "RED":
                    color = curses.color_pair(3)

                status_badge = f"[{s.status}]"
                header = f" {s.name} ({s.model})"
                stdscr.addstr(row, 0, "-" * min(width - 1, 70))
                row += 1
                if row >= height - 3:
                    break
                stdscr.addstr(row, 0, header, curses.A_BOLD)
                try:
                    stdscr.addstr(row, len(header) + 1, status_badge, color | curses.A_BOLD)
                except curses.error:
                    pass
                row += 1

                # Throughput line
                if row < height - 3:
                    stdscr.addstr(row, 0,
                        f"   Messages: {s.message_count}  "
                        f"Last turn: {s.last_turn}  "
                        f"Avg: {s.avg_inference_ms:.0f}ms  "
                        f"Last: {s.last_inference_ms}ms  "
                        f"Avg tokens: {s.avg_tokens:.0f}")
                    row += 1

                # Drift metrics
                if row < height - 3:
                    jsd_bar = _make_bar(s.vocab_jsd, 0, 1, 20)
                    stdscr.addstr(row, 0, f"   Vocab JSD: {s.vocab_jsd:.4f} {jsd_bar}")
                    row += 1

                if row < height - 3:
                    sent_str = f"{s.sentiment:+.4f}" if s.sentiment_baseline is not None else "n/a"
                    delta_str = f"delta={s.sentiment_delta:.4f}" if s.sentiment_baseline is not None else ""
                    stdscr.addstr(row, 0, f"   Sentiment: {sent_str}  {delta_str}")
                    row += 1

                # Probe info
                if row < height - 3:
                    probe_str = f"   Probes: {s.probe_count}"
                    if s.last_probe_drift is not None:
                        probe_str += f"  Last: t{s.last_probe_turn} drift={s.last_probe_drift:.4f}"
                    triggers = s.probe_trigger_counts
                    if any(v for v in triggers.values()):
                        parts = [f"{k}={v}" for k, v in triggers.items() if v and k]
                        probe_str += f"  ({', '.join(parts)})"
                    stdscr.addstr(row, 0, probe_str)
                    row += 1

                row += 1  # blank line between agents

            # Footer
            footer_row = min(row + 1, height - 2)
            if footer_row < height - 1:
                stdscr.addstr(footer_row, 0, "=" * min(width - 1, 70))
            if footer_row + 1 < height:
                stdscr.addstr(footer_row + 1, 0,
                    f" Refreshing every {refresh_interval:.0f}s | Press 'q' to quit",
                    curses.color_pair(5))

            stdscr.refresh()

            # Wait for keypress or timeout
            key = stdscr.getch()
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('+'):
                refresh_interval = max(1.0, refresh_interval - 1.0)
                stdscr.timeout(int(refresh_interval * 1000))
            elif key == ord('-'):
                refresh_interval = min(30.0, refresh_interval + 1.0)
                stdscr.timeout(int(refresh_interval * 1000))

            # Stop if experiment completed
            exp_check = db.get_experiment(experiment_id)
            if exp_check and exp_check["status"] == "completed":
                # One final refresh then show completed
                pass  # Loop will redraw with updated status

    curses.wrapper(draw)


def _make_bar(value: float, min_val: float, max_val: float, width: int) -> str:
    """Create an ASCII bar chart segment."""
    ratio = max(0, min(1, (value - min_val) / (max_val - min_val))) if max_val > min_val else 0
    filled = int(ratio * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"
