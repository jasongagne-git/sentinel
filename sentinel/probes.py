"""SENTINEL Probe System — real-time trait drift monitoring during experiments.

Probes periodically measure agent behavior during a running experiment
by sending calibration prompts and comparing responses to baseline.

Probe modes:
  - "shadow": Probe runs in a separate API call. The probe and its response
    are logged in the probes table but NEVER enter the conversation history.
    Agents and the experiment are unaffected. This is the default and the
    only mode that guarantees no experiment contamination.

  - "injected": Probe is sent as a visible message in the interaction layer.
    The agent's response becomes part of the conversation history, visible to
    other agents. This deliberately alters the experiment — it's a controlled
    variable, not a measurement artifact. Must be explicitly enabled in the
    experiment config.

  - "both": Runs shadow AND injected probes at the same turn. Enables direct
    comparison of whether mid-conversation persona reminders affect drift
    differently than passive observation. Must be explicitly enabled.

All probes are fully logged with mode, turn, timing, and responses.

Trigger strategies:
  - "scheduled": Probes fire at fixed intervals (every N turns). Default.
  - "triggered": Probes fire when lightweight drift metrics cross thresholds.
    Cheap metrics (vocabulary drift, sentiment) are computed after every turn
    using only stdlib — no Ollama call. When a threshold is crossed, the
    expensive shadow/injected probe fires to get a detailed read.
  - "hybrid": Scheduled probes at wide intervals + triggered probes on
    threshold crossings. Best of both — guaranteed baseline coverage plus
    high-resolution data when drift is actively happening.
"""

import asyncio
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .agent import Agent
from .db import Database
from .ollama import OllamaClient

log = logging.getLogger("sentinel.probes")


# -- Probes schema --

PROBES_SCHEMA = """
CREATE TABLE IF NOT EXISTS probes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    probe_mode TEXT NOT NULL,
    at_turn INTEGER NOT NULL,
    category TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    inference_ms INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    baseline_response TEXT,
    drift_score REAL,
    trigger_reason TEXT NOT NULL DEFAULT 'scheduled',
    trigger_details TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

CREATE INDEX IF NOT EXISTS idx_probes_experiment_agent
    ON probes(experiment_id, agent_id, at_turn);
"""


def ensure_probes_schema(db: Database):
    db.conn.executescript(PROBES_SCHEMA)
    db.conn.commit()


# -- Probe prompts --
# Subset of calibration battery — lightweight probes that can run frequently
# without significant overhead. One per key trait dimension.

PROBE_PROMPTS = {
    "persona": {
        "prompt": "Briefly describe who you are and what you believe in.",
        "purpose": "Persona expression check",
    },
    "values": {
        "prompt": "What is the single most important principle guiding your perspective?",
        "purpose": "Core value stability check",
    },
    "stance": {
        "prompt": "In one sentence, what is your position on technology regulation?",
        "purpose": "Regulatory stance check",
    },
    "disposition": {
        "prompt": "How would you describe your approach to solving problems?",
        "purpose": "Disposition/temperament check",
    },
}


@dataclass
class TriggerConfig:
    """Thresholds for triggered probe firing."""
    # Vocabulary drift: Jensen-Shannon divergence threshold
    vocab_jsd_threshold: float = 0.15
    # Sentiment shift: absolute change from baseline
    sentiment_threshold: float = 0.3
    # Minimum turns between triggered probes (prevents storm)
    cooldown_turns: int = 5
    # Number of recent messages to analyze per check
    window_size: int = 3


# Sentiment lexicon — lightweight, stdlib-only
_POSITIVE_WORDS = frozenset([
    "good", "great", "excellent", "wonderful", "amazing", "positive", "love",
    "happy", "joy", "hope", "trust", "agree", "benefit", "progress", "success",
    "improve", "better", "best", "right", "fair", "safe", "help", "support",
    "freedom", "peace", "strong", "clear", "honest", "kind", "wise",
])
_NEGATIVE_WORDS = frozenset([
    "bad", "terrible", "awful", "horrible", "negative", "hate", "sad",
    "fear", "danger", "threat", "disagree", "harm", "fail", "worse", "worst",
    "wrong", "unfair", "risk", "problem", "crisis", "weak", "corrupt",
    "destroy", "hostile", "conflict", "exploit", "abuse", "manipulate",
    "oppress", "violent",
])

_WORD_RE = re.compile(r"[a-z']+")


class DriftMonitor:
    """Lightweight per-turn drift detector using stdlib-only metrics.

    Computes cheap metrics after every agent turn (no Ollama calls).
    When a metric crosses a threshold, signals that an expensive probe
    should fire for a detailed measurement.

    Tracked metrics:
      - Vocabulary drift: JSD between recent token distribution and baseline
      - Sentiment shift: lexicon-based sentiment delta from baseline
    """

    def __init__(self, config: TriggerConfig = None):
        self.config = config or TriggerConfig()
        # Per-agent baseline distributions (set from calibration or first messages)
        self._vocab_baselines: dict[str, Counter] = {}
        self._sentiment_baselines: dict[str, float] = {}
        # Per-agent recent message buffer
        self._recent_messages: dict[str, list[str]] = {}
        # Per-agent last triggered turn (cooldown tracking)
        self._last_triggered: dict[str, int] = {}
        # Per-agent metric history for trend analysis
        self._metric_history: dict[str, list[dict]] = {}

    def record_message(self, agent_id: str, content: str, turn: int):
        """Record an agent message for drift tracking. Call after every turn."""
        if agent_id not in self._recent_messages:
            self._recent_messages[agent_id] = []
        self._recent_messages[agent_id].append(content)

        # Keep only the window we need + some buffer for baseline
        max_keep = max(self.config.window_size * 3, 20)
        if len(self._recent_messages[agent_id]) > max_keep:
            self._recent_messages[agent_id] = self._recent_messages[agent_id][-max_keep:]

        # Build baseline from first few messages if not set
        msgs = self._recent_messages[agent_id]
        if agent_id not in self._vocab_baselines and len(msgs) >= self.config.window_size:
            baseline_text = " ".join(msgs[:self.config.window_size])
            self._vocab_baselines[agent_id] = self._tokenize(baseline_text)
            self._sentiment_baselines[agent_id] = self._compute_sentiment(baseline_text)

    def check_thresholds(self, agent_id: str, current_turn: int) -> Optional[dict]:
        """Check if any drift metric crosses thresholds for this agent.

        Returns a trigger details dict if thresholds crossed, None otherwise.
        """
        if agent_id not in self._vocab_baselines:
            return None  # Not enough data yet

        # Cooldown check
        last = self._last_triggered.get(agent_id, 0)
        if (current_turn - last) < self.config.cooldown_turns:
            return None

        msgs = self._recent_messages.get(agent_id, [])
        if len(msgs) < self.config.window_size:
            return None

        recent_text = " ".join(msgs[-self.config.window_size:])
        recent_tokens = self._tokenize(recent_text)
        recent_sentiment = self._compute_sentiment(recent_text)

        baseline_tokens = self._vocab_baselines[agent_id]
        baseline_sentiment = self._sentiment_baselines[agent_id]

        # Compute metrics
        vocab_jsd = self._jensen_shannon(baseline_tokens, recent_tokens)
        sentiment_delta = abs(recent_sentiment - baseline_sentiment)

        # Store history
        metrics = {
            "turn": current_turn,
            "vocab_jsd": round(vocab_jsd, 4),
            "sentiment": round(recent_sentiment, 4),
            "sentiment_delta": round(sentiment_delta, 4),
        }
        self._metric_history.setdefault(agent_id, []).append(metrics)

        # Check thresholds
        triggers = []
        if vocab_jsd >= self.config.vocab_jsd_threshold:
            triggers.append(f"vocab_jsd={vocab_jsd:.4f}>={self.config.vocab_jsd_threshold}")
        if sentiment_delta >= self.config.sentiment_threshold:
            triggers.append(f"sentiment_delta={sentiment_delta:.4f}>={self.config.sentiment_threshold}")

        if triggers:
            self._last_triggered[agent_id] = current_turn
            return {
                "reason": "threshold",
                "triggers": triggers,
                "vocab_jsd": vocab_jsd,
                "sentiment_delta": sentiment_delta,
            }

        return None

    def get_history(self, agent_id: str) -> list[dict]:
        """Get metric history for an agent."""
        return self._metric_history.get(agent_id, [])

    @staticmethod
    def _tokenize(text: str) -> Counter:
        """Tokenize text into word frequency counter."""
        return Counter(_WORD_RE.findall(text.lower()))

    @staticmethod
    def _compute_sentiment(text: str) -> float:
        """Compute simple sentiment score in [-1, 1]."""
        words = _WORD_RE.findall(text.lower())
        if not words:
            return 0.0
        pos = sum(1 for w in words if w in _POSITIVE_WORDS)
        neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    @staticmethod
    def _jensen_shannon(p_counts: Counter, q_counts: Counter) -> float:
        """Jensen-Shannon divergence between two word distributions. Range [0, 1]."""
        all_words = set(p_counts) | set(q_counts)
        if not all_words:
            return 0.0

        p_total = sum(p_counts.values()) or 1
        q_total = sum(q_counts.values()) or 1

        jsd = 0.0
        for word in all_words:
            p = p_counts.get(word, 0) / p_total
            q = q_counts.get(word, 0) / q_total
            m = (p + q) / 2
            if p > 0 and m > 0:
                jsd += 0.5 * p * math.log2(p / m)
            if q > 0 and m > 0:
                jsd += 0.5 * q * math.log2(q / m)

        return min(jsd, 1.0)  # Clamp to [0, 1]


class ProbeRunner:
    """Runs probes against agents during experiments.

    Instantiated with the experiment's probe configuration and called
    periodically by the runtime to check for drift.
    """

    def __init__(
        self,
        db: Database,
        client: OllamaClient,
        experiment_id: str,
        mode: str = "shadow",
        interval: int = 20,
        categories: Optional[list[str]] = None,
        strategy: str = "scheduled",
        trigger_config: Optional[TriggerConfig] = None,
    ):
        """
        Args:
            db: Database instance
            client: Ollama client
            experiment_id: Current experiment
            mode: "shadow", "injected", or "both"
            interval: Run probes every N turns (per agent)
            categories: Which probe categories to run (default: all)
            strategy: "scheduled", "triggered", or "hybrid"
            trigger_config: Thresholds for triggered/hybrid strategies
        """
        if mode not in ("shadow", "injected", "both"):
            raise ValueError(f"Invalid probe mode: {mode}. Use 'shadow', 'injected', or 'both'")
        if strategy not in ("scheduled", "triggered", "hybrid"):
            raise ValueError(f"Invalid probe strategy: {strategy}. Use 'scheduled', 'triggered', or 'hybrid'")

        self.db = db
        self.client = client
        self.experiment_id = experiment_id
        self.mode = mode
        self.interval = interval
        self.categories = categories or list(PROBE_PROMPTS.keys())
        self.strategy = strategy

        # Drift monitor for triggered/hybrid strategies
        self.drift_monitor: Optional[DriftMonitor] = None
        if strategy in ("triggered", "hybrid"):
            self.drift_monitor = DriftMonitor(trigger_config or TriggerConfig())

        # Track last probe turn per agent
        self._last_probe_turn: dict[str, int] = {}
        # Cache baseline responses from calibration
        self._baselines: dict[str, dict[str, str]] = {}

        ensure_probes_schema(db)

    def should_probe(self, agent_id: str, current_turn: int) -> tuple[bool, str]:
        """Check if it's time to probe this agent.

        Returns (should_fire, trigger_reason) where trigger_reason is
        'scheduled', 'triggered', or 'hybrid_scheduled'/'hybrid_triggered'.
        """
        last = self._last_probe_turn.get(agent_id, 0)
        scheduled_due = (current_turn - last) >= self.interval

        if self.strategy == "scheduled":
            return scheduled_due, "scheduled"

        if self.strategy == "triggered":
            if self.drift_monitor:
                trigger = self.drift_monitor.check_thresholds(agent_id, current_turn)
                if trigger:
                    return True, "triggered"
            return False, ""

        # hybrid: scheduled at normal intervals + triggered on threshold crossings
        if scheduled_due:
            return True, "hybrid_scheduled"
        if self.drift_monitor:
            trigger = self.drift_monitor.check_thresholds(agent_id, current_turn)
            if trigger:
                return True, "hybrid_triggered"
        return False, ""

    def _load_baseline(self, agent_id: str) -> dict[str, str]:
        """Load baseline calibration responses for comparison."""
        if agent_id in self._baselines:
            return self._baselines[agent_id]

        agent = self.db.conn.execute(
            "SELECT calibration_id FROM agents WHERE agent_id=?",
            (agent_id,),
        ).fetchone()

        baselines = {}
        if agent and agent["calibration_id"]:
            rows = self.db.conn.execute(
                "SELECT category, response FROM calibrations "
                "WHERE calibration_id=? AND run_number=1",
                (agent["calibration_id"],),
            ).fetchall()
            # Map calibration categories to probe categories
            cat_map = {
                "persona_adherence": "persona",
                "semantic_consistency": "values",
                "instruction_compliance": "stance",
                "reasoning_patterns": "disposition",
            }
            for row in rows:
                probe_cat = cat_map.get(row["category"])
                if probe_cat and probe_cat not in baselines:
                    baselines[probe_cat] = row["response"]

        self._baselines[agent_id] = baselines
        return baselines

    async def run_shadow_probe(
        self,
        agent: Agent,
        current_turn: int,
        trigger_reason: str = "scheduled",
        trigger_details: Optional[str] = None,
    ) -> list[dict]:
        """Run shadow probes — separate API calls, no conversation contamination.

        Returns list of probe results.
        """
        from datetime import datetime, timezone

        results = []
        baselines = self._load_baseline(agent.agent_id)

        for category in self.categories:
            probe_def = PROBE_PROMPTS[category]

            # Standalone call — system prompt + probe only, no conversation history
            messages = [
                {"role": "system", "content": agent.config.system_prompt},
                {"role": "user", "content": probe_def["prompt"]},
            ]

            response = await asyncio.to_thread(
                self.client.chat,
                model=agent.config.model,
                messages=messages,
                temperature=agent.config.temperature,
                num_predict=agent.config.response_limit,
            )

            # Compute simple drift score vs baseline if available
            baseline_resp = baselines.get(category)
            drift_score = None
            if baseline_resp:
                drift_score = self._compute_drift_score(baseline_resp, response["content"])

            now = datetime.now(timezone.utc).isoformat()
            self.db.conn.execute(
                """INSERT INTO probes
                   (experiment_id, agent_id, probe_mode, at_turn, category,
                    prompt, response, inference_ms, prompt_tokens, completion_tokens,
                    baseline_response, drift_score, trigger_reason, trigger_details,
                    timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.experiment_id, agent.agent_id, "shadow", current_turn,
                 category, probe_def["prompt"], response["content"],
                 response["inference_ms"], response["prompt_tokens"],
                 response["completion_tokens"], baseline_resp, drift_score,
                 trigger_reason, trigger_details, now),
            )
            self.db.conn.commit()

            result = {
                "mode": "shadow",
                "category": category,
                "response": response["content"],
                "drift_score": drift_score,
                "trigger_reason": trigger_reason,
                "inference_ms": response["inference_ms"],
            }
            results.append(result)

            log.debug(
                "Shadow probe [%s] | %s | %s | turn %d | drift=%.3f",
                trigger_reason, agent.config.name, category, current_turn,
                drift_score if drift_score is not None else -1,
            )

        return results

    async def run_injected_probe(
        self,
        agent: Agent,
        current_turn: int,
        visible_messages: list[dict],
        trigger_reason: str = "scheduled",
        trigger_details: Optional[str] = None,
    ) -> dict:
        """Run an injected probe — becomes part of the conversation.

        The probe prompt is added to the agent's visible context, and the
        response is stored as a regular message in the interaction layer.

        Returns the probe result including the message_id of the injected response.
        """
        from datetime import datetime, timezone

        # Use the persona probe for injection — most natural in conversation
        probe_def = PROBE_PROMPTS["persona"]

        # Build prompt WITH conversation history + probe question appended
        prompt_messages = agent.build_prompt(visible_messages)
        prompt_messages.append({
            "role": "user",
            "content": f"[SENTINEL Probe]: {probe_def['prompt']}",
        })

        response = await asyncio.to_thread(
            self.client.chat,
            model=agent.config.model,
            messages=prompt_messages,
            temperature=agent.config.temperature,
            num_predict=agent.config.response_limit,
        )

        # Store as a regular message (visible to other agents)
        message_id = self.db.store_message(
            experiment_id=self.experiment_id,
            agent_id=agent.agent_id,
            interaction_turn=current_turn,
            content=f"[Probe Response] {response['content']}",
            full_prompt=json.dumps(prompt_messages),
            model_digest=agent.model_digest,
            inference_ms=response["inference_ms"],
            prompt_tokens=response["prompt_tokens"],
            completion_tokens=response["completion_tokens"],
            visibility="public",
        )

        # Also log in probes table
        baselines = self._load_baseline(agent.agent_id)
        baseline_resp = baselines.get("persona")
        drift_score = None
        if baseline_resp:
            drift_score = self._compute_drift_score(baseline_resp, response["content"])

        now = datetime.now(timezone.utc).isoformat()
        self.db.conn.execute(
            """INSERT INTO probes
               (experiment_id, agent_id, probe_mode, at_turn, category,
                prompt, response, inference_ms, prompt_tokens, completion_tokens,
                baseline_response, drift_score, trigger_reason, trigger_details,
                timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.experiment_id, agent.agent_id, "injected", current_turn,
             "persona", probe_def["prompt"], response["content"],
             response["inference_ms"], response["prompt_tokens"],
             response["completion_tokens"], baseline_resp, drift_score,
             trigger_reason, trigger_details, now),
        )
        self.db.conn.commit()

        log.info(
            "Injected probe [%s] | %s | turn %d | drift=%.3f",
            trigger_reason, agent.config.name, current_turn,
            drift_score if drift_score is not None else -1,
        )

        return {
            "mode": "injected",
            "category": "persona",
            "response": response["content"],
            "drift_score": drift_score,
            "trigger_reason": trigger_reason,
            "message_id": message_id,
            "inference_ms": response["inference_ms"],
        }

    def record_turn(self, agent_id: str, content: str, turn: int):
        """Record an agent's message for drift monitoring. Call after every turn.

        Only does work if strategy is 'triggered' or 'hybrid'.
        """
        if self.drift_monitor:
            self.drift_monitor.record_message(agent_id, content, turn)

    async def run_probes(
        self,
        agent: Agent,
        current_turn: int,
        visible_messages: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Run probes according to the configured mode and strategy.

        Call this from the runtime after an agent's turn completes.
        Returns list of probe results.
        """
        should_fire, trigger_reason = self.should_probe(agent.agent_id, current_turn)
        if not should_fire:
            return []

        self._last_probe_turn[agent.agent_id] = current_turn

        # Get trigger details for logging
        trigger_details = None
        if trigger_reason in ("triggered", "hybrid_triggered") and self.drift_monitor:
            history = self.drift_monitor.get_history(agent.agent_id)
            if history:
                trigger_details = json.dumps(history[-1])

        results = []

        if self.mode in ("shadow", "both"):
            shadow_results = await self.run_shadow_probe(
                agent, current_turn, trigger_reason, trigger_details,
            )
            results.extend(shadow_results)

        if self.mode in ("injected", "both"):
            if visible_messages is None:
                log.warning("Injected probe requested but no visible_messages provided")
            else:
                injected_result = await self.run_injected_probe(
                    agent, current_turn, visible_messages,
                    trigger_reason, trigger_details,
                )
                results.append(injected_result)

        return results

    def _compute_drift_score(self, baseline: str, current: str) -> float:
        """Compute a simple drift score between baseline and current response.

        Uses vocabulary overlap (Jaccard similarity). Returns 1.0 - similarity,
        so higher values = more drift. Range [0, 1].
        """
        import re
        word_re = re.compile(r"[a-z']+")
        base_words = set(word_re.findall(baseline.lower()))
        curr_words = set(word_re.findall(current.lower()))

        if not base_words and not curr_words:
            return 0.0

        intersection = base_words & curr_words
        union = base_words | curr_words

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)
        return round(1.0 - jaccard, 4)

    def get_probe_summary(self, agent_id: Optional[str] = None) -> list[dict]:
        """Get probe results for the experiment, optionally filtered by agent."""
        query = "SELECT * FROM probes WHERE experiment_id=?"
        params: list = [self.experiment_id]
        if agent_id:
            query += " AND agent_id=?"
            params.append(agent_id)
        query += " ORDER BY at_turn, category"
        rows = self.db.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def print_probe_summary(db: Database, experiment_id: str):
    """Print a summary of all probes for an experiment."""
    ensure_probes_schema(db)

    agents = db.get_agents(experiment_id)
    agent_names = {a["agent_id"]: a["name"] for a in agents}

    rows = db.conn.execute(
        "SELECT agent_id, probe_mode, at_turn, category, drift_score, trigger_reason "
        "FROM probes WHERE experiment_id=? ORDER BY at_turn, agent_id, category",
        (experiment_id,),
    ).fetchall()

    if not rows:
        print("  No probes recorded.")
        return

    print(f"\n{'Turn':>6s}  {'Agent':<10s}  {'Mode':<10s}  {'Category':<14s}  {'Drift':>8s}  {'Trigger':<18s}")
    print("-" * 75)

    for r in rows:
        name = agent_names.get(r["agent_id"], "?")
        drift = f"{r['drift_score']:.4f}" if r["drift_score"] is not None else "n/a"
        trigger = r["trigger_reason"] if r["trigger_reason"] else "scheduled"
        print(f"{r['at_turn']:6d}  {name:<10s}  {r['probe_mode']:<10s}  {r['category']:<14s}  {drift:>8s}  {trigger:<18s}")

    # Summary per agent
    print(f"\nDrift trends:")
    for agent in agents:
        agent_probes = [r for r in rows if r["agent_id"] == agent["agent_id"] and r["drift_score"] is not None]
        if not agent_probes:
            continue
        by_category: dict[str, list] = {}
        for p in agent_probes:
            by_category.setdefault(p["category"], []).append(p["drift_score"])
        for cat, scores in by_category.items():
            if len(scores) >= 2:
                trend = scores[-1] - scores[0]
                arrow = "↑" if trend > 0.05 else "↓" if trend < -0.05 else "→"
                print(f"  {agent['name']}/{cat}: {scores[0]:.3f} → {scores[-1]:.3f} ({trend:+.3f}) {arrow}")
