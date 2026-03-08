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
"""

import asyncio
import json
import logging
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
    ):
        """
        Args:
            db: Database instance
            client: Ollama client
            experiment_id: Current experiment
            mode: "shadow", "injected", or "both"
            interval: Run probes every N turns (per agent)
            categories: Which probe categories to run (default: all)
        """
        if mode not in ("shadow", "injected", "both"):
            raise ValueError(f"Invalid probe mode: {mode}. Use 'shadow', 'injected', or 'both'")

        self.db = db
        self.client = client
        self.experiment_id = experiment_id
        self.mode = mode
        self.interval = interval
        self.categories = categories or list(PROBE_PROMPTS.keys())

        # Track last probe turn per agent
        self._last_probe_turn: dict[str, int] = {}
        # Cache baseline responses from calibration
        self._baselines: dict[str, dict[str, str]] = {}

        ensure_probes_schema(db)

    def should_probe(self, agent_id: str, current_turn: int) -> bool:
        """Check if it's time to probe this agent."""
        last = self._last_probe_turn.get(agent_id, 0)
        return (current_turn - last) >= self.interval

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
                    baseline_response, drift_score, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.experiment_id, agent.agent_id, "shadow", current_turn,
                 category, probe_def["prompt"], response["content"],
                 response["inference_ms"], response["prompt_tokens"],
                 response["completion_tokens"], baseline_resp, drift_score, now),
            )
            self.db.conn.commit()

            result = {
                "mode": "shadow",
                "category": category,
                "response": response["content"],
                "drift_score": drift_score,
                "inference_ms": response["inference_ms"],
            }
            results.append(result)

            log.debug(
                "Shadow probe | %s | %s | turn %d | drift=%.3f",
                agent.config.name, category, current_turn,
                drift_score if drift_score is not None else -1,
            )

        return results

    async def run_injected_probe(
        self,
        agent: Agent,
        current_turn: int,
        visible_messages: list[dict],
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
                baseline_response, drift_score, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.experiment_id, agent.agent_id, "injected", current_turn,
             "persona", probe_def["prompt"], response["content"],
             response["inference_ms"], response["prompt_tokens"],
             response["completion_tokens"], baseline_resp, drift_score, now),
        )
        self.db.conn.commit()

        log.info(
            "Injected probe | %s | turn %d | drift=%.3f",
            agent.config.name, current_turn,
            drift_score if drift_score is not None else -1,
        )

        return {
            "mode": "injected",
            "category": "persona",
            "response": response["content"],
            "drift_score": drift_score,
            "message_id": message_id,
            "inference_ms": response["inference_ms"],
        }

    async def run_probes(
        self,
        agent: Agent,
        current_turn: int,
        visible_messages: Optional[list[dict]] = None,
    ) -> list[dict]:
        """Run probes according to the configured mode.

        Call this from the runtime after an agent's turn completes.
        Returns list of probe results.
        """
        if not self.should_probe(agent.agent_id, current_turn):
            return []

        self._last_probe_turn[agent.agent_id] = current_turn
        results = []

        if self.mode in ("shadow", "both"):
            shadow_results = await self.run_shadow_probe(agent, current_turn)
            results.extend(shadow_results)

        if self.mode in ("injected", "both"):
            if visible_messages is None:
                log.warning("Injected probe requested but no visible_messages provided")
            else:
                injected_result = await self.run_injected_probe(
                    agent, current_turn, visible_messages,
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
        "SELECT agent_id, probe_mode, at_turn, category, drift_score "
        "FROM probes WHERE experiment_id=? ORDER BY at_turn, agent_id, category",
        (experiment_id,),
    ).fetchall()

    if not rows:
        print("  No probes recorded.")
        return

    print(f"\n{'Turn':>6s}  {'Agent':<10s}  {'Mode':<10s}  {'Category':<14s}  {'Drift':>8s}")
    print("-" * 55)

    for r in rows:
        name = agent_names.get(r["agent_id"], "?")
        drift = f"{r['drift_score']:.4f}" if r["drift_score"] is not None else "n/a"
        print(f"{r['at_turn']:6d}  {name:<10s}  {r['probe_mode']:<10s}  {r['category']:<14s}  {drift:>8s}")

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
