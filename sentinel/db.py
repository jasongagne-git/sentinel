"""SQLite message broker and experiment logging.

Stores all agent interactions, experiment metadata, and calibration data.
Uses WAL mode for concurrent read access during experiments.
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    topology TEXT NOT NULL DEFAULT 'full_mesh',
    cycle_delay_s REAL NOT NULL DEFAULT 30.0,
    max_turns INTEGER,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'created',
    forked_from_experiment_id TEXT,
    fork_at_turn INTEGER,
    FOREIGN KEY (forked_from_experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    model TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    temperature REAL NOT NULL DEFAULT 0.7,
    max_history INTEGER NOT NULL DEFAULT 50,
    response_limit INTEGER NOT NULL DEFAULT 256,
    calibration_id TEXT,
    is_control INTEGER NOT NULL DEFAULT 0,
    config_json TEXT NOT NULL,
    traits_json TEXT,
    trait_fingerprint TEXT,
    forked_from_agent_id TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY (forked_from_agent_id) REFERENCES agents(agent_id)
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    interaction_turn INTEGER NOT NULL,
    content TEXT NOT NULL,
    reply_to TEXT,
    visibility TEXT NOT NULL DEFAULT 'public',
    timestamp TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    inference_ms INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    full_prompt TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

CREATE TABLE IF NOT EXISTS calibrations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    calibration_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    model TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    category TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    inference_ms INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (agent_id) REFERENCES agents(agent_id)
);

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

CREATE INDEX IF NOT EXISTS idx_messages_experiment ON messages(experiment_id, interaction_turn);
CREATE INDEX IF NOT EXISTS idx_messages_agent ON messages(agent_id, interaction_turn);
CREATE INDEX IF NOT EXISTS idx_agents_experiment ON agents(experiment_id);
CREATE INDEX IF NOT EXISTS idx_calibrations_agent ON calibrations(agent_id);
CREATE INDEX IF NOT EXISTS idx_probes_experiment_agent ON probes(experiment_id, agent_id, at_turn);
"""


class Database:
    """SQLite database for experiment logging."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # -- Experiments --

    def create_experiment(
        self,
        name: str,
        config: dict,
        description: str = "",
        topology: str = "full_mesh",
        cycle_delay_s: float = 30.0,
        max_turns: Optional[int] = None,
    ) -> str:
        import json

        experiment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO experiments
               (experiment_id, name, description, topology, cycle_delay_s, max_turns, config_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, name, description, topology, cycle_delay_s, max_turns,
             json.dumps(config), now),
        )
        self.conn.commit()
        return experiment_id

    def update_experiment_status(self, experiment_id: str, status: str):
        now = datetime.now(timezone.utc).isoformat()
        field = "started_at" if status == "running" else "completed_at"
        self.conn.execute(
            f"UPDATE experiments SET status=?, {field}=? WHERE experiment_id=?",
            (status, now, experiment_id),
        )
        self.conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
        ).fetchone()
        return dict(row) if row else None

    # -- Agents --

    def create_agent(
        self,
        experiment_id: str,
        name: str,
        system_prompt: str,
        model: str,
        model_digest: str,
        temperature: float = 0.7,
        max_history: int = 50,
        response_limit: int = 256,
        is_control: bool = False,
        traits_json: Optional[str] = None,
        trait_fingerprint: Optional[str] = None,
        forked_from_agent_id: Optional[str] = None,
        calibration_id: Optional[str] = None,
    ) -> str:
        import json

        agent_id = str(uuid.uuid4())
        config = {
            "name": name,
            "system_prompt": system_prompt,
            "model": model,
            "model_digest": model_digest,
            "temperature": temperature,
            "max_history": max_history,
            "response_limit": response_limit,
            "is_control": is_control,
        }
        self.conn.execute(
            """INSERT INTO agents
               (agent_id, experiment_id, name, system_prompt, model, model_digest,
                temperature, max_history, response_limit, is_control, config_json,
                traits_json, trait_fingerprint, forked_from_agent_id, calibration_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (agent_id, experiment_id, name, system_prompt, model, model_digest,
             temperature, max_history, response_limit, int(is_control), json.dumps(config),
             traits_json, trait_fingerprint, forked_from_agent_id, calibration_id),
        )
        self.conn.commit()
        return agent_id

    def find_calibration_id(
        self,
        name: str,
        model: str,
        model_digest: str,
    ) -> Optional[str]:
        """Find the most recent calibration_id for an agent name + model + digest.

        Looks up prior agents with the same name, model, and model_digest that
        have a calibration_id set, and returns the most recent one.
        """
        row = self.conn.execute(
            """SELECT a.calibration_id
               FROM agents a
               JOIN calibrations c ON a.calibration_id = c.calibration_id
               WHERE a.name = ? AND a.model = ? AND a.model_digest = ?
                 AND a.calibration_id IS NOT NULL
               GROUP BY a.calibration_id
               ORDER BY MAX(c.timestamp) DESC
               LIMIT 1""",
            (name, model, model_digest),
        ).fetchone()
        return row["calibration_id"] if row else None

    def get_agents(self, experiment_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM agents WHERE experiment_id=? ORDER BY name",
            (experiment_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Messages --

    def store_message(
        self,
        experiment_id: str,
        agent_id: str,
        interaction_turn: int,
        content: str,
        full_prompt: str,
        model_digest: str,
        inference_ms: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reply_to: Optional[str] = None,
        visibility: str = "public",
    ) -> str:
        message_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO messages
               (message_id, experiment_id, agent_id, interaction_turn, content,
                reply_to, visibility, timestamp, model_digest, inference_ms,
                prompt_tokens, completion_tokens, full_prompt)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (message_id, experiment_id, agent_id, interaction_turn, content,
             reply_to, visibility, now, model_digest, inference_ms,
             prompt_tokens, completion_tokens, full_prompt),
        )
        self.conn.commit()
        return message_id

    def get_messages(
        self,
        experiment_id: str,
        limit: Optional[int] = None,
        after_turn: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> list[dict]:
        query = "SELECT * FROM messages WHERE experiment_id=?"
        params: list = [experiment_id]
        if after_turn is not None:
            query += " AND interaction_turn > ?"
            params.append(after_turn)
        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY interaction_turn ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_latest_turn(self, experiment_id: str) -> int:
        row = self.conn.execute(
            "SELECT MAX(interaction_turn) as max_turn FROM messages WHERE experiment_id=?",
            (experiment_id,),
        ).fetchone()
        return row["max_turn"] if row and row["max_turn"] is not None else 0

    def get_resume_position(self, experiment_id: str) -> tuple:
        """Return (last_turn, last_agent_id) for mid-turn resume.

        Finds the last successfully stored message so the runtime can
        resume from the next agent in the rotation, not the next full cycle.
        """
        row = self.conn.execute(
            "SELECT interaction_turn, agent_id FROM messages "
            "WHERE experiment_id=? ORDER BY interaction_turn DESC LIMIT 1",
            (experiment_id,),
        ).fetchone()
        if row:
            return row["interaction_turn"], row["agent_id"]
        return 0, None

    # -- Calibrations --

    def store_calibration(
        self,
        calibration_id: str,
        agent_id: str,
        model: str,
        model_digest: str,
        run_number: int,
        category: str,
        prompt: str,
        response: str,
        inference_ms: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO calibrations
               (calibration_id, agent_id, model, model_digest, run_number,
                category, prompt, response, inference_ms, prompt_tokens,
                completion_tokens, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (calibration_id, agent_id, model, model_digest, run_number,
             category, prompt, response, inference_ms, prompt_tokens,
             completion_tokens, now),
        )
        self.conn.commit()
