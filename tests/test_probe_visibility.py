# Copyright 2026 Jason Gagne
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Regression tests for probe visibility filtering.

The bug class these tests catch:
  Probe responses are stored in the messages table and, if not filtered
  by the runtime, leak into every other agent's context window. The fix
  uses the messages.visibility column as the source of truth — runtime
  filters on visibility, storage writes the configured visibility, and
  ProbeRunner respects its probe_visibility setting.

These are integration tests against an in-memory-style temporary
SQLite database. No Ollama required. Stdlib unittest only.

Run with:
    python3 -m unittest sentinel.tests.test_probe_visibility -v
or:
    python3 -m unittest tests/test_probe_visibility.py -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Make the sentinel package importable when running this file directly.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sentinel.agent import Agent, AgentConfig
from sentinel.db import Database
from sentinel.runtime import ExperimentRuntime
from sentinel.probes import ProbeRunner


class _StubClient:
    """Stub OllamaClient — never makes a real request. Sufficient for the
    runtime/storage paths exercised by these tests."""

    def get_model_digest(self, model: str) -> str:
        return "stub-digest"

    def chat(self, *args, **kwargs):
        raise RuntimeError("StubClient.chat should not be called by these tests")

    def is_available(self) -> bool:
        return True


def _make_runtime_with_agent(probe_visibility: str):
    """Build a Database, ExperimentRuntime, and one Agent. Returns
    (tmpdir, db, runtime, agent). Caller cleans up tmpdir."""
    tmpdir = tempfile.mkdtemp(prefix="sentinel_test_pv_")
    db = Database(Path(tmpdir) / "test.db")
    client = _StubClient()

    experiment_id = db.create_experiment(
        name="probe_visibility_test",
        config={"topology": "full_mesh"},
    )
    agent_id = db.create_agent(
        experiment_id=experiment_id,
        name="alice",
        system_prompt="You are Alice.",
        model="stub:latest",
        model_digest="stub-digest",
    )

    runtime = ExperimentRuntime(db, client, experiment_id)
    runtime._probe_visibility = probe_visibility

    cfg = AgentConfig(name="alice", system_prompt="You are Alice.", model="stub:latest")
    agent = Agent(agent_id, cfg, client, "stub-digest")
    runtime.add_agent(agent)

    return tmpdir, db, runtime, agent


def _store_two_messages(db: Database, runtime: ExperimentRuntime, agent: Agent):
    """Insert one public conversation message and one hidden probe message."""
    db.store_message(
        experiment_id=runtime.experiment_id,
        agent_id=agent.agent_id,
        interaction_turn=1,
        content="Hello, this is a normal conversation message.",
        full_prompt="[]",
        model_digest="stub-digest",
        inference_ms=10,
        visibility="public",
    )
    db.store_message(
        experiment_id=runtime.experiment_id,
        agent_id=agent.agent_id,
        interaction_turn=2,
        content="[Probe Response] I am Alice and I value honesty.",
        full_prompt="[]",
        model_digest="stub-digest",
        inference_ms=10,
        visibility="hidden",
    )


def _cleanup(tmpdir: str, db: Database):
    db.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


class HiddenModeFiltersProbes(unittest.TestCase):
    """When _probe_visibility='hidden', _get_visible_messages must omit
    rows whose visibility column is 'hidden'."""

    def test_hidden_messages_are_filtered(self):
        tmpdir, db, runtime, agent = _make_runtime_with_agent("hidden")
        try:
            _store_two_messages(db, runtime, agent)
            visible = runtime._get_visible_messages(agent)
            self.assertEqual(
                len(visible), 1,
                f"expected 1 visible message after filtering, got {len(visible)}",
            )
            self.assertEqual(visible[0]["visibility"], "public")
            self.assertNotIn("[Probe Response]", visible[0]["content"])
        finally:
            _cleanup(tmpdir, db)


class PublicModeReturnsProbes(unittest.TestCase):
    """When _probe_visibility='public', _get_visible_messages must return
    everything — even rows whose visibility column is 'hidden' (which
    shouldn't exist in a public-mode run, but the runtime should not
    silently drop them either)."""

    def test_public_mode_returns_all_messages(self):
        tmpdir, db, runtime, agent = _make_runtime_with_agent("public")
        try:
            _store_two_messages(db, runtime, agent)
            visible = runtime._get_visible_messages(agent)
            self.assertEqual(
                len(visible), 2,
                f"expected 2 visible messages in public mode, got {len(visible)}",
            )
            contents = [m["content"] for m in visible]
            self.assertTrue(any("[Probe Response]" in c for c in contents))
        finally:
            _cleanup(tmpdir, db)


class ProbeRunnerStoresCorrectVisibility(unittest.TestCase):
    """ProbeRunner must persist its probe_visibility setting and use it
    when writing injected probe responses to the messages table. This
    catches the storage/runtime mismatch bug at the unit level."""

    def test_default_visibility_is_hidden(self):
        tmpdir, db, runtime, agent = _make_runtime_with_agent("hidden")
        try:
            runner = ProbeRunner(
                db=db,
                client=_StubClient(),
                experiment_id=runtime.experiment_id,
                mode="injected",
            )
            self.assertEqual(runner.probe_visibility, "hidden")
        finally:
            _cleanup(tmpdir, db)

    def test_explicit_public_visibility_is_honored(self):
        tmpdir, db, runtime, agent = _make_runtime_with_agent("public")
        try:
            runner = ProbeRunner(
                db=db,
                client=_StubClient(),
                experiment_id=runtime.experiment_id,
                mode="injected",
                probe_visibility="public",
            )
            self.assertEqual(runner.probe_visibility, "public")
        finally:
            _cleanup(tmpdir, db)

    def test_invalid_visibility_raises(self):
        with self.assertRaises(ValueError):
            ProbeRunner(
                db=None,  # never reached — validation runs before db touch
                client=_StubClient(),
                experiment_id="ignored",
                mode="injected",
                probe_visibility="banana",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
