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

"""SENTINEL Batch Runner — run a sequence of experiments from a JSON config.

Replaces per-milestone shell scripts with a single general-purpose runner.
Handles calibration, model switching, health checks, thermal guards, metrics,
and resume after interruption. SSH-safe (ignores SIGHUP).

Usage:
    python3 run_batch.py batch/m4_rerun.json              # fresh run
    python3 run_batch.py batch/m4_rerun.json --resume      # resume last batch
    python3 run_batch.py batch/m4_rerun.json --dry-run     # validate + show plan
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from sentinel.agent import Agent, AgentConfig
from sentinel.calibration import calibrate_all_agents, BATTERY_SIZE
from sentinel.control import create_control_experiment
from sentinel.db import Database
from sentinel.fork import fork_experiment
from sentinel.metrics import MetricsConfig, run_metrics_pipeline, print_metrics_summary
from sentinel.ollama import OllamaClient
from sentinel.persona import load_persona_config
from sentinel.probes import TriggerConfig
from sentinel.runtime import ExperimentRuntime, create_experiment
from sentinel.thermal import ThermalGuard

# Lazy imports for post-batch analysis (avoid circular at module level)
# from run_analyze import auto_analyze_and_save  # imported in run_post_batch_analysis()

log = logging.getLogger("sentinel.batch")


# ── Batch ID and logging ──────────────────────────────────────────

def generate_batch_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hex_suffix = os.urandom(3).hex()
    return f"{ts}_{hex_suffix}"


def setup_batch_logging(log_dir: Path, level: str = "INFO"):
    """Configure logging to write to batch log file."""
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # File handler for batch log
    fh = logging.FileHandler(log_dir / "batch.log")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Stderr handler (minimal — just errors)
    sh = logging.StreamHandler()
    sh.setLevel(logging.WARNING)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def setup_run_log(log_dir: Path, label: str) -> logging.FileHandler:
    """Add a per-run file handler. Returns it so caller can remove it later."""
    fh = logging.FileHandler(log_dir / f"run_{label}.log")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    return fh


# ── Batch state persistence ───────────────────────────────────────

def load_state(log_dir: Path) -> dict:
    state_path = log_dir / "batch_state.json"
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {}


def save_state(state: dict, log_dir: Path):
    with open(log_dir / "batch_state.json", "w") as f:
        json.dump(state, f, indent=2)


def update_status(log_dir: Path, msg: str):
    """Write one-line status to status.txt (for monitoring)."""
    status_path = log_dir.parent / "status.txt"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    batch_id = log_dir.name
    with open(status_path, "w") as f:
        f.write(f"[{ts}] [{batch_id}] {msg}\n")
    log.info("STATUS: %s", msg)


# ── Ollama orchestration ──────────────────────────────────────────

# Populated dynamically from Ollama on startup; used for unloading between runs.
KNOWN_MODELS = []


def unload_models(client: OllamaClient):
    """Unload all models from VRAM."""
    global KNOWN_MODELS
    # Populate dynamically if not already set by pre-flight check
    if not KNOWN_MODELS:
        try:
            from sentinel.models import discover_models
            KNOWN_MODELS = [m.name for m in discover_models(client)]
        except Exception:
            KNOWN_MODELS = ["gemma2:2b", "llama3.2:3b", "phi3:mini"]  # last-resort fallback
    log.info("Unloading models from VRAM...")
    for model in KNOWN_MODELS:
        try:
            data = json.dumps({"model": model, "keep_alive": 0}).encode()
            req = urllib.request.Request(
                f"{client.base_url}/api/generate",
                data=data, method="POST",
            )
            req.add_header("Content-Type", "application/json")
            urllib.request.urlopen(req, timeout=10)
        except (urllib.error.URLError, OSError):
            pass
    time.sleep(3)


def drop_caches():
    """Drop page caches to free unified memory."""
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, timeout=10,
        )
    except Exception as exc:
        log.warning("Could not drop caches: %s", exc)


def check_ollama_health(client: OllamaClient, model: str) -> bool:
    """Verify Ollama can generate with a model."""
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": "Say hello."}],
            temperature=0.0,
            num_predict=8,
        )
        return bool(resp.get("content"))
    except Exception as exc:
        log.warning("Health check failed for %s: %s", model, exc)
        return False


def recover_ollama(client: OllamaClient, model: str) -> bool:
    """3-level recovery: soft reset → service restart → give up."""
    # Level 1: soft reset
    log.info("Recovery level 1: soft reset")
    unload_models(client)
    drop_caches()
    time.sleep(5)
    if check_ollama_health(client, model):
        return True

    # Level 2: service restart
    log.info("Recovery level 2: restarting ollama service")
    try:
        subprocess.run(
            ["sudo", "systemctl", "restart", "ollama"],
            capture_output=True, timeout=30,
        )
    except Exception as exc:
        log.error("Failed to restart ollama: %s", exc)
        return False

    # Wait for Ollama to come back
    for _ in range(12):
        time.sleep(5)
        if client.is_available():
            break
    else:
        log.error("Ollama did not come back after restart")
        return False

    time.sleep(5)
    if check_ollama_health(client, model):
        return True

    # Level 3: give up (no auto-reboot)
    log.error("Recovery failed for %s — manual intervention needed", model)
    return False


async def ensure_ollama_ready(
    client: OllamaClient, model: str, label: str, thermal: ThermalGuard,
) -> bool:
    """Thermal check + unload + health check + recovery."""
    await thermal.check()
    unload_models(client)
    drop_caches()

    if check_ollama_health(client, model):
        return True

    log.warning("%s failed health check before %s — attempting recovery", model, label)
    return recover_ollama(client, model)


# ── Agent config building ─────────────────────────────────────────

def build_agent_configs(config: dict) -> list[AgentConfig]:
    """Build AgentConfig list from experiment config JSON."""
    agent_configs = []
    for agent_def in config["agents"]:
        name, traits, system_prompt = load_persona_config(agent_def)
        fingerprint = traits.fingerprint() if traits else None
        traits_json_str = json.dumps(agent_def["traits"]) if traits else None
        agent_configs.append(AgentConfig(
            name=name,
            system_prompt=system_prompt,
            model=agent_def.get("model", config.get("default_model", "llama3.2:3b")),
            temperature=agent_def.get("temperature", config.get("default_temperature", 0.7)),
            max_history=agent_def.get("max_history", config.get("default_max_history", 50)),
            response_limit=agent_def.get("response_limit", config.get("default_response_limit", 256)),
            is_control=False,
            traits_json=traits_json_str,
            trait_fingerprint=fingerprint,
        ))
    return agent_configs


def merge_config(defaults: dict, run_def: dict) -> dict:
    """Merge batch defaults with per-run overrides."""
    merged = dict(defaults)
    for key in ("max_turns", "delay", "probe", "probe_strategy", "probe_interval",
                "vocab_threshold", "sentiment_threshold", "metrics", "metrics_window"):
        if key in run_def:
            merged[key] = run_def[key]
    return merged


# ── Run type implementations ──────────────────────────────────────

async def on_message(turn: int, agent_name: str, content: str):
    preview = content[:80].replace("\n", " ")
    log.info("Turn %d | %s | %s", turn, agent_name, preview)


async def run_calibration(
    db: Database, client: OllamaClient, config: dict, agent_configs: list[AgentConfig],
) -> dict[str, str]:
    """Run calibration battery. Returns {agent_id: calibration_id}."""
    experiment_id = db.create_experiment(
        name=f"Calibration: {config['name']}",
        config=config,
        description=f"Calibration for {len(agent_configs)} agents",
    )

    agent_ids = []
    model_digests = []
    digest_cache = {}
    for ac in agent_configs:
        if ac.model not in digest_cache:
            digest_cache[ac.model] = client.get_model_digest(ac.model)
        digest = digest_cache[ac.model]
        model_digests.append(digest)
        agent_id = db.create_agent(
            experiment_id=experiment_id,
            name=ac.name,
            system_prompt=ac.system_prompt,
            model=ac.model,
            model_digest=digest,
            temperature=ac.temperature,
            max_history=ac.max_history,
            response_limit=ac.response_limit,
        )
        agent_ids.append(agent_id)

    db.update_experiment_status(experiment_id, "running")

    async def _cal_progress(msg):
        log.info(msg)

    cal_ids = await calibrate_all_agents(
        db=db, client=client,
        experiment_id=experiment_id,
        agent_configs=agent_configs,
        agent_ids=agent_ids,
        model_digests=model_digests,
        num_runs=3,
        on_progress=_cal_progress,
    )

    db.update_experiment_status(experiment_id, "completed")
    log.info("Calibration complete for %d agents", len(agent_configs))
    return cal_ids


async def run_single_experiment(
    db: Database, client: OllamaClient, run_def: dict, merged: dict,
) -> str:
    """Calibration + experiment. Returns experiment_id."""
    config_path = Path(run_def["config"])
    with open(config_path) as f:
        config = json.load(f)

    agent_configs = build_agent_configs(config)

    # Calibration
    log.info("Running calibration for %s...", run_def["label"])
    await run_calibration(db, client, config, agent_configs)

    # Build trigger config
    trigger_config = None
    if merged.get("probe_strategy") in ("triggered", "hybrid"):
        trigger_config = TriggerConfig(
            vocab_jsd_threshold=merged.get("vocab_threshold", 0.15),
            sentiment_threshold=merged.get("sentiment_threshold", 0.3),
        )

    # Create and run experiment
    log.info("Starting experiment for %s...", run_def["label"])
    runtime = create_experiment(
        db=db, client=client,
        name=config["name"],
        agent_configs=agent_configs,
        topology=config.get("topology", "full_mesh"),
        cycle_delay_s=merged.get("delay", 5.0),
        max_turns=merged.get("max_turns"),
        description=config.get("description", ""),
        probe_mode=merged.get("probe"),
        probe_interval=merged.get("probe_interval", 20),
        probe_strategy=merged.get("probe_strategy", "scheduled"),
        trigger_config=trigger_config,
    )

    await runtime.run(
        max_turns=merged.get("max_turns"),
        cycle_delay_s=merged.get("delay", 5.0),
        on_message=on_message,
    )
    return runtime.experiment_id


async def run_single_paired(
    db: Database, client: OllamaClient, run_def: dict, merged: dict,
) -> dict:
    """Calibration + paired (experimental + control). Returns {exp_id, ctrl_id}."""
    config_path = Path(run_def["config"])
    with open(config_path) as f:
        config = json.load(f)

    agent_configs = build_agent_configs(config)

    # Calibration
    log.info("Running calibration for %s...", run_def["label"])
    await run_calibration(db, client, config, agent_configs)

    trigger_config = None
    if merged.get("probe_strategy") in ("triggered", "hybrid"):
        trigger_config = TriggerConfig(
            vocab_jsd_threshold=merged.get("vocab_threshold", 0.15),
            sentiment_threshold=merged.get("sentiment_threshold", 0.3),
        )

    max_turns = merged.get("max_turns")
    delay = merged.get("delay", 5.0)
    result = {"exp_id": None, "ctrl_id": None}

    # Experimental arm
    log.info("Starting experimental arm for %s...", run_def["label"])
    try:
        exp_runtime = create_experiment(
            db=db, client=client,
            name=config["name"],
            agent_configs=agent_configs,
            topology=config.get("topology", "full_mesh"),
            cycle_delay_s=delay,
            max_turns=max_turns,
            description=config.get("description", ""),
            probe_mode=merged.get("probe"),
            probe_interval=merged.get("probe_interval", 20),
            probe_strategy=merged.get("probe_strategy", "scheduled"),
            trigger_config=trigger_config,
        )
        await exp_runtime.run(max_turns=max_turns, cycle_delay_s=delay, on_message=on_message)
        result["exp_id"] = exp_runtime.experiment_id
    except Exception as exc:
        log.error("Experimental arm failed: %s", exc)

    # Control arm
    log.info("Starting control arm for %s...", run_def["label"])
    try:
        ctrl_runtime = create_control_experiment(
            db=db, client=client,
            name=config["name"],
            agent_configs=agent_configs,
            cycle_delay_s=delay,
            max_turns=max_turns,
            description=config.get("description", ""),
            linked_experiment_id=result["exp_id"],
        )
        await ctrl_runtime.run(max_turns=max_turns, cycle_delay_s=delay, on_message=on_message)
        result["ctrl_id"] = ctrl_runtime.experiment_id
    except Exception as exc:
        log.error("Control arm failed: %s", exc)

    return result


async def run_single_fork(
    db: Database, client: OllamaClient, run_def: dict, merged: dict,
    source_experiment_id: str,
) -> str:
    """Fork from a prior experiment. Returns fork experiment_id."""
    at_turn = run_def.get("at_turn")
    mutations = run_def.get("mutate", [])

    # Parse mutations
    trait_overrides = None
    if mutations:
        overrides = {}
        for s in mutations:
            agent_name, trait_spec = s.split(":", 1)
            trait_dim, trait_val = trait_spec.split("=", 1)
            if agent_name not in overrides:
                overrides[agent_name] = {}
            overrides[agent_name][trait_dim] = trait_val
        trait_overrides = overrides

    suffix = run_def.get("suffix")
    if not suffix and trait_overrides:
        parts = []
        for name, changes in trait_overrides.items():
            for dim, val in changes.items():
                parts.append(f"{name}-{val}")
        suffix = "mutate-" + "-".join(parts)
    elif not suffix:
        suffix = "fork"

    log.info("Forking %s at turn %s...", source_experiment_id[:8], at_turn)
    fork_runtime = fork_experiment(
        db=db, client=client,
        source_experiment_id=source_experiment_id,
        fork_at_turn=at_turn,
        name_suffix=suffix,
        trait_overrides=trait_overrides,
    )

    max_turns = merged.get("max_turns", at_turn)
    await fork_runtime.run(
        max_turns=max_turns,
        cycle_delay_s=merged.get("delay", 5.0),
        on_message=on_message,
    )
    return fork_runtime.experiment_id


async def resume_experiment(
    db: Database, client: OllamaClient, experiment_id: str, merged: dict,
) -> str:
    """Resume an interrupted experiment from its last turn."""
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found in DB")

    current_turn, last_agent_id = db.get_resume_position(experiment_id)
    max_turns = merged.get("max_turns") or exp["max_turns"]
    last_name = "?"
    if last_agent_id:
        arow = db.conn.execute(
            "SELECT name FROM agents WHERE agent_id=?", (last_agent_id,),
        ).fetchone()
        if arow:
            last_name = arow["name"]
    log.info("Resuming %s from turn %d (last agent: %s) → %d", experiment_id[:8], current_turn, last_name, max_turns)

    if current_turn >= max_turns:
        log.info("Already at max turns, nothing to resume")
        return experiment_id

    runtime = ExperimentRuntime(db, client, experiment_id)
    agents_rows = db.get_agents(experiment_id)
    digest_cache = {}

    for row in agents_rows:
        model = row["model"]
        if model not in digest_cache:
            digest_cache[model] = client.get_model_digest(model)
        config = AgentConfig(
            name=row["name"],
            system_prompt=row["system_prompt"],
            model=model,
            temperature=row["temperature"],
            max_history=row["max_history"],
            response_limit=row["response_limit"],
            is_control=bool(row["is_control"]),
            traits_json=row["traits_json"],
            trait_fingerprint=row["trait_fingerprint"],
        )
        agent = Agent(row["agent_id"], config, client, digest_cache[model])
        runtime.add_agent(agent)

    await runtime.run(
        max_turns=max_turns,
        cycle_delay_s=merged.get("delay", 5.0),
        on_message=on_message,
    )
    return experiment_id


# ── Metrics ────────────────────────────────────────────────────────

def run_post_metrics(db: Database, client: OllamaClient, experiment_id: str, merged: dict):
    """Run metrics pipeline for a completed experiment."""
    if not merged.get("metrics"):
        return
    fast = merged["metrics"] == "fast"
    window = merged.get("metrics_window", 50)
    try:
        metrics_config = MetricsConfig(
            window_size=window,
            compute_semantic=not fast,
            compute_persona=not fast,
        )
        summary = run_metrics_pipeline(
            db, client, experiment_id, metrics_config,
            on_progress=lambda msg: log.info(msg),
        )
        log.info("Metrics complete for %s", experiment_id[:8])
    except Exception as exc:
        log.error("Metrics failed for %s: %s", experiment_id[:8], exc)


# ── Post-batch analysis ───────────────────────────────────────────

def run_post_batch_analysis(
    db: Database, state: dict, batch_config: dict, log_dir: Path,
):
    """Run diff → analyze → auto-findings for all configured comparisons.

    The batch config can include a "post_batch" section:
    {
        "post_batch": {
            "analyze_each": true,       // analyze each experiment individually
            "auto_findings": true,      // auto-generate findings (vs templates)
            "comparisons": [
                {"type": "cross-model", "a": "gemma-baseline", "b": "llama-baseline"},
                {"type": "paired", "label": "paired-gemma"},
                {"type": "fork", "a": "gemma-baseline", "b": "fork-skeptical"},
            ]
        }
    }
    """
    from run_analyze import auto_analyze_and_save, analyze_single, analyze_comparison, print_report

    post = batch_config.get("post_batch")
    if not post:
        return

    log.info("")
    log.info("=" * 50)
    log.info("POST-BATCH ANALYSIS")
    log.info("=" * 50)
    update_status(log_dir, "Post-batch analysis — starting")

    auto_findings = post.get("auto_findings", True)
    runs_state = state.get("runs", {})

    def get_exp_id(label: str) -> str | None:
        rs = runs_state.get(label, {})
        return rs.get("experiment_id") if rs.get("status") == "completed" else None

    # Individual experiment analysis
    if post.get("analyze_each", False):
        log.info("")
        log.info("─── Analyzing individual experiments ───")
        for label, rs in runs_state.items():
            if rs.get("status") != "completed" or not rs.get("experiment_id"):
                continue
            exp_id = rs["experiment_id"]
            try:
                log.info("Analyzing %s (%s)...", label, exp_id[:8])
                auto_analyze_and_save(db, [exp_id], auto_finding=auto_findings, quiet=True)
                log.info("  Analysis complete for %s", label)

                # Also analyze control arm if this was a paired run
                ctrl_id = rs.get("ctrl_id")
                if ctrl_id:
                    log.info("Analyzing control arm for %s (%s)...", label, ctrl_id[:8])
                    auto_analyze_and_save(db, [ctrl_id], auto_finding=auto_findings, quiet=True)
            except Exception as exc:
                log.error("Analysis failed for %s: %s", label, exc)

    # Comparison analysis
    comparisons = post.get("comparisons", [])
    if comparisons:
        log.info("")
        log.info("─── Running comparison analyses ───")

    for comp in comparisons:
        comp_type = comp.get("type", "unknown")
        try:
            if comp_type == "paired":
                # Paired: compare experimental vs control from the same run
                label = comp.get("label") or comp.get("a")
                if not label:
                    log.warning("Paired comparison missing 'label': %s", comp)
                    continue
                exp_id = get_exp_id(label)
                ctrl_id = runs_state.get(label, {}).get("ctrl_id")
                if not exp_id or not ctrl_id:
                    log.warning("Paired comparison skipped — missing IDs for %s", label)
                    continue
                log.info("Paired diff: %s exp=%s ctrl=%s", label, exp_id[:8], ctrl_id[:8])
                auto_analyze_and_save(db, [exp_id, ctrl_id], auto_finding=auto_findings, quiet=True)
                log.info("  Paired analysis complete")

            elif comp_type == "fork":
                # Fork: compare source vs fork
                a_label = comp.get("a")
                b_label = comp.get("b")
                if not a_label or not b_label:
                    log.warning("Fork comparison missing 'a' or 'b': %s", comp)
                    continue
                a_id = get_exp_id(a_label)
                b_id = get_exp_id(b_label)
                if not a_id or not b_id:
                    log.warning("Fork comparison skipped — missing IDs (a=%s, b=%s)", a_label, b_label)
                    continue
                log.info("Fork diff: %s (%s) vs %s (%s)", a_label, a_id[:8], b_label, b_id[:8])
                auto_analyze_and_save(db, [a_id, b_id], auto_finding=auto_findings, quiet=True)
                log.info("  Fork analysis complete")

            elif comp_type in ("cross-model", "comparison"):
                # Generic comparison: diff A vs B by label
                a_label = comp.get("a")
                b_label = comp.get("b")
                if not a_label or not b_label:
                    log.warning("Comparison missing 'a' or 'b': %s", comp)
                    continue
                a_id = get_exp_id(a_label)
                b_id = get_exp_id(b_label)
                if not a_id or not b_id:
                    log.warning("Comparison skipped — missing IDs (a=%s, b=%s)", a_label, b_label)
                    continue
                log.info("Cross-model diff: %s (%s) vs %s (%s)", a_label, a_id[:8], b_label, b_id[:8])
                auto_analyze_and_save(db, [a_id, b_id], auto_finding=auto_findings, quiet=True)
                log.info("  Cross-model analysis complete")

            else:
                log.warning("Unknown comparison type: %s", comp_type)

        except Exception as exc:
            log.error("Comparison analysis failed (%s): %s", comp, exc)

    update_status(log_dir, "Post-batch analysis — complete")
    log.info("Post-batch analysis complete")


# ── Main ──────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="SENTINEL Batch Runner")
    parser.add_argument("config", help="Path to batch config JSON")
    parser.add_argument("--resume", action="store_true",
                        help="Resume last batch (skip completed runs)")
    parser.add_argument("--batch-id", default=None,
                        help="Specific batch ID to resume (with --resume)")
    parser.add_argument("--db", default="experiments/sentinel.db")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and show plan")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # Load batch config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Batch config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as f:
        batch_config = json.load(f)

    defaults = batch_config.get("defaults", {})
    runs = batch_config.get("runs", [])

    # Validate labels are unique
    labels = [r["label"] for r in runs]
    if len(labels) != len(set(labels)):
        print("Error: duplicate labels in batch config", file=sys.stderr)
        sys.exit(1)

    # Validate fork references
    for run_def in runs:
        if run_def["type"] == "fork":
            src = run_def.get("source_label")
            if not src or src not in labels:
                print(f"Error: fork '{run_def['label']}' references unknown source_label '{src}'",
                      file=sys.stderr)
                sys.exit(1)
            if labels.index(src) >= labels.index(run_def["label"]):
                print(f"Error: fork '{run_def['label']}' must come after source '{src}'",
                      file=sys.stderr)
                sys.exit(1)

    # Validate experiment configs exist
    for run_def in runs:
        if "config" in run_def:
            if not Path(run_def["config"]).exists():
                print(f"Error: config not found: {run_def['config']}", file=sys.stderr)
                sys.exit(1)

    # Pre-flight model compatibility check
    from sentinel.models import (
        get_system_specs, get_compatible_models, discover_models,
        check_model_compatibility,
    )
    client = OllamaClient()
    if client.is_available():
        specs = get_system_specs()
        compatibilities = get_compatible_models(client, specs)
        compat_map = {c.model.name: c for c in compatibilities}
        # Populate KNOWN_MODELS from discovered models
        global KNOWN_MODELS
        KNOWN_MODELS = [c.model.name for c in compatibilities]

        # Check each run's model
        batch_models = set()
        for run_def in runs:
            model = run_def.get("model", defaults.get("model", "gemma2:2b"))
            batch_models.add(model)

        bad_models = []
        for model in sorted(batch_models):
            c = compat_map.get(model)
            if c is None:
                bad_models.append((model, "not found in Ollama — pull it first"))
            elif not c.fits:
                bad_models.append((model, c.reason))

        if bad_models:
            print(f"\nModel compatibility issues on {specs.gpu_name}:", file=sys.stderr)
            print(f"  Available VRAM: {specs.max_model_vram_gb} GB", file=sys.stderr)
            for model, reason in bad_models:
                print(f"  [NO] {model}: {reason}", file=sys.stderr)
            print(f"\nRun 'python3 list_models.py' to see compatible models.", file=sys.stderr)
            sys.exit(1)

        # Warn about tight fits
        for model in sorted(batch_models):
            c = compat_map.get(model)
            if c and c.fits and c.margin_gb < 0.5:
                print(f"  Warning: {model} is a tight fit ({c.margin_gb:+.1f} GB headroom)",
                      file=sys.stderr)

    # Dry run
    if args.dry_run:
        print(f"\nBatch: {batch_config.get('name', config_path.stem)}")
        print(f"Runs:  {len(runs)}\n")
        for i, run_def in enumerate(runs, 1):
            merged = merge_config(defaults, run_def)
            extra = ""
            if run_def["type"] == "fork":
                extra = f" (from {run_def['source_label']} @t{run_def.get('at_turn', '?')})"
            elif run_def["type"] == "paired":
                extra = " (experimental + control)"
            print(f"  {i}. [{run_def['type']:10s}] {run_def['label']}{extra}")
            print(f"     model={run_def.get('model', '?')} turns={merged.get('max_turns', '?')} "
                  f"probe={merged.get('probe', 'none')}")
        print("\nConfig valid. Remove --dry-run to execute.")
        return

    # Determine batch ID and log dir
    logs_base = Path("experiments/logs")
    logs_base.mkdir(parents=True, exist_ok=True)

    if args.resume:
        if args.batch_id:
            batch_id = args.batch_id
        else:
            # Find latest batch for this config
            state_files = sorted(logs_base.glob("*/batch_state.json"))
            found = None
            for sf in reversed(state_files):
                # Skip symlinks (e.g. "latest") to avoid circular symlink bug
                if sf.parent.is_symlink():
                    continue
                state = json.loads(sf.read_text())
                if state.get("config_file") == str(config_path):
                    found = sf.parent.name
                    break
            if not found:
                print("No previous batch found to resume. Run without --resume.", file=sys.stderr)
                sys.exit(1)
            batch_id = found
        log_dir = logs_base / batch_id
        if not log_dir.exists():
            print(f"Batch directory not found: {log_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        batch_id = generate_batch_id()
        log_dir = logs_base / batch_id

    setup_batch_logging(log_dir, args.log_level)

    # Symlink latest
    latest = logs_base / "latest"
    try:
        latest.unlink(missing_ok=True)
    except (OSError, TypeError):
        try:
            latest.unlink()
        except OSError:
            pass
    try:
        latest.symlink_to(batch_id)
    except OSError:
        pass

    # Load or init state
    state = load_state(log_dir)
    if not state:
        state = {
            "batch_id": batch_id,
            "config_file": str(config_path),
            "batch_name": batch_config.get("name", config_path.stem),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "runs": {},
        }
        save_state(state, log_dir)

    # Print launch info to stdout
    print(f"SENTINEL Batch: {batch_config.get('name', config_path.stem)}")
    print(f"Batch ID: {batch_id}")
    print(f"Runs: {len(runs)}")
    print(f"Monitor: tail -f {log_dir}/batch.log")
    print(f"Status:  cat {logs_base}/status.txt")

    log.info("=" * 50)
    log.info("SENTINEL Batch: %s", batch_config.get("name", config_path.stem))
    log.info("Batch ID: %s", batch_id)
    log.info("Config:   %s", config_path)
    log.info("Runs:     %d", len(runs))
    if args.resume:
        completed = sum(1 for r in state.get("runs", {}).values() if r.get("status") == "completed")
        partial = sum(1 for r in state.get("runs", {}).values() if r.get("status") == "partial")
        log.info("Resume mode: %d/%d completed, %d partial (will re-run)", completed, len(runs), partial)
    log.info("=" * 50)

    # Connect
    db = Database(args.db)
    client = OllamaClient()
    thermal = ThermalGuard()

    try:
        for i, run_def in enumerate(runs, 1):
            label = run_def["label"]
            run_type = run_def["type"]
            model = run_def.get("model", "gemma2:2b")
            merged = merge_config(defaults, run_def)
            run_state = state.get("runs", {}).get(label, {})

            # Skip completed runs
            if run_state.get("status") == "completed":
                log.info("Skipping completed run %d/%d: %s", i, len(runs), label)
                continue

            # Skip forks whose source failed
            if run_type == "fork":
                src_label = run_def["source_label"]
                src_state = state.get("runs", {}).get(src_label, {})
                if src_state.get("status") not in ("completed", "partial") or not src_state.get("experiment_id"):
                    log.error("Skipping fork '%s' — source '%s' not completed", label, src_label)
                    state.setdefault("runs", {})[label] = {
                        "status": "skipped",
                        "error": f"source '{src_label}' not available",
                    }
                    save_state(state, log_dir)
                    continue

            log.info("")
            log.info("─── Run %d/%d: %s [%s] ───", i, len(runs), label, run_type)
            update_status(log_dir, f"Run {i}/{len(runs)}: {label} ({run_type}) — starting")

            # Add per-run log handler
            run_fh = setup_run_log(log_dir, label)

            # Ensure Ollama healthy
            if not await ensure_ollama_ready(client, model, label, thermal):
                log.error("Ollama recovery failed for %s — skipping", label)
                state.setdefault("runs", {})[label] = {
                    "status": "failed",
                    "error": "ollama_recovery_failed",
                }
                save_state(state, log_dir)
                logging.getLogger().removeHandler(run_fh)
                continue

            # Check if resuming an interrupted run
            experiment_id = None
            try:
                if run_state.get("status") == "running" and run_state.get("experiment_id"):
                    # Resume interrupted experiment
                    log.info("Resuming interrupted experiment %s", run_state["experiment_id"][:8])
                    experiment_id = await resume_experiment(
                        db, client, run_state["experiment_id"], merged,
                    )
                elif run_type == "experiment":
                    experiment_id = await run_single_experiment(db, client, run_def, merged)
                elif run_type == "paired":
                    result = await run_single_paired(db, client, run_def, merged)
                    experiment_id = result.get("exp_id")
                    ctrl_id = result.get("ctrl_id")
                    # Both arms must succeed for paired run to be "completed"
                    if experiment_id and ctrl_id:
                        paired_status = "completed"
                    elif experiment_id:
                        paired_status = "partial"
                        log.warning("Paired run %s: control arm failed — marking partial", label)
                    else:
                        paired_status = "failed"
                        log.error("Paired run %s: experimental arm failed", label)
                    state.setdefault("runs", {})[label] = {
                        "status": paired_status,
                        "experiment_id": experiment_id,
                        "ctrl_id": ctrl_id,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }
                elif run_type == "fork":
                    source_exp_id = state["runs"][run_def["source_label"]]["experiment_id"]
                    experiment_id = await run_single_fork(
                        db, client, run_def, merged, source_exp_id,
                    )

                # Update state (paired already set above)
                if run_type != "paired":
                    state.setdefault("runs", {})[label] = {
                        "status": "completed",
                        "experiment_id": experiment_id,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    }

                save_state(state, log_dir)
                log.info("Run %s complete: %s", label, experiment_id[:8] if experiment_id else "?")
                update_status(log_dir, f"Run {i}/{len(runs)}: {label} — DONE ({experiment_id[:8] if experiment_id else '?'})")

                # Post-run metrics
                if experiment_id:
                    run_post_metrics(db, client, experiment_id, merged)

            except Exception as exc:
                log.error("Run %s failed: %s", label, exc, exc_info=True)
                state.setdefault("runs", {})[label] = {
                    "status": "failed",
                    "experiment_id": experiment_id,
                    "error": str(exc),
                }
                save_state(state, log_dir)
                update_status(log_dir, f"Run {i}/{len(runs)}: {label} — FAILED: {exc}")

            # Remove per-run log handler
            logging.getLogger().removeHandler(run_fh)
            run_fh.close()

        # Post-batch analysis pipeline (diff → analyze → findings)
        if batch_config.get("post_batch"):
            try:
                run_post_batch_analysis(db, state, batch_config, log_dir)
            except Exception as exc:
                log.error("Post-batch analysis failed: %s", exc, exc_info=True)

    finally:
        db.close()

    # Summary
    log.info("")
    log.info("=" * 50)
    log.info("SENTINEL Batch Complete: %s", batch_id)
    log.info("=" * 50)
    for run_def in runs:
        label = run_def["label"]
        rs = state.get("runs", {}).get(label, {})
        status = rs.get("status", "not_run")
        exp_id = rs.get("experiment_id", "")
        exp_str = exp_id[:8] if exp_id else ""
        log.info("  %-20s  %-10s  %s", label, status, exp_str)

    update_status(log_dir, "Batch complete")

    # Print summary to stdout too
    print(f"\nBatch {batch_id} complete.")
    for run_def in runs:
        label = run_def["label"]
        rs = state.get("runs", {}).get(label, {})
        status = rs.get("status", "not_run")
        exp_id = rs.get("experiment_id", "")
        print(f"  {label}: {status} {exp_id[:8] if exp_id else ''}")


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    asyncio.run(main())
