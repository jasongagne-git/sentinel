# SENTINEL

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19032840.svg)](https://doi.org/10.5281/zenodo.19032840)

**Systematic Evaluation Network for Testing Intelligent Agent Limits**

An empirical research framework for measuring behavioral drift in multi-agent LLM systems. SENTINEL subjects groups of LLM agents to sustained multi-party conversation and measures their behavioral trajectories over hundreds of interaction turns, using calibrated baselines, controlled experimental conditions, and a dual-probe methodology.

## Key Findings

From 40+ experiments across two model families (gemma2:2b, llama3.2:3b), totaling 20,000+ agent messages:

- **Agent collapse is triggered, not spontaneous.** 0/12 baseline runs collapse; 5/6 mutation-fork runs collapse at a deterministic turn. Collapse requires a perturbation -- it is not inevitable degradation.
- **Pre-collapse thinning gradient.** Agents that will collapse show measurable output shrinkage (-2 to -5 chars/turn) before the event -- a potential early-warning signal.
- **Three post-collapse modes.** Cascade (40%), isolation (20%), and compensatory expansion (40%) -- surviving agents may shrink, hold steady, or *grow* to fill the void.
- **Probe-conversation dissociation.** Collapsed agents continue passing identity probes. Probe drift scores are flat across 5,760 measurements -- probes measure identity recall, not behavioral drift.
- **Hollow verbosity.** Agents under mutation stress that avoid collapse enter repetitive loops (78% of messages repeating the same phrase), maintaining output volume while losing substance.
- **Model-dependent observer effects.** The same probing protocol is neutral on gemma2:2b but drift-suppressing on llama3.2:3b. Measurement calibration must be per-model.

Full details: *Behavioral Drift in Multi-Agent LLM Systems: Emergent Failure Modes, Cascade Dynamics, and Measurement Challenges* (preprint forthcoming)

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with at least one model pulled (e.g., `ollama pull gemma2:2b`)
- No pip dependencies -- stdlib only (asyncio, sqlite3, json, argparse, curses)

## Quick Start

```bash
# Check available models and hardware compatibility
python3 list_models.py

# List available persona trait dimensions
python3 list_traits.py

# Run a baseline experiment (3-agent conversation, 200 turns)
python3 run_experiment.py config/long_run_baseline.json --max-turns 200

# With probes (shadow + injected, hybrid trigger strategy)
python3 run_experiment.py config/long_run_baseline.json --max-turns 200 \
    --probe both --probe-strategy hybrid

# Run a paired experiment (multi-agent + isolated control arm)
python3 run_paired.py config/long_run_baseline.json --max-turns 200

# Run a fork experiment (mutate Beck's disposition at turn 25)
python3 run_fork.py -e <experiment-id> --at-turn 25 --mutate disposition=skeptical

# Analyze an experiment
python3 run_analyze.py -e <experiment-id>

# Analyze all experiments with auto-finding generation
python3 run_analyze.py --all --auto-finding

# Compare two experiments
python3 run_diff.py -a <exp-a> -b <exp-b>

# Live monitoring dashboard
python3 run_monitor.py
```

## Experiment Lifecycle

1. **Configure** -- Define agents, topology, and timing in a JSON config
2. **Calibrate** -- Run calibration battery (4 categories x 3 prompts x 3 runs) to establish per-agent baselines
3. **Run** -- Async round-robin conversation loop with optional probing and thermal guards
4. **Validate** -- Post-run checks on data completeness and turn counts
5. **Analyze** -- Compute metrics, run 10 automated pattern detectors, generate findings
6. **Compare** -- Diff against control arms, forks, or cross-model runs

## Experimental Design

SENTINEL supports four experimental paradigms:

1. **Baseline** -- Multi-agent conversation with periodic probing. Repeated runs assess reproducibility and establish behavioral variance.

2. **Paired** -- Experimental arm (multi-agent) + control arm (isolated agents responding to fixed prompts). Isolates interaction effects from single-agent context drift.

3. **Fork** -- Clone experiment state at a turn, mutate one variable, continue. Enables causal attribution of behavioral changes and path-dependence analysis.

4. **Cross-model** -- Identical configs on different model families to assess model-dependent effects.

### Agent Personas

Each agent is defined by six trait dimensions, composed deterministically into a system prompt:

| Dimension | Example values |
|-----------|---------------|
| Role | policy_analyst, innovator, ethics_researcher, security_expert, ... |
| Disposition | pragmatic, optimistic, cautious, analytical, skeptical, ... |
| Values | stability, freedom, equity, security, transparency, ... |
| Stance on regulation | minimal, moderate, proactive, adaptive, sector_specific |
| Communication style | concise, detailed, socratic, narrative, assertive, diplomatic |
| Conflict approach | debate, bridge, defer, challenge, reframe |

Default three-agent setup:

| Agent | Role | Disposition | Values | Conflict |
|-------|------|-------------|--------|----------|
| Aria | Policy analyst | Pragmatic | Stability | Defer |
| Beck | Innovator | Optimistic | Freedom | Debate |
| Cass | Ethics researcher | Cautious | Equity | Reframe |

See `python3 list_traits.py` for all options.

### Probe Methodology

**Two probe modes** enable measurement of observer effects:

- **Shadow probes** -- Separate inference calls that do not enter the conversation context. Approximate passive observation.
- **Injected probes** -- Probe responses stored as conversation messages. Measure how probing itself alters behavior.

**Three trigger strategies:**

- **Scheduled** -- Fire probes at fixed intervals (every N turns)
- **Triggered** -- Fire when lightweight metrics (vocabulary JSD, sentiment) cross configurable thresholds
- **Hybrid** -- Scheduled probes at wide intervals + triggered probes on threshold crossings

```bash
python3 run_experiment.py config/... \
    --probe both \
    --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 \
    --sentiment-threshold 0.25
```

### Batch Configs

Batch configs (`batch/*.json`) define multi-run experiments:

```json
{
  "name": "Experiment Batch",
  "defaults": { "max_turns": 200, "probe_mode": "both" },
  "runs": [
    { "label": "baseline-01", "type": "experiment", "config": "config/long_run_baseline.json" },
    { "label": "fork-01", "type": "fork", "source_label": "baseline-01",
      "at_turn": 25, "mutate": ["disposition=skeptical"] }
  ],
  "post_batch": { "analyze_each": true, "auto_findings": true }
}
```

Batches support resume from interruption (`--resume`), dry-run validation (`--dry-run`), model switching between runs, and automated post-batch analysis.

## Detection Pipeline

`run_analyze.py` implements 10 automated detectors:

| Detector | What it measures |
|----------|-----------------|
| Collapse | Sudden cessation of output (near-zero message length) |
| Convergence | Agents' outputs becoming indistinguishable |
| Vocabulary drift | Jensen-Shannon divergence of word frequency distributions |
| Sentiment drift | Directional shift in sentiment polarity over time |
| Hollow verbosity | Output length increasing while vocabulary diversity decreases |
| Probe contamination | Agents echoing probe identity markers in conversation |
| Dissociation gap | Divergence between shadow and injected probe measurements |
| Content repetition | N-gram overlap between early and late messages |
| Context saturation | Prompt token accumulation relative to collapse timing |
| Cascade propagation | Output degradation spreading from one agent to others |

Detectors are configurable via `config/detection_patterns.json` (thresholds, severity levels, pattern implications).

## Metrics

Four core E-ASI dimensions computed per agent over sliding windows:

| Metric | Method | Range |
|--------|--------|-------|
| Vocabulary drift | Jensen-Shannon divergence of token frequency distributions | [0, 1] |
| Sentiment trajectory | Lexicon-based polarity scoring (40 positive + 40 negative terms) | [-1, 1] |
| Semantic coherence | Cosine similarity of message embeddings to system prompt (via Ollama) | [0, 1] |
| Persona adherence | LLM-as-judge scoring against persona definition | [1, 5] |

See `docs/Metrics_Reference.md` for detailed algorithms.

## Architecture

```
sentinel/              # Core library (stdlib-only Python)
  agent.py             # Agent runtime and message handling
  calibration.py       # Pre-experiment calibration battery
  control.py           # Control arm (isolated agent) runner
  db.py                # SQLite schema and data access
  diff.py              # Cross-experiment comparison engine
  fork.py              # Fork/clone experiment state
  metrics.py           # Vocabulary drift, sentiment, output analysis
  models.py            # Model compatibility and hardware checks
  monitor.py           # Curses live dashboard
  ollama.py            # Ollama API client
  persona.py           # Trait dimensions and persona generation
  probes.py            # Shadow and injected probe subsystem
  runtime.py           # Async experiment orchestration
  thermal.py           # Thermal throttle guard (Jetson)
```

### CLI Tools

| Command | Purpose |
|---------|---------|
| `run_experiment.py` | Run a multi-agent conversation experiment |
| `run_calibration.py` | Run calibration battery only |
| `run_paired.py` | Run paired experiment + control |
| `run_fork.py` | Fork an experiment with trait mutations |
| `run_batch.py` | Orchestrate multi-run batch from JSON config |
| `run_resume.py` | Resume an interrupted experiment |
| `run_analyze.py` | Automated pattern detection and finding generation |
| `run_diff.py` | Compare two experiments |
| `run_metrics.py` | Compute metrics for a completed experiment |
| `run_findings.py` | Browse and manage the findings knowledge base |
| `run_monitor.py` | Curses live dashboard |
| `run_matrix.py` | Generate cross-model experiment matrices |
| `list_models.py` | List Ollama models with hardware compatibility |
| `list_traits.py` | List persona trait dimensions and values |

### Infrastructure

| Script | Purpose |
|--------|---------|
| `recover_ollama.sh` | Recover Ollama from stuck states |
| `thermal_guard.sh` | Monitor thermals during long runs |
| `check_services.sh` | Verify Ollama and system health |

Man pages are available in `man/man1/`.

## Data Storage

All experiment data is stored in a SQLite database (`experiments/sentinel.db`, WAL mode):

| Table | Contents |
|-------|----------|
| `experiments` | Metadata, config, fork lineage, status |
| `agents` | Configs, system prompts, trait fingerprints, calibration refs |
| `messages` | All agent messages with turn numbers, token counts, timing |
| `probes` | Probe prompts, responses, drift scores, mode (shadow/injected) |
| `calibration_results` | Pre-experiment baseline measurements |
| `metrics` | Computed metric values per agent per window |

Findings are stored as structured JSON in `findings/`.

## Distributed Mode

For experiments at scale (10+ agents across multiple nodes), SENTINEL extends into a distributed cluster via three companion repositories:

| Repository | Purpose |
|------------|---------|
| [sentinel-coordinator](../sentinel-coordinator) | Cluster orchestration, distributed runtime, failure management |
| [sentinel-worker](../sentinel-worker) | HTTP inference service running on each node |
| [sentinel-common](../sentinel-common) | Shared auth, config models, Ollama client, thermal, validation |

The coordinator extends the core database schema with cluster event tracking and worker snapshots, and adds six distributed-specific pattern detectors on top of the 10 single-node detectors.

## Hardware

Developed and tested on NVIDIA Jetson Orin Nano (8GB unified RAM, CUDA 12.6). The framework includes Jetson-specific optimizations:

- Thermal throttle guard for sustained batch runs
- CUDA memory management for unified memory architecture
- Model switching with explicit unload and page cache flush
- NvMap-aware compatibility checks

The framework should work on any system running Ollama, but hardware compatibility checks and thermal monitoring are Jetson-specific.

## Data

Experiment data and findings from the published research are available as release artifacts:

- **Experiment database** -- Full SQLite database with 40+ experiments, 20,000+ messages, and probe measurements
- **Findings knowledge base** -- 89 structured findings (F-0001 through F-0089) from automated and manual analysis

Download from [GitHub Releases](https://github.com/jasongagne-git/sentinel/releases) or cite via [DOI: 10.5281/zenodo.19032840](https://doi.org/10.5281/zenodo.19032840).

## License

Copyright 2026 Jason Gagne

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@article{gagne2026behavioral,
  title={Behavioral Drift in Multi-Agent LLM Systems: Emergent Failure Modes,
         Cascade Dynamics, and Measurement Challenges},
  author={Gagne, Jason},
  year={2026},
  note={arXiv preprint}
}
```

## Related Work

- Gagne, J. (2026). "The Behavioral Sufficiency Problem." SSRN Working Paper.
- Rath, A. (2026). "Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems." arXiv:2601.04170.
- Becker, J., et al. (2025). "Stay Focused: Problem Drift in Multi-Agent Debate." arXiv:2502.19559.
- Chen, R., et al. (2025). "Persona Vectors." arXiv:2507.21509.
