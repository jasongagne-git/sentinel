# SENTINEL Experiment Scenarios

A catalog of experiment designs and the research questions they address.
Each scenario describes a configuration, the command to run it, and what to
look for in the results.

> **Note:** Some scenarios reference config files (e.g., `config/three_agent_mesh.json`)
> that are not included in the repository. These serve as design templates — create
> them by copying `config/long_run_baseline.json` and modifying the agent definitions,
> model, or parameters as described. The included configs (`config/long_run_baseline.json`,
> `config/long_run_llama3.json`) and batch configs (`batch/*.json`) cover the experiments
> from the published research.

---

## Probe Methodology: Multi-Pronged Approach

**Lesson from Milestone 4 (500-turn baseline):** Shadow probes alone are
insufficient. In the first long-run experiment, all three agents maintained
perfect persona identity under shadow probing while their conversational
behavior collapsed into vocabulary-depleted agreement rituals. One agent
(Beck) produced 448 consecutive empty messages yet gave articulate, on-brand
probe responses throughout. The drift signal is in the *dissociation*
between probe identity and conversational behavior, not in either measure
alone.

**All future experiments MUST use the following multi-pronged probe approach:**

### 1. Shadow + Injected Probes (mode: `both`)

Run shadow and injected probes simultaneously. Shadow probes measure
preserved identity in isolation; injected probes test whether the agent can
express identity *within the group context*. If injected probe responses
degrade while shadow probe responses hold, this confirms context-dependent
drift.

### 2. Conversational Probes (new category)

In addition to the 4 identity probes (persona, values, stance, disposition),
add **conversation-contextual probes** that require agents to activate their
persona against recent context:
- "What did you disagree with in the recent discussion?"
- "What's your strongest objection to what was just said?"
- "How does the current conversation align with your core values?"

These detect agents that can recite their identity but no longer perform it.

### 3. Triggered Probes (mandatory for long runs)

Use `--probe-strategy hybrid` for any experiment over 100 turns. The
DriftMonitor's lightweight per-turn JSD and sentiment tracking fires probes
at transition points that scheduled probes miss. The Milestone 4 run used
only scheduled probes and missed Beck's exact collapse point (somewhere
between t50-t53).

### 4. Drift Score Computation

Always run `run_metrics.py` after experiments to compute probe drift scores
(early vs late response similarity). The `drift_score` column must be
populated — NULL drift scores provide no quantitative basis for comparison.

### 5. Behavioral Metrics Alongside Probes

Probes alone can be misleading. Always pair probe data with conversational
metrics:
- Content length trajectory (detects output collapse)
- Vocabulary size over time (detects vocabulary shrinkage)
- Emoji/exclamation frequency (detects tone convergence)
- Unique word count per window (detects loss of distinctiveness)

**Standard probe invocation for future experiments:**

```bash
python3 run_experiment.py <config> \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns <N> --delay 5
```

---

## 1. Baseline Drift (3-Agent Full Mesh)

**Research question**: Do agents drift from their assigned personas during
multi-agent interaction, and how quickly?

**Config**: `config/three_agent_mesh.json`

```bash
# Calibrate first (establishes baselines)
python3 run_calibration.py config/three_agent_mesh.json

# Run experiment
python3 run_experiment.py config/three_agent_mesh.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5

# Compute metrics
python3 run_metrics.py
```

**What to look for**:
- Vocabulary drift (JSD) trend over time — does it plateau or keep climbing?
- Persona adherence scores — do agents start "forgetting" who they are?
- Sentiment convergence — do distinct agents start sounding the same?

---

## 2. Paired Experiment (Interaction vs Isolation)

**Research question**: Is observed drift caused by multi-agent interaction,
or does it happen even in isolation (single-agent context drift)?

**Config**: `config/three_agent_mesh.json`

```bash
python3 run_calibration.py config/three_agent_mesh.json

python3 run_paired.py config/three_agent_mesh.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5
```

**What to look for**:
- Compare drift between experimental (interacting) and control (isolated) arms
- Drift present in experimental but not control = multi-agent interaction effect
- Drift present in both = inherent model behavior, not interaction-driven

---

## 3. Long-Run Drift (500-1000+ Interactions)

**Research question**: What does the drift curve look like at scale? Is there
a saturation point, or does drift accelerate?

**Config**: `config/long_run_baseline.json`

```bash
python3 run_calibration.py config/long_run_baseline.json

python3 run_experiment.py config/long_run_baseline.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 500 --delay 5
```

**Milestone 4 findings (500-turn gemma2:2b run)**:
- Vocabulary collapse by t100: 263→73 unique words (Aria), 252→88 (Cass)
- Beck (assertive/debate style) collapsed to empty output at t53; silent for
  remaining 448 turns. Shadow probes showed intact persona throughout.
- Aria and Cass converged to short emoji-laden agreement rituals (~55 chars/msg)
- Sentiment *declined* despite surface enthusiasm (loss of substantive content)
- Probe-conversation dissociation: all agents maintained identity under probing
  while exhibiting severe behavioral drift in conversation

**Milestone 5 findings (multi-model fork matrix, 9 runs)**:
- **Mutation-triggered collapse**: Beck collapses at turn 53 in fork experiments
  where a trait mutation is applied. Collapse is deterministic in timing (always
  t53) but requires a perturbation trigger — see M6 findings below.
- **Cascade propagation**: One agent's collapse can trigger thinning or collapse
  in others. The cascade is stochastic (timing and severity vary across runs)
  while the individual collapse itself is deterministic.
- **Model stochasticity**: gemma2:2b shows 60-70% coefficient of variation
  across runs; llama3.2:3b shows <1% CV. Model architecture strongly determines
  run-to-run reproducibility.
- **Multi-agent interaction constrains vocabulary**: Agents in group conversation
  show reduced vocabulary diversity compared to isolated agents, confirming that
  interaction itself is a drift driver.
- **Double mutations produce non-linear effects**: Applying two trait mutations
  simultaneously produced a delayed cascade at turn 387, far later than single
  mutations. Non-linear interaction between mutations.
- **Collapsed-dissent sentiment artifact**: Group sentiment improves when the
  adversarial agent goes silent — a measurement artifact where loss of dissent
  reads as increased agreement.
- 18 auto-findings generated (F-0008 through F-0025) plus 3 manual findings
  (F-0005 through F-0007)

**Milestone 6 findings (statistical power batch, 19 runs)**:
- **Collapse is triggered, not spontaneous**: 0/12 gemma2:2b baselines collapsed
  (95% CI: 0–24.2%), while 5/6 mutation forks collapsed at exactly t53. Collapse
  requires a perturbation — unmutated baselines are stable.
- **Context saturation disproved as mechanism**: Collapsing forks hit collapse at
  prompt token counts of 2,000–3,090, while baselines survive at 3,900–4,090.
  The mechanism is mutation-triggered degenerate generation, not context overflow.
- **Pre-collapse thinning gradient**: Collapsing agents show negative output slope
  (-2 to -5 chars/turn) before collapse. The surviving fork showed +9.5 chars/turn.
  Potential early-warning signal.
- **Three post-collapse modes**: Cascade degradation (40%), isolated collapse (20%),
  and compensatory expansion (40%) — surviving agents may shrink, hold steady, or
  *grow* to fill the void left by a collapsed peer.
- **Hollow verbosity as alternative failure**: The one fork that avoided collapse
  entered a repetitive loop (78% of messages repeating the same phrase). Collapse
  and hollow verbosity are two expressions of the same failure — loss of generative
  diversity.
- **Model-dependent observer effects**: Same probing protocol is neutral on
  gemma2:2b but drift-suppressing on llama3.2:3b (dissociation gap +0.13–0.15).
  Measurement calibration must be per-model.
- **Probe drift is temporally flat**: Confirmed with n=12 baselines (5,760 probe
  measurements): early vs late drift delta is +0.001 to +0.006. Probes measure
  identity recall, not behavioral drift.
- 89 total findings in knowledge base (F-0001 through F-0089)

**What to look for**:
- Drift trajectory shape: linear, logarithmic, S-curve, or unbounded
- Dissociation between probe identity and conversational behavior
- Output collapse: agents producing empty or near-empty messages
- Whether triggered probes cluster at specific phases (early, mid, late)
- Context window effects — agents only see last 50 messages, so early
  conversation is forgotten. Does this cause phase transitions in drift?
- Whether injected probes disrupt or reset the convergence pattern

---

## 4. Probe Strategy Comparison

**Research question**: Do triggered probes capture drift events that
scheduled probes miss? Does probe injection itself affect drift?

**Config**: `config/three_agent_mesh.json`

```bash
python3 run_calibration.py config/three_agent_mesh.json

# Run A: scheduled probes only
python3 run_experiment.py config/three_agent_mesh.json \
    --probe shadow --probe-strategy scheduled \
    --probe-interval 10 --max-turns 100 --delay 5

# Run B: triggered probes only
python3 run_experiment.py config/three_agent_mesh.json \
    --probe shadow --probe-strategy triggered \
    --vocab-threshold 0.12 --max-turns 100 --delay 5

# Run C: hybrid
python3 run_experiment.py config/three_agent_mesh.json \
    --probe shadow --probe-strategy hybrid \
    --probe-interval 20 --max-turns 100 --delay 5

# Run D: injected probes (deliberately contaminates experiment)
python3 run_experiment.py config/three_agent_mesh.json \
    --probe injected --probe-strategy scheduled \
    --probe-interval 10 --max-turns 100 --delay 5

# Run E: both shadow and injected (direct comparison)
python3 run_experiment.py config/three_agent_mesh.json \
    --probe both --probe-strategy scheduled \
    --probe-interval 10 --max-turns 100 --delay 5
```

**What to look for**:
- Compare probe timing: does triggered mode fire at meaningful moments?
- Compare runs D and E: does injected probe presence change drift trajectory?
- Compare run A vs B: what drift events does each strategy capture?

---

## 5. Path Dependence (Fork Experiments)

**Research question**: If you restart agents from the same checkpoint with
fresh context, do they follow the same drift trajectory or diverge?

**Config**: `config/three_agent_mesh.json`

```bash
# Run base experiment
python3 run_experiment.py config/three_agent_mesh.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5

# Fork at turn 50 — agents start fresh from that checkpoint
python3 run_fork.py --at-turn 50 --max-turns 50 --delay 5 \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25

# Fork again at the same point — second independent run
python3 run_fork.py --at-turn 50 --max-turns 50 --delay 5 \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --suffix fork-2

# Compare drift trajectories of the two forks
python3 run_metrics.py  # run on each fork
```

**What to look for**:
- Do the two forks show similar or different drift trajectories?
- Similar = drift is deterministic (driven by conditions, not history)
- Different = drift is path-dependent (contingent on random conversation turns)

---

## 6. Trait Mutation (Controlled Variable Change)

**Research question**: How does changing a single agent's trait affect
group drift dynamics?

**Config**: `config/three_agent_mesh.json`

```bash
# Run base experiment
python3 run_experiment.py config/three_agent_mesh.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5

# Fork with mutation: make the pragmatist contrarian
python3 run_fork.py --mutate "Aria:disposition=contrarian" \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5

# Fork with mutation: make the optimist skeptical
python3 run_fork.py --mutate "Beck:disposition=skeptical" \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5

# Fork with multiple mutations
python3 run_fork.py \
    --mutate "Aria:disposition=contrarian" \
    --mutate "Beck:values=security" \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5
```

**What to look for**:
- Does the mutated agent's drift differ from its original?
- Do OTHER agents drift differently in response to the mutation?
- Which trait dimensions cause the most group-level disruption?

---

## 7. Cross-Model Drift (Homogeneous)

**Research question**: Do different models drift at different rates under
identical conditions?

**Config**: `config/three_agent_mesh.json`

```bash
python3 run_calibration.py config/three_agent_mesh.json

# Preview the matrix
python3 run_matrix.py config/three_agent_mesh.json --preview

# Run: one experiment per model, all agents same model
python3 run_matrix.py config/three_agent_mesh.json \
    --mode homogeneous \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5
```

**What to look for**:
- Absolute drift rates per model
- Which models maintain persona adherence longest?
- Do smaller models (gemma2:2b) drift faster than larger (llama3.2:3b)?

---

## 8. Asymmetric Model Experiments

**Research question**: Does introducing one agent on a different model
change group drift dynamics?

**Config**: `config/three_agent_mesh.json`

```bash
# Run: one agent different from the rest
python3 run_matrix.py config/three_agent_mesh.json \
    --mode asymmetric \
    --models gemma2:2b,llama3.2:3b \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5
```

**What to look for**:
- Does the "different" agent drift more or less than the homogeneous group?
- Do the homogeneous agents adapt to the different agent's style?
- Is there vocabulary or sentiment convergence across model boundaries?

---

## 9. Disposition Spectrum

**Research question**: How does the range of dispositions in a group affect
drift patterns? Does a group of similar agents drift less than a diverse one?

**Config**: `config/disposition_uniform.json` (all pragmatic) vs
`config/disposition_diverse.json` (one of each)

```bash
python3 run_calibration.py config/disposition_uniform.json
python3 run_calibration.py config/disposition_diverse.json

python3 run_experiment.py config/disposition_uniform.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5

python3 run_experiment.py config/disposition_diverse.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5
```

**What to look for**:
- Group convergence rate: does the uniform group stay stable?
- Does diversity cause more drift, or does conflict maintain distinctiveness?
- Sentiment trajectory differences between uniform and diverse groups

---

## 10. Scale Experiment (Agent Count)

**Research question**: Does adding more agents accelerate or dampen drift?

**Config**: `config/two_agent_mesh.json`, `config/three_agent_mesh.json`,
`config/five_agent_mesh.json`

```bash
# Run experiments with 2, 3, and 5 agents
python3 run_experiment.py config/two_agent_mesh.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5
python3 run_experiment.py config/three_agent_mesh.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5
python3 run_experiment.py config/five_agent_mesh.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5
```

**What to look for**:
- Per-agent drift rate vs group size
- Does a larger group amplify convergence pressure?
- Context window effects — more agents = fewer turns per agent in window

---

## 11. Temperature Sensitivity

**Research question**: Does inference temperature affect drift rate?
Higher temperature = more random = more or less drift?

**Config**: `config/temperature_low.json` (temp=0.3),
`config/temperature_high.json` (temp=1.0)

```bash
python3 run_experiment.py config/temperature_low.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5
python3 run_experiment.py config/temperature_high.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 100 --delay 5
```

**What to look for**:
- Does low temperature produce more consistent (less drifty) responses?
- Does high temperature cause faster initial drift but earlier stabilization?
- Vocab drift JSD should be higher at higher temperature — but does persona
  adherence drop faster too?

---

## 12. Context Window Pressure

**Research question**: Does the size of the sliding context window affect
drift? Do agents with shorter memory drift differently?

**Config**: `config/context_short.json` (max_history=10),
`config/context_long.json` (max_history=100)

```bash
python3 run_experiment.py config/context_short.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 200 --delay 5
python3 run_experiment.py config/context_long.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 200 --delay 5
```

**What to look for**:
- Short context = agents forget faster. Does this cause MORE drift or less?
- Long context = agents remember more conversation. Does conversation
  pressure compound over a longer window?
- Phase transitions: does drift behavior change when the window "fills up"?

---

## 13. Value Conflict Experiments

**Research question**: Do agents with opposing core values drift toward
consensus, maintain distinct positions, or destabilize?

**Config**: `config/value_conflict.json`

```bash
python3 run_calibration.py config/value_conflict.json

python3 run_experiment.py config/value_conflict.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 200 --delay 5
```

**What to look for**:
- Do agents with opposing values converge toward a middle ground?
- Or does conflict maintain persona distinctiveness (lower drift)?
- Triggered probes should fire during high-conflict exchanges

---

## 14. Regulation Stance Spectrum

**Research question**: On a specific policy topic, do agents with different
regulatory stances converge, and if so, toward which position?

**Config**: `config/regulation_spectrum.json`

```bash
python3 run_experiment.py config/regulation_spectrum.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 150 --delay 5
```

**What to look for**:
- Direction of convergence: toward minimal, moderate, or proactive regulation?
- Does the majority position "win" or does a confident minority dominate?
- Stance probe drift scores over time

---

## Live Monitoring

Any experiment can be monitored in real time from a second terminal.
The monitor shows per-agent drift metrics, probe results, throughput,
and governance status (GREEN/YELLOW/RED) — all computed live from the
SQLite database with no Ollama overhead.

```bash
# Terminal 1: run the experiment
python3 run_experiment.py config/value_conflict.json \
    --probe both --probe-strategy hybrid \
    --probe-interval 20 \
    --vocab-threshold 0.12 --sentiment-threshold 0.25 \
    --max-turns 200 --delay 5

# Terminal 2: watch it live
python3 run_monitor.py

# Or monitor a specific experiment with fast refresh
python3 run_monitor.py -e 3f9dd450 --refresh 1
```

**Keyboard controls**: `q` quit, `+`/`-` adjust refresh rate.

---

## Experiment Diffing

After running experiments, use the diff tool to compare results
side by side. Works for any pair of experiments.

```bash
# Diff two experiments by ID
python3 run_diff.py -a 3f9dd450 -b c6793fd1

# Auto-detect: diff a fork against its parent
python3 run_diff.py --fork-parent

# Auto-detect: diff experimental vs control arm
python3 run_diff.py --paired

# List experiments to find IDs
python3 run_diff.py --list
```

**Recommended diff workflows**:

```bash
# After a paired experiment (scenario 2):
python3 run_paired.py config/three_agent_mesh.json --max-turns 100 --delay 5
python3 run_diff.py --paired

# After a fork experiment (scenario 5):
python3 run_experiment.py config/three_agent_mesh.json --max-turns 100 --delay 5
python3 run_fork.py --at-turn 50 --max-turns 50 --delay 5
python3 run_diff.py --fork-parent

# After a trait mutation (scenario 6):
python3 run_fork.py --mutate "Aria:disposition=contrarian" --max-turns 100 --delay 5
python3 run_diff.py --fork-parent

# Comparing temperature experiments (scenario 11):
python3 run_experiment.py config/temperature_low.json --max-turns 100 --delay 5
python3 run_experiment.py config/temperature_high.json --max-turns 100 --delay 5
python3 run_diff.py -a <low_id> -b <high_id>
```

---

## Quick Reference: Running All Scenarios

```bash
# Calibrate all configs first
for cfg in config/*.json; do
    echo "Calibrating: $cfg"
    python3 run_calibration.py "$cfg" --runs 3
done

# Then run experiments (adjust --max-turns and --delay as needed)
python3 run_experiment.py config/three_agent_mesh.json --max-turns 100 --delay 5
python3 run_paired.py config/three_agent_mesh.json --max-turns 100 --delay 5
# ... etc
```

---

## Batch Runner

For multi-run experiments (milestones), use the batch runner instead of
running experiments individually. Batch configs define sequences of
calibration, experiment, paired, and fork runs with automated recovery
and post-batch analysis.

```bash
# Run a milestone batch
python3 run_batch.py batch/m5_multi_model_fork.json

# Resume after interruption (skips completed, re-runs partial)
python3 run_batch.py batch/m5_multi_model_fork.json --resume

# Preview without running
python3 run_batch.py batch/m5_multi_model_fork.json --dry-run
```

**Between-run automation:**
- Model unloading and page cache drops
- Ollama health checks with 3-level recovery escalation
- Thermal guards (pauses at 82°C, resumes at 68°C)
- Per-run metrics computation

**Post-batch automation:**
- Diff comparisons (cross-model, paired, fork)
- Pattern analysis (run_analyze.py)
- Auto-generated findings (run_findings.py)

Batch configs: `batch/*.json`. See `batch/m5_multi_model_fork.json` for a
complete example with 9 runs and 9 auto-comparisons.

---

## Findings Knowledge Base

After running experiments and diffs, use the findings tools to capture
and organize discoveries.

```bash
# Run automated pattern analysis on one or more experiments
# Implements 10+ detector types (collapse, cascade, correlation, vocabulary, etc.)
# Detection patterns and thresholds configurable via config/detection_patterns.json
# Auto-finding deduplication prevents duplicate findings for same experiment+pattern
python3 run_analyze.py -e <exp_id_1> -e <exp_id_2>

# List all findings
python3 run_findings.py list

# Search findings
python3 run_findings.py search "vocabulary collapse"

# Show a specific finding
python3 run_findings.py show F-0001

# List all tags
python3 run_findings.py tags

# Find related findings
python3 run_findings.py related F-0001
```

Findings are stored as structured JSON in `findings/F-*.json`.

---

## Troubleshooting

### Resuming Interrupted Experiments

If an experiment is interrupted (SSH disconnect, thermal shutdown, Ollama
crash), resume from the exact point of interruption:

```bash
# Resume a single experiment by ID
python3 run_resume.py -e <experiment_id> --max-turns 500

# Resume a batch (skips completed, resumes interrupted, re-runs partial)
python3 run_batch.py batch/<config>.json --resume
```

The resume system tracks which agent spoke last in each turn cycle, so it
picks up from the next agent — no turns are skipped or duplicated.

### Ollama Recovery

If Ollama becomes unresponsive:

```bash
# Manual recovery (3-level escalation)
source recover_ollama.sh
recover_ollama

# Or restart the service directly
sudo systemctl restart ollama
```

The batch runner has built-in Python-native recovery that runs
automatically between experiments.

### Thermal Issues

The Jetson Orin Nano throttles at 74°C and shuts down at 104.5°C. SENTINEL
has two-layer thermal protection:

- **thermal_guard.sh**: Shell functions for manual/scripted use
- **sentinel/thermal.py**: Async Python integration in the runtime

Both check the tj-thermal sensor (thermal_zone8). The runtime automatically
pauses inference when temperature exceeds 82°C and resumes at 68°C.

If you see PCIe bus errors in `dmesg`:
- **Corrected errors** (severity=Corrected): Normal, hardware recovered automatically
- **Uncorrectable errors**: Usually thermal — check temperature, improve cooling

### Model Switching on Jetson

Jetson unified memory means CUDA and system RAM share the same 8GB. When
switching models:

1. Unload the current model from Ollama VRAM
2. Drop page caches (`echo 3 | sudo tee /proc/sys/vm/drop_caches`)
3. Verify the new model loads successfully before starting experiments

The batch runner handles this automatically between runs.
