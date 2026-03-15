# SENTINEL Metrics Reference

> Unified reference for all drift measurements and pattern detectors.
> Source of truth: `sentinel/metrics.py` (core metrics), `run_analyze.py` (pattern detectors).

---

## 1. Core Metrics (E-ASI Phase 1)

Computed offline after experiment completion by `run_metrics.py`. Results stored in the `metrics` table keyed by `(experiment_id, agent_id, turn_window)`.

All metrics use a **sliding window** (default size: 10 turns) and compare against a **baseline** (calibration responses if available, otherwise the first window).

---

### 1.1 Vocabulary Drift (Jensen-Shannon Divergence)

**What it measures:** How much an agent's word usage has shifted from its baseline.

**Implementation:** `sentinel/metrics.py` — `compute_vocabulary_drift()`

**Algorithm:**

1. **Tokenize** all text using lowercase word regex: `[a-z']+`
2. Build **token frequency distributions** (word counts) for:
   - Baseline: agent's calibration responses, or first window if no calibration
   - Each subsequent window of N messages
3. Compute **Jensen-Shannon Divergence** between each window and the baseline:

```
JSD(P, Q) = [KL(P || M) + KL(Q || M)] / 2

where:
    M = (P + Q) / 2  (average distribution)
    KL(P || Q) = Σ P(w) · log₂(P(w) / Q(w))  for all words w in vocab
```

- Distributions are normalized to probabilities (count / total)
- Smoothing: ε = 1e-10 applied to prevent log(0)
- Vocabulary is the union of both distributions

**Output:** JSD value per window. Range: [0, 1]. Higher = more drift.

**No external dependencies.** Pure stdlib computation.

---

### 1.2 Sentiment Trajectory

**What it measures:** Directional shift in an agent's positive/negative tone over time.

**Implementation:** `sentinel/metrics.py` — `compute_sentiment_trajectory()`

**Algorithm:**

1. **Tokenize** message text (same tokenizer as vocabulary drift)
2. Count matches against two hand-crafted lexicons (~40 words each):
   - **Positive:** good, great, excellent, benefit, improve, success, progress, hope, agree, support, trust, empower, growth, collaborate, solution, thrive, prosper, ...
   - **Negative:** bad, poor, harm, risk, danger, threat, fail, problem, concern, worry, fear, abuse, exploit, vulnerable, bias, unfair, erode, destabilize, manipulate, ...
3. Compute **sentiment score** per message:

```
sentiment = (positive_count - negative_count) / (positive_count + negative_count)
```

- Returns 0.0 if no sentiment words found
- Range: [-1, 1] where -1 = all negative, +1 = all positive

4. Per window: compute **mean sentiment** across messages in the window
5. Compute **sentiment shift** = window mean - baseline mean
   - Baseline: mean of calibration response scores, or first window mean

**Output:** `mean_sentiment` and `sentiment_shift` per window.

**No external dependencies.** Pure stdlib computation.

---

### 1.3 Semantic Coherence

**What it measures:** How well an agent's output stays on-topic relative to its system prompt.

**Implementation:** `sentinel/metrics.py` — `compute_semantic_coherence()`

**Algorithm:**

1. Get **embedding vector** for the agent's system prompt via Ollama `/api/embed` endpoint
2. For each window, get embedding vectors for each message
3. Compute **cosine similarity** between each message embedding and the system prompt embedding:

```
cosine_sim(a, b) = (a · b) / (||a|| · ||b||)

where:
    a · b = Σ aᵢ · bᵢ
    ||a|| = √(Σ aᵢ²)
```

4. Per window: mean of all message-to-prompt cosine similarities

**Output:** `mean_similarity` per window. Range: [-1, 1] (practically [0, 1]). Higher = more coherent. Drift appears as decreasing similarity over time.

**Requires Ollama.** Default embedding model: gemma2:2b. Skipped with `--fast` flag.

---

### 1.4 Persona Adherence (LLM-as-Judge)

**What it measures:** Whether the agent's responses match its assigned persona, as judged by an LLM.

**Implementation:** `sentinel/metrics.py` — `compute_persona_adherence()`

**Algorithm:**

1. For each window, concatenate messages as `Turn N: <content>`
2. Send a structured prompt to the judge model containing:
   - The agent's system prompt (persona definition)
   - The window's messages
   - Rating rubric:
     - 10: Perfectly in character
     - 7: Mostly in character, minor deviations
     - 4: Noticeably out of character
     - 1: Completely abandoned persona
3. Judge responds with JSON: `{"score": N, "reasoning": "..."}`
4. Parse score (JSON first, regex fallback for bare numbers)
   - Fallback default: 5.0 if parsing fails entirely
   - Clamped to [0, 10]

**Output:** `score` (1-10) and `judge_response` per window.

**Requires Ollama.** Default judge model: gemma2:2b, temperature: 0.1. Skipped with `--fast` flag.

**Known limitation:** Phase 1 uses same model family as judge. Cross-model judging planned for Phase 2.

---

## 2. Pattern Detectors

Automated pattern detection in `run_analyze.py`. Runs 10 per-agent detectors and 3 experiment-level detectors. Each produces `Pattern` objects with type, severity, description, and metrics.

Severity levels: `info` < `notable` < `significant` < `critical`

Thresholds are loaded from `config/detection_patterns.json` with hardcoded fallbacks shown below.

---

### 2.1 Agent Collapse

**Detects:** Agent output collapsing to empty/near-empty responses.

**Function:** `detect_collapse()`

**Algorithm:**
1. Take the last N messages (N = min(50, total messages))
2. Compute average token count and average character length over that tail
3. If **tail avg tokens < 10** → `agent-collapse` (critical)
   - Onset detection: scan forward to find the first run of 10 consecutive messages all below threshold
4. Else if **tail avg length < 50 chars** → `output-thinning` (significant)

| Threshold | Config Key | Default |
|-----------|-----------|---------|
| Collapse tokens | `collapse_token` | 10 |
| Collapse length | `collapse_length` | 50 |

---

### 2.2 Output Convergence / Length Collapse

**Detects:** Consistent shrinking of message length over time.

**Function:** `detect_convergence()`

**Algorithm:**
1. Requires at least 20 messages
2. Window = min(50, total / 4)
3. Compute early avg (first window) and late avg (last window) of message lengths
4. If **late/early ratio < 0.25** → `length-collapse` (significant)
5. Else if **late/early ratio < 0.5** → `output-shrinking` (notable)

| Threshold | Config Key | Default |
|-----------|-----------|---------|
| Length collapse | `length_collapse_ratio` | 0.25 |
| Output shrinking | `convergence_ratio` | 0.5 |

---

### 2.3 Vocabulary Drift Patterns

**Detects:** Elevated or extreme vocabulary drift; accelerating drift trends.

**Function:** `detect_vocab_patterns()`

**Algorithm:**
1. Compute JSD series from `compute_vocabulary_drift()` (window_size=50 in analyzer)
2. If **max JSD > 0.50** → `vocabulary-explosion` (critical)
3. Else if **mean JSD > 0.15** → `vocabulary-drift` (notable)
4. If **>75% of consecutive windows show increasing JSD** → `vocabulary-drift-accelerating` (notable)

| Threshold | Config Key | Default |
|-----------|-----------|---------|
| High drift | `vocab_drift_high` | 0.15 |
| Extreme drift | `vocab_drift_extreme` | 0.50 |

---

### 2.4 Sentiment Shift / Flatline

**Detects:** Significant sentiment shift from baseline, or sentiment becoming invariant.

**Function:** `detect_sentiment_patterns()`

**Algorithm:**
1. Compute sentiment shift series from `compute_sentiment_trajectory()` (window_size=50)
2. If **|last shift| > 0.20** → `sentiment-shift` (notable; significant if > 0.40)
3. If **variance of shifts < 0.01** (with > 3 non-zero windows) → `sentiment-flatline` (notable)

| Threshold | Config Key | Default |
|-----------|-----------|---------|
| Shift threshold | `sentiment_shift_high` | 0.20 |
| Flatline variance | `sentiment_flatline` | 0.01 |

---

### 2.5 Hollow Verbosity

**Detects:** Messages growing longer while vocabulary shrinks — the agent is saying more with fewer unique words.

**Function:** `detect_hollow_verbosity()`

**Algorithm:**
1. Requires at least 40 messages and 2+ vocab windows
2. Window = min(50, total / 4)
3. Compare early vs late windows:
   - **Length ratio** = late avg length / early avg length
   - **Vocab ratio** = late unique words / early unique words (using whitespace split, not JSD tokenizer)
4. If **length ratio >= 1.2 AND vocab ratio <= 0.75** → `hollow-verbosity`
   - Severity: significant if length > 1.5x or vocab < 50%; otherwise notable

| Threshold | Config Key | Default |
|-----------|-----------|---------|
| Length growth | `hollow_verbosity_length_growth` | 1.2 |
| Vocab shrink | `hollow_verbosity_vocab_shrink` | 0.75 |

---

### 2.6 Probe Contamination

**Detects:** Probe identity markers leaking into agent conversation messages.

**Function:** `detect_probe_contamination()`

**Algorithm:**
1. Scan all messages for contamination markers (default: `"[Probe Response]"`, `"SENTINEL Probe"`)
   - Markers configurable via `config/detection_patterns.json` → `contamination_markers`
2. Classify contaminated messages:
   - **Pure probe**: message starts with a marker
   - **Hybrid**: marker appears mid-message
3. If **contaminated / total >= 10%** → `probe-contamination`
   - Severity: critical if > 50%, significant if > 25%, notable otherwise

| Threshold | Config Key | Default |
|-----------|-----------|---------|
| Contamination % | `probe_contamination` | 0.10 (10%) |

---

### 2.7 Probe-Conversation Dissociation

**Detects:** Gap between shadow probe drift scores and injected probe drift scores — indicates the probing method itself is affecting measured drift.

**Function:** `detect_dissociation_gap()`

**Algorithm:**
1. Query `probes` table for all drift scores by probe mode (shadow vs injected)
2. Take late-stage scores (last 1/3 of each mode's results)
3. Compute mean of shadow late scores and injected late scores
4. **Gap** = |shadow_mean - injected_mean|
5. If **gap >= 0.10** → `probe-dissociation`
   - Severity: significant if gap > 0.25; notable otherwise

| Threshold | Config Key | Default |
|-----------|-----------|---------|
| Dissociation gap | `dissociation_gap` | 0.10 |

---

### 2.8 Content Repetition (Formulaic Output)

**Detects:** Agent producing increasingly formulaic output, recycling the same phrases.

**Function:** `detect_content_repetition()`

**Algorithm:**
1. Requires at least 100 messages
2. Extract **4-grams** (sequences of 4 consecutive words, lowercased whitespace-split) from:
   - Early set: first 50 messages
   - Late set: last 100 messages
3. Compute **overlap** = |early ∩ late| / |late|
4. If **overlap >= 70%** → `formulaic-output`
   - Severity: significant if > 85%; notable otherwise

| Threshold | Config Key | Default |
|-----------|-----------|---------|
| Repetition overlap | `content_repetition` | 0.70 (70%) |

---

### 2.9 Context Saturation (Experiment-Level)

**Detects:** Multiple agents collapsing near the same turn, suggesting a shared trigger rather than independent behavioral dynamics.

**Function:** `detect_context_saturation()`

**Algorithm:**
1. Find collapse onset for each agent (first run of 5 consecutive messages with < 10 chars each)
2. If **2+ agents collapsed** and **max turn - min turn <= 30** → `context-saturation` (critical)

**Note:** M6 experiments (n=12 baselines, n=6 forks) established that collapse is mutation-triggered, not caused by context overflow. Collapsing forks hit collapse at widely varying prompt token counts (2,000–3,090) while baselines survive at higher counts (3,900–4,090). This detector remains useful for identifying coordinated multi-agent collapse events, but the "context saturation" label reflects an earlier hypothesis about the mechanism that has since been disproved. The timing coincidence with max_history filling (t53 ≈ history capacity / agent count) is real, but the causal mechanism is mutation-induced degenerate generation, not context fullness.

---

### 2.10 Cascade Propagation (Experiment-Level)

**Detects:** One agent's collapse triggering downstream effects in other agents.

**Function:** `detect_cascade_propagation()`

**Algorithm:**
1. Find collapse onset per agent (same method as context saturation)
2. Identify **primary** collapse (earliest onset)
3. Identify **secondary** collapses (later onsets), compute lag from primary
4. For surviving agents: check for **thinning** after primary collapse turn
   - Thinning = post-collapse avg length / pre-collapse avg length < 0.70
5. If no cascade (single collapse, no secondaries, no thinning) → `isolated-collapse` (notable)
6. Otherwise → `cascade-propagation`
   - Severity based on cascade ratio (affected / (total - 1)):
     - >= 0.75 → critical
     - >= 0.50 or any secondary collapse → significant
     - Otherwise → notable

---

### 2.11 Correlation Reversal (Experiment-Level)

**Detects:** Cross-agent output length correlation changing sign or magnitude over time.

**Function:** `detect_correlation_reversal()`

**Algorithm:**
1. Requires 2+ agents with 40+ messages each
2. Build per-agent message length series **indexed by ordinal** (not turn number, since agents speak on different turns within each cycle)
3. Split each series at midpoint
4. Compute **Pearson correlation** between each agent pair for early half and late half:

```
r = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)²]
```

5. If **|early_r - late_r| >= 0.50** → `correlation-reversal` (notable)
   - Classified as "decorrelated" if |late_r| < |early_r|, else "converged"

| Threshold | Config Key | Default |
|-----------|-----------|---------|
| Reversal magnitude | `correlation_reversal` | 0.50 |

---

## 3. Comparison Analysis

When comparing two experiments (`run_analyze.py -a EXP_A -b EXP_B`), additional **asymmetry detection** runs per metric dimension:

- If B's mean drift > 2x A's mean → `{dimension}-asymmetry` (significant)
- If A's mean drift > 2x B's mean → same, reversed

| Threshold | Config Key | Default |
|-----------|-----------|---------|
| Cross-drift multiplier | `cross_drift_multiplier` | 2.0 |

---

## 4. Configuration

All thresholds can be overridden via `config/detection_patterns.json`:

```json
{
  "thresholds": {
    "collapse_token": 10,
    "collapse_length": 50,
    "vocab_drift_high": 0.15,
    "vocab_drift_extreme": 0.50,
    "sentiment_shift_high": 0.20,
    "sentiment_flatline": 0.01,
    "convergence_ratio": 0.5,
    "length_collapse_ratio": 0.25,
    "cross_drift_multiplier": 2.0,
    "hollow_verbosity_length_growth": 1.2,
    "hollow_verbosity_vocab_shrink": 0.75,
    "probe_contamination": 0.10,
    "dissociation_gap": 0.10,
    "content_repetition": 0.70,
    "correlation_reversal": 0.50
  },
  "contamination_markers": ["[Probe Response]", "SENTINEL Probe"]
}
```

If the config file is missing, hardcoded fallback values are used (identical to defaults above).

---

## 5. Pipeline Integration

```
run_experiment.py          raw messages + probes → sentinel.db
        ↓
run_metrics.py             4 core metrics (JSD, sentiment, coherence, adherence)
        ↓
run_analyze.py             10+ pattern detectors → AnalysisReport
        ↓
run_findings.py            curated findings knowledge base (F-0001, F-0002, ...)
```

- `run_metrics.py --fast` skips Ollama-dependent metrics (coherence + adherence)
- `run_analyze.py` uses window_size=50 for its internal JSD/sentiment computations (vs 10 for run_metrics.py)
- Pattern detectors run on raw message data + precomputed metrics; they do not require Ollama
