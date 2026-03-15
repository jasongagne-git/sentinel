#!/usr/bin/env bash

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

# SENTINEL — llama3 isolation test
#
# Runs calibration + 500-turn experiment using ONLY llama3:latest.
# No model switching. Designed to run on a clean boot to determine
# whether llama3 works standalone on this Jetson.
#
# Usage:
#   nohup ./run_llama3_test.sh >> experiments/logs/nohup.out 2>&1 &
#   # or interactively:
#   ./run_llama3_test.sh

set -uo pipefail  # no -e: we handle errors ourselves
trap '' HUP        # survive SSH disconnects

MODEL="llama3.2:3b"
CONFIG="config/long_run_llama3.json"
DB="experiments/sentinel.db"
LOG_DIR="experiments/logs"
LOG="$LOG_DIR/llama3_test.log"
STATUS_FILE="$LOG_DIR/status.txt"
PROBE_ARGS="--probe both --probe-strategy hybrid --probe-interval 20 --vocab-threshold 0.12 --sentiment-threshold 0.25"
COMMON_ARGS="--max-turns 500 --delay 5 --db $DB"
STALL_TIMEOUT=300
TICKER_PID=""
EXP_ID="NOT_RUN"

cd "$(dirname "$0")"
mkdir -p "$LOG_DIR"

# Source recovery functions
source ./recover_ollama.sh --source-only

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

log() { echo "[$(timestamp)] $*" >> "$LOG"; }

status() { echo "[$(timestamp)] $*" > "$STATUS_FILE"; log "STATUS: $*"; }

echo "llama3 test starting. Monitor with:"
echo "  tail -f $LOG"
echo "  cat $STATUS_FILE"

# Background progress ticker
start_ticker() {
    local run_label="$1"
    stop_ticker
    (
        local last_turn=-1
        local last_change
        last_change=$(date +%s)
        while true; do
            sleep 30
            local turn
            turn=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$DB')
row = conn.execute(
    'SELECT MAX(m.interaction_turn) FROM messages m JOIN experiments e ON m.experiment_id=e.experiment_id WHERE e.status=\"running\"'
).fetchone()
print(row[0] if row and row[0] else 0)
" 2>/dev/null || echo 0)
            local now
            now=$(date +%s)
            if [ "$turn" != "$last_turn" ]; then
                log "  [$run_label] turn $turn"
                last_turn="$turn"
                last_change=$now
                status "$run_label — turn $turn in progress"
            else
                local stall_secs=$(( now - last_change ))
                if [ "$stall_secs" -ge "$STALL_TIMEOUT" ]; then
                    log "  [$run_label] WARNING: STALLED at turn $turn for ${stall_secs}s"
                    status "STALL: $run_label stuck at turn $turn for ${stall_secs}s"
                fi
            fi
        done
    ) &
    TICKER_PID=$!
}

stop_ticker() {
    if [ -n "$TICKER_PID" ]; then
        kill "$TICKER_PID" 2>/dev/null || true
        wait "$TICKER_PID" 2>/dev/null || true
        TICKER_PID=""
    fi
}

trap stop_ticker EXIT

run_metrics() {
    local exp_id="$1"
    log "Computing fast metrics..."
    python3 run_metrics.py -e "$exp_id" --db "$DB" --fast -w 50 \
        >> "$LOG" 2>&1
    log "Metrics complete."
}

# ── Pre-flight ──────────────────────────────────────────────────

log "=========================================="
log "SENTINEL — llama3 Isolation Test"
log "=========================================="
log "Model:  $MODEL"
log "Config: $CONFIG"
log "Goal:   Verify llama3 works standalone (no model switching)"
log ""

# Free memory by stopping unnecessary services
log "Checking system services..."
source ./check_services.sh --source-only
log "Stopping unnecessary services to free memory..."
YES=true stop_services 2>&1 | while IFS= read -r line; do log "  $line"; done
sleep 2

# Verify no other models are loaded — unload everything first
log "Unloading all models from VRAM..."
for m in "gemma2:2b" "llama3.2:3b" "llama3:latest" "phi3:mini"; do
    curl -s http://localhost:11434/api/generate \
        -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null 2>&1 || true
done
sleep 3

# Drop caches for a clean slate
log "Dropping page caches..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || \
    log "WARNING: Could not drop caches (sudo failed — run after fresh reboot)"
sleep 2

# Health check — llama3 must load cleanly
log "Health check: loading $MODEL..."
status "Pre-flight — loading $MODEL"
if ! check_ollama_health "$MODEL"; then
    log "ERROR: $MODEL failed to load on health check."
    log ""
    log "This likely means NvMap is corrupted from a previous session."
    log "Reboot first, then re-run this script:"
    log "  sudo reboot"
    log "  # after reboot:"
    log "  cd ~/Projects/sentinel && ./run_llama3_test.sh"
    status "FAILED: $MODEL did not load — reboot needed"
    exit 1
fi
log "Health check PASSED — $MODEL loaded successfully"
log ""

# ── Calibration ─────────────────────────────────────────────────

log "--- Phase 1: Calibration ---"
status "Calibrating $MODEL"
if ! python3 run_calibration.py "$CONFIG" --db "$DB" \
    >> "$LOG_DIR/llama3_test_calibration.log" 2>&1; then
    log "ERROR: Calibration failed (exit $?). See llama3_test_calibration.log"
    status "FAILED: Calibration"
    exit 1
fi
log "Calibration complete."
log ""

# ── Experiment ──────────────────────────────────────────────────

log "--- Phase 2: 500-turn experiment ---"
status "Experiment starting"
start_ticker "llama3-test"

if python3 run_experiment.py "$CONFIG" \
    $PROBE_ARGS $COMMON_ARGS \
    >> "$LOG_DIR/llama3_test_experiment.log" 2>&1; then
    stop_ticker
    EXP_ID=$(python3 -c "
import sqlite3
conn = sqlite3.connect('$DB')
row = conn.execute(
    \"SELECT experiment_id FROM experiments WHERE name LIKE '%llama3%' AND status='completed' ORDER BY completed_at DESC LIMIT 1\"
).fetchone()
print(row[0][:8] if row else 'UNKNOWN')
")
    log "Experiment complete: $EXP_ID"
    status "Experiment DONE: $EXP_ID"
else
    stop_ticker
    log "ERROR: Experiment failed (exit $?). See llama3_test_experiment.log"
    status "FAILED: Experiment"
    exit 1
fi
log ""

# ── Metrics ─────────────────────────────────────────────────────

log "--- Phase 3: Metrics ---"
status "Computing metrics"
run_metrics "$EXP_ID"
log ""

# ── Summary ─────────────────────────────────────────────────────

log "=========================================="
log "SENTINEL — llama3 Isolation Test COMPLETE"
log "=========================================="
log "Experiment ID: $EXP_ID"
log "Logs:"
log "  Calibration: $LOG_DIR/llama3_test_calibration.log"
log "  Experiment:  $LOG_DIR/llama3_test_experiment.log"
log "  This log:    $LOG"
log ""
log "Next steps:"
log "  python3 run_metrics.py -e $EXP_ID --db $DB -w 50   # full metrics"
log "  python3 run_diff.py -a <gemma2_id> -b $EXP_ID --db $DB   # cross-model diff"
log "=========================================="
status "COMPLETE: llama3 test — $EXP_ID"
