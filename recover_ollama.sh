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

# SENTINEL — Ollama Recovery & Pre-flight Check
#
# Attempts to recover Ollama from NvMap/CUDA corruption and verify
# model health before resuming experiments. Can be called standalone
# or sourced from overnight/resume scripts.
#
# Usage:
#   ./recover_ollama.sh [--model MODEL] [--resume-script SCRIPT] [--reboot-ok]
#
#   --model MODEL        Model to verify (default: gemma2:2b)
#   --resume-script PATH Script to launch after successful recovery
#   --reboot-ok          Allow automatic reboot if soft recovery fails
#                        (will prompt for sudo password)
#   --no-prompt          Skip interactive prompts (for cron/automation)
#
# Exit codes:
#   0 = Ollama healthy, model verified
#   1 = Recovery failed, manual intervention needed
#   2 = Rebooting (script scheduled post-reboot resume)
#
# Can also be sourced for individual functions:
#   source recover_ollama.sh --source-only
#   check_ollama_health "gemma2:2b" || recover_ollama "gemma2:2b"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RECOVER_LOG_DIR="$SCRIPT_DIR/experiments/logs"
RECOVER_LOG="$RECOVER_LOG_DIR/recovery.log"
RESUME_FLAG="$RECOVER_LOG_DIR/.pending_resume"

# Defaults
TARGET_MODEL="gemma2:2b"
RESUME_SCRIPT=""
REBOOT_OK=false
NO_PROMPT=false
SOURCE_ONLY=false

# Known models to unload
ALL_MODELS=("gemma2:2b" "llama3.2:3b" "llama3:latest" "phi3:mini")

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

rlog() {
    mkdir -p "$RECOVER_LOG_DIR"
    echo "[$(timestamp)] $*" | tee -a "$RECOVER_LOG"
}

# ── Health checks ──────────────────────────────────────────────

ollama_is_running() {
    curl -sf http://localhost:11434/api/tags > /dev/null 2>&1
}

# Try a single inference with the given model. Returns 0 if OK.
check_ollama_health() {
    local model="${1:-$TARGET_MODEL}"
    local response
    response=$(curl -sf --max-time 120 http://localhost:11434/api/generate \
        -d "{\"model\":\"$model\",\"prompt\":\"Say hello.\",\"stream\":false}" 2>&1)
    local rc=$?
    if [ $rc -eq 0 ] && echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('response')" 2>/dev/null; then
        return 0
    fi
    return 1
}

# ── Service check (Level 0) ────────────────────────────────────

# Source the service auditor
if [ -f "$SCRIPT_DIR/check_services.sh" ]; then
    source "$SCRIPT_DIR/check_services.sh" --source-only
    _HAS_SERVICE_CHECK=true
else
    _HAS_SERVICE_CHECK=false
fi

# Stop unnecessary services to free memory for model loading.
# Respects user keep-list in ~/.config/sentinel/keep-services.conf
free_memory_for_model() {
    if [ "$_HAS_SERVICE_CHECK" != true ]; then
        rlog "WARNING: check_services.sh not found, skipping service audit"
        return 0
    fi

    local need_mb="${1:-4200}"

    if preflight_memory_check "$need_mb" 2>/dev/null; then
        rlog "Memory OK for model (need ${need_mb} MB)"
        return 0
    fi

    rlog "Insufficient memory — stopping unnecessary services..."
    YES=true QUIET=false stop_services 2>&1 | while IFS= read -r line; do
        rlog "  $line"
    done

    # Drop caches after stopping services
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    sleep 2

    local avail_mb
    avail_mb=$(awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo)
    rlog "Available after service cleanup: ${avail_mb} MB"
    return 0
}

# ── Recovery steps (escalating) ────────────────────────────────

# Level 1: Unload all models, drop page caches
soft_recover() {
    rlog "Level 1: Soft recovery — unload models + drop caches"

    # Unload all known models
    for m in "${ALL_MODELS[@]}"; do
        curl -s http://localhost:11434/api/generate \
            -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null 2>&1 || true
    done
    sleep 3

    # Drop page caches (needs sudo)
    rlog "Dropping page caches..."
    if ! sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then
        rlog "WARNING: Could not drop caches (sudo failed)"
    fi
    sleep 2
}

# Level 2: Restart Ollama systemd service
restart_ollama() {
    rlog "Level 2: Restarting Ollama service..."
    sudo systemctl restart ollama 2>/dev/null
    local rc=$?
    if [ $rc -ne 0 ]; then
        rlog "WARNING: systemctl restart failed (rc=$rc)"
        return 1
    fi

    # Wait for Ollama to come back up
    local tries=0
    while [ $tries -lt 15 ]; do
        sleep 2
        if ollama_is_running; then
            rlog "Ollama service is back up"
            return 0
        fi
        tries=$((tries + 1))
    done
    rlog "ERROR: Ollama did not come back after restart"
    return 1
}

# Level 3: Full reboot
do_reboot() {
    rlog "Level 3: Scheduling reboot..."

    # Write resume flag so we know to resume after reboot
    if [ -n "$RESUME_SCRIPT" ]; then
        echo "$RESUME_SCRIPT" > "$RESUME_FLAG"
        rlog "Resume flag written: $RESUME_FLAG → $RESUME_SCRIPT"
    fi

    rlog "Rebooting NOW"
    sudo reboot
    exit 2
}

# ── Model verification ────────────────────────────────────────

verify_model() {
    local model="${1:-$TARGET_MODEL}"
    rlog "Verifying model: $model"

    if check_ollama_health "$model"; then
        rlog "Model $model is healthy"
        return 0
    fi

    rlog "Model $model failed health check"
    return 1
}

# ── Main recovery sequence ────────────────────────────────────

recover_ollama() {
    local model="${1:-$TARGET_MODEL}"

    rlog "=========================================="
    rlog "SENTINEL Ollama Recovery"
    rlog "Target model: $model"
    rlog "=========================================="

    # Already healthy?
    if check_ollama_health "$model"; then
        rlog "Ollama is already healthy with $model — no recovery needed"
        return 0
    fi

    rlog "Ollama is NOT healthy — starting recovery sequence"

    # Level 0: free memory by stopping unnecessary services
    free_memory_for_model 4200

    # Level 1: soft recovery
    if ollama_is_running; then
        soft_recover
        sleep 3
        if check_ollama_health "$model"; then
            rlog "Recovery SUCCESS after Level 1 (soft)"
            return 0
        fi
    fi

    # Level 2: service restart
    if restart_ollama; then
        # Drop caches after fresh restart too
        sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
        sleep 5
        if check_ollama_health "$model"; then
            rlog "Recovery SUCCESS after Level 2 (service restart)"
            return 0
        fi
    fi

    # Level 3: reboot
    rlog "Soft recovery and service restart both failed"
    rlog "This is likely NvMap driver corruption — reboot required"

    if [ "$REBOOT_OK" = true ]; then
        do_reboot
        # does not return
    fi

    if [ "$NO_PROMPT" = true ]; then
        rlog "ERROR: Reboot required but --no-prompt set. Giving up."
        return 1
    fi

    echo ""
    echo "=== Ollama recovery failed — reboot required ==="
    echo "NvMap driver is likely corrupted from CUDA OOM."
    echo ""
    read -r -p "Reboot now? [y/N] " answer
    if [[ "$answer" =~ ^[Yy] ]]; then
        do_reboot
    else
        rlog "User declined reboot"
        echo "To recover manually:"
        echo "  sudo reboot"
        echo "  # after reboot:"
        echo "  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'"
        echo "  ollama run $model 'hello'   # verify"
        if [ -n "$RESUME_SCRIPT" ]; then
            echo "  nohup $RESUME_SCRIPT >> $RECOVER_LOG_DIR/nohup.out 2>&1 &"
        fi
        return 1
    fi
}

# ── Post-reboot resume check ─────────────────────────────────

check_pending_resume() {
    if [ -f "$RESUME_FLAG" ]; then
        local script
        script=$(cat "$RESUME_FLAG")
        rlog "Found pending resume flag: $script"

        # Verify Ollama is healthy first
        local tries=0
        while [ $tries -lt 30 ]; do
            if ollama_is_running; then
                break
            fi
            sleep 2
            tries=$((tries + 1))
        done

        if ! ollama_is_running; then
            rlog "ERROR: Ollama not running after reboot"
            rm -f "$RESUME_FLAG"
            return 1
        fi

        # Drop caches and verify model
        sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
        sleep 3

        if check_ollama_health "$TARGET_MODEL"; then
            rlog "Post-reboot health check passed"
            rm -f "$RESUME_FLAG"

            if [ -n "$script" ] && [ -x "$script" ]; then
                rlog "Launching resume script: $script"
                echo "Ollama healthy after reboot. Launching: $script"
                nohup "$script" >> "$RECOVER_LOG_DIR/nohup.out" 2>&1 &
                rlog "Resume script launched (PID $!)"
                return 0
            else
                rlog "Resume script not found or not executable: $script"
                rm -f "$RESUME_FLAG"
                return 1
            fi
        else
            rlog "ERROR: Model still failing after reboot!"
            rm -f "$RESUME_FLAG"
            return 1
        fi
    fi
    return 0
}

# ── Parse arguments ───────────────────────────────────────────

if [[ "${1:-}" == "--source-only" ]]; then
    SOURCE_ONLY=true
fi

if [ "$SOURCE_ONLY" = false ]; then
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)       TARGET_MODEL="$2"; shift 2 ;;
            --resume-script) RESUME_SCRIPT="$2"; shift 2 ;;
            --reboot-ok)   REBOOT_OK=true; shift ;;
            --no-prompt)   NO_PROMPT=true; shift ;;
            --check-resume) check_pending_resume; exit $? ;;
            --health)      check_ollama_health "$TARGET_MODEL"; exit $? ;;
            --help|-h)
                head -20 "$0" | grep '^#' | sed 's/^# \?//'
                exit 0
                ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done

    # Main: run recovery
    recover_ollama "$TARGET_MODEL"
    exit $?
fi
