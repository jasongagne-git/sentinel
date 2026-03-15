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

# SENTINEL Thermal Guard — sourceable thermal monitoring for Jetson
# Source with: source ./thermal_guard.sh --source-only
# Or run standalone: ./thermal_guard.sh --status
#
# Reads tj-thermal (junction temperature) which is the max across all
# Jetson thermal domains. Temps in sysfs are millidegrees C.

# --- Configuration (override before sourcing if needed) ---
THERMAL_ZONE="${THERMAL_ZONE:-/sys/class/thermal/thermal_zone8}"  # tj-thermal
THERMAL_WARN_C="${THERMAL_WARN_C:-70}"       # Warning: add extra delays
THERMAL_CRIT_C="${THERMAL_CRIT_C:-82}"       # Critical: full pause
THERMAL_RESUME_C="${THERMAL_RESUME_C:-68}"   # Resume threshold (hysteresis)
THERMAL_POLL_S="${THERMAL_POLL_S:-10}"        # Poll interval during pause
THERMAL_MAX_WAIT="${THERMAL_MAX_WAIT:-600}"   # Max seconds to wait for cooldown

# --- Functions ---

# Read current junction temperature in degrees C
thermal_read_temp() {
    local raw
    raw=$(cat "$THERMAL_ZONE/temp" 2>/dev/null)
    if [ -z "$raw" ]; then
        echo "ERROR"
        return 1
    fi
    echo $(( raw / 1000 ))
}

# Read millidegree precision (for logging)
thermal_read_temp_precise() {
    local raw
    raw=$(cat "$THERMAL_ZONE/temp" 2>/dev/null)
    if [ -z "$raw" ]; then
        echo "ERROR"
        return 1
    fi
    local whole=$(( raw / 1000 ))
    local frac=$(( (raw % 1000) / 100 ))
    echo "${whole}.${frac}"
}

# Check thermal state. Returns:
#   0 = OK (below warning)
#   1 = WARN (above warning, below critical)
#   2 = CRITICAL (above critical)
thermal_check() {
    local temp
    temp=$(thermal_read_temp)
    if [ "$temp" = "ERROR" ]; then
        return 0  # Can't read — don't block on sensor failure
    fi
    if [ "$temp" -ge "$THERMAL_CRIT_C" ]; then
        return 2
    elif [ "$temp" -ge "$THERMAL_WARN_C" ]; then
        return 1
    fi
    return 0
}

# Wait for temperature to drop below resume threshold.
# Calls an optional callback to unload models during cooldown.
# Usage: thermal_wait_cooldown [unload_callback] [log_callback]
thermal_wait_cooldown() {
    local unload_fn="${1:-}"
    local log_fn="${2:-echo}"
    local waited=0

    local temp
    temp=$(thermal_read_temp_precise)
    $log_fn "THERMAL: Critical temperature ${temp}°C >= ${THERMAL_CRIT_C}°C — pausing"

    # Unload models to help cool down
    if [ -n "$unload_fn" ] && type "$unload_fn" &>/dev/null; then
        $log_fn "THERMAL: Unloading models to accelerate cooldown"
        $unload_fn
    fi

    while true; do
        sleep "$THERMAL_POLL_S"
        waited=$(( waited + THERMAL_POLL_S ))

        temp=$(thermal_read_temp)
        if [ "$temp" = "ERROR" ]; then
            $log_fn "THERMAL: WARNING — cannot read sensor, resuming"
            return 0
        fi

        if [ "$temp" -le "$THERMAL_RESUME_C" ]; then
            $log_fn "THERMAL: Cooled to ${temp}°C (<= ${THERMAL_RESUME_C}°C) after ${waited}s — resuming"
            return 0
        fi

        if [ "$waited" -ge "$THERMAL_MAX_WAIT" ]; then
            $log_fn "THERMAL: WARNING — still ${temp}°C after ${THERMAL_MAX_WAIT}s max wait — resuming anyway"
            return 1
        fi

        # Log every 60s during cooldown
        if (( waited % 60 == 0 )); then
            $log_fn "THERMAL: Waiting... ${temp}°C (target <= ${THERMAL_RESUME_C}°C, ${waited}s elapsed)"
        fi
    done
}

# All-in-one: check thermal, pause if critical, return extra delay if warm.
# Usage: extra_delay=$(thermal_guard [unload_callback] [log_callback])
# Returns extra delay seconds on stdout (0=ok, >0=warm, "PAUSED"=was critical)
thermal_guard() {
    local unload_fn="${1:-}"
    local log_fn="${2:-echo}"

    thermal_check
    local state=$?

    case $state in
        0)
            echo 0
            ;;
        1)
            local temp
            temp=$(thermal_read_temp_precise)
            $log_fn "THERMAL: Warm ${temp}°C — adding delay"
            echo 10
            ;;
        2)
            thermal_wait_cooldown "$unload_fn" "$log_fn"
            echo "PAUSED"
            ;;
    esac
}

# Print status summary
thermal_status() {
    local temp
    temp=$(thermal_read_temp_precise)
    local zone_type
    zone_type=$(cat "$THERMAL_ZONE/type" 2>/dev/null || echo "unknown")

    echo "Thermal zone: $zone_type ($THERMAL_ZONE)"
    echo "Temperature:  ${temp}°C"
    echo "Thresholds:   warn=${THERMAL_WARN_C}°C  crit=${THERMAL_CRIT_C}°C  resume=${THERMAL_RESUME_C}°C"

    thermal_check
    local state=$?
    case $state in
        0) echo "Status:       OK" ;;
        1) echo "Status:       WARNING" ;;
        2) echo "Status:       CRITICAL" ;;
    esac
}

# --- Main (standalone mode) ---
if [ "${1:-}" = "--status" ]; then
    thermal_status
    exit 0
fi

if [ "${1:-}" != "--source-only" ]; then
    thermal_status
fi
