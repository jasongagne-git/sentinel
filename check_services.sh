#!/usr/bin/env bash
# SENTINEL — System Service Auditor
#
# Checks for services consuming memory and optionally stops/disables
# them to maximize RAM for model inference.
#
# Usage:
#   ./check_services.sh [OPTIONS]
#
#   --report             Show reclaimable memory report (default action)
#   --stop               Stop stoppable services (current session only)
#   --disable            Stop AND disable on boot
#   --enable             Re-enable and start all managed services
#   --keep SVC[,SVC...]  Exclude services (comma-separated, additive with config)
#   --min-mb MB          Only report services using >= MB (default: 1)
#   --quiet              Machine-readable output (reclaimable MB on stdout)
#   --yes                Skip confirmation prompt for --stop/--disable
#   --init               Generate default config files in ~/.config/sentinel/
#
# Configuration:
#   ~/.config/sentinel/services.conf       Services to manage (stop list)
#   ~/.config/sentinel/keep-services.conf  Services to never stop (keep list)
#
#   Run --init to generate both config files with documented defaults.
#   The script will not stop ANY services until services.conf exists.
#
# Can also be sourced for functions:
#   source check_services.sh --source-only
#   audit_services        # returns reclaimable MB
#   stop_services         # stops services from config (minus keep list)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Defaults ───────────────────────────────────────────────────

ACTION="report"
KEEP_LIST=()
MIN_MB=1
QUIET=false
YES=false
SOURCE_ONLY=false

CONF_DIR="${SENTINEL_CONF_DIR:-${HOME}/.config/sentinel}"
SERVICES_CONF="${CONF_DIR}/services.conf"
KEEP_CONF="${CONF_DIR}/keep-services.conf"

# Populated by load_services_conf / load_keep_conf
MANAGED_SERVICES=()

# ── Config file loading ───────────────────────────────────────

# Parse a config file into an array.
# Format: one entry per line, "service_name  # optional description"
# Lines starting with # are comments. Blank lines ignored.
# Returns entries as "unit_name:description" in the target array.
_load_conf_entries() {
    local file="$1"
    local -n _target="$2"
    [ -f "$file" ] || return 1

    while IFS= read -r line; do
        # Strip leading/trailing whitespace
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"

        # Skip blank lines and pure comments
        [[ -z "$line" || "$line" == \#* ]] && continue

        local svc desc=""
        # Split on # for inline comment (used as description)
        if [[ "$line" == *"#"* ]]; then
            svc="${line%%#*}"
            desc="${line#*#}"
            # Trim whitespace
            svc="${svc#"${svc%%[![:space:]]*}"}"
            svc="${svc%"${svc##*[![:space:]]}"}"
            desc="${desc#"${desc%%[![:space:]]*}"}"
            desc="${desc%"${desc##*[![:space:]]}"}"
        else
            svc="$line"
        fi

        # Append .service if not present
        [[ "$svc" == *.service ]] || svc="${svc}.service"

        _target+=("${svc}:${desc}")
    done < "$file"
    return 0
}

# Load the managed services list from config
load_services_conf() {
    MANAGED_SERVICES=()
    if [ -f "$SERVICES_CONF" ]; then
        _load_conf_entries "$SERVICES_CONF" MANAGED_SERVICES
        return 0
    fi
    return 1
}

# Load the keep list from config and --keep flags
load_keep_conf() {
    if [ -f "$KEEP_CONF" ]; then
        while IFS= read -r line; do
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"
            [[ -z "$line" || "$line" == \#* ]] && continue
            line="${line%%#*}"
            line="${line#"${line%%[![:space:]]*}"}"
            line="${line%"${line##*[![:space:]]}"}"
            [ -z "$line" ] && continue
            [[ "$line" == *.service ]] || line="${line}.service"
            KEEP_LIST+=("$line")
        done < "$KEEP_CONF"
    fi
}

# ── Init: generate default config files ───────────────────────

generate_default_configs() {
    mkdir -p "$CONF_DIR"

    if [ -f "$SERVICES_CONF" ]; then
        echo "services.conf already exists: $SERVICES_CONF"
        echo "  Delete it first to regenerate, or edit it directly."
    else
        cat > "$SERVICES_CONF" << 'SERVICES_EOF'
# SENTINEL — Managed Services Configuration
#
# Services listed here will be candidates for stopping/disabling
# when SENTINEL needs to free memory for model inference.
#
# Format: service_name  # optional description
# Lines starting with # are comments.
#
# Only UNCOMMENTED lines are active. Review and uncomment the
# services appropriate for your system.
#
# To see what's running: systemctl list-units --type=service --state=running

# ── Container runtimes (large, usually idle on dev boards) ────
docker             # Docker container runtime
containerd         # Container runtime (Docker dependency)

# ── Desktop/GUI services (unnecessary on headless systems) ────
pulseaudio         # PulseAudio sound server
bluetooth          # Bluetooth support
avahi-daemon       # mDNS/DNS-SD discovery
udisks2            # Desktop disk management
switcheroo-control # GPU switching (desktop)
accounts-daemon    # Desktop account management
power-profiles-daemon  # Desktop power profiles

# ── Print services ────────────────────────────────────────────
snapd              # Snap package daemon
snap.cups.cups-browsed  # CUPS printer browsing (snap)
snap.cups.cupsd    # CUPS print server (snap)
lpd                # BSD line printer daemon

# ── Network services (disable if not used) ────────────────────
# smbd             # Samba file sharing
# nmbd             # Samba NetBIOS name service
ModemManager       # Modem management
rpcbind            # RPC portmap (NFS)

# ── Misc ──────────────────────────────────────────────────────
kerneloops         # Kernel crash reporter
SERVICES_EOF
        echo "Created: $SERVICES_CONF"
    fi

    if [ -f "$KEEP_CONF" ]; then
        echo "keep-services.conf already exists: $KEEP_CONF"
        echo "  Delete it first to regenerate, or edit it directly."
    else
        cat > "$KEEP_CONF" << 'KEEP_EOF'
# SENTINEL — Services to NEVER stop
#
# Services listed here are excluded from management even if they
# appear in services.conf. Use this to protect services you need.
#
# Format: service_name (one per line, # comments allowed)
#
# Example: uncomment to keep Samba running for LAN file access
# smbd
# nmbd
KEEP_EOF
        echo "Created: $KEEP_CONF"
    fi

    echo ""
    echo "Edit these files to match your system:"
    echo "  $SERVICES_CONF"
    echo "  $KEEP_CONF"
}

# ── Functions ──────────────────────────────────────────────────

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Check if a service is in the keep list
is_kept() {
    local svc="$1"
    [ "${#KEEP_LIST[@]}" -eq 0 ] && return 1
    for kept in "${KEEP_LIST[@]}"; do
        [ "$svc" = "$kept" ] && return 0
    done
    return 1
}

# Get RSS of a service's main PID in MB (integer)
service_rss_mb() {
    local svc="$1"
    local pid
    pid=$(systemctl show "$svc" --property=MainPID --value 2>/dev/null)
    if [ -n "$pid" ] && [ "$pid" != "0" ]; then
        local rss_kb
        rss_kb=$(awk '/^VmRSS:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
        echo $(( rss_kb / 1024 ))
    else
        echo 0
    fi
}

# Get state of a service
service_state() {
    systemctl is-active "$1" 2>/dev/null
}

# Build list of running managed services (not in keep list)
# Sets: SVC_NAMES[], SVC_DESCS[], SVC_MBS[], TOTAL_RECLAIMABLE
build_audit() {
    SVC_NAMES=()
    SVC_DESCS=()
    SVC_MBS=()
    TOTAL_RECLAIMABLE=0

    # Load config if not already loaded
    if [ "${#MANAGED_SERVICES[@]}" -eq 0 ]; then
        if ! load_services_conf; then
            [ "$QUIET" != true ] && \
                echo "[check_services] No services.conf found. Run: $0 --init" >&2
            return 1
        fi
    fi

    for entry in "${MANAGED_SERVICES[@]}"; do
        local svc="${entry%%:*}"
        local desc="${entry#*:}"

        is_kept "$svc" && continue

        local state
        state=$(service_state "$svc")
        [ "$state" != "active" ] && continue

        local mb
        mb=$(service_rss_mb "$svc")
        [ "$mb" -lt "$MIN_MB" ] && continue

        SVC_NAMES+=("$svc")
        SVC_DESCS+=("$desc")
        SVC_MBS+=("$mb")
        TOTAL_RECLAIMABLE=$(( TOTAL_RECLAIMABLE + mb ))
    done
}

# ── Actions ────────────────────────────────────────────────────

# Print a human-readable report
audit_services() {
    build_audit

    if [ "$QUIET" = true ]; then
        echo "$TOTAL_RECLAIMABLE"
        return 0
    fi

    if [ "${#SVC_NAMES[@]}" -eq 0 ]; then
        echo "No stoppable services running (or all are in keep-list)."
        return 0
    fi

    echo "Services managed by SENTINEL (from services.conf):"
    echo ""
    printf "  %-45s %8s  %s\n" "SERVICE" "RSS (MB)" "DESCRIPTION"
    printf "  %-45s %8s  %s\n" "-------" "--------" "-----------"
    for i in "${!SVC_NAMES[@]}"; do
        printf "  %-45s %6d    %s\n" "${SVC_NAMES[$i]}" "${SVC_MBS[$i]}" "${SVC_DESCS[$i]}"
    done
    echo ""
    echo "  Total reclaimable: ~${TOTAL_RECLAIMABLE} MB"
    echo ""

    local free_mb
    free_mb=$(awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo)
    echo "  Current available RAM: ${free_mb} MB"
    echo "  After reclaim:        ~$(( free_mb + TOTAL_RECLAIMABLE )) MB"
    echo ""

    if [ "${#KEEP_LIST[@]}" -gt 0 ]; then
        echo "  Kept (excluded): ${KEEP_LIST[*]}"
        echo ""
    fi

    echo "  Config: $SERVICES_CONF"
    echo "  Keep:   $KEEP_CONF"
    echo ""

    return 0
}

# Stop services (session only)
stop_services() {
    build_audit

    if [ "${#SVC_NAMES[@]}" -eq 0 ]; then
        [ "$QUIET" != true ] && echo "Nothing to stop."
        return 0
    fi

    if [ "$YES" != true ] && [ "$QUIET" != true ]; then
        echo "Will stop ${#SVC_NAMES[@]} services (freeing ~${TOTAL_RECLAIMABLE} MB):"
        for i in "${!SVC_NAMES[@]}"; do
            echo "  - ${SVC_NAMES[$i]} (${SVC_MBS[$i]} MB)"
        done
        echo ""
        read -r -p "Proceed? [y/N] " answer
        [[ "$answer" =~ ^[Yy] ]] || { echo "Aborted."; return 1; }
    fi

    local stopped=0
    local freed=0
    for i in "${!SVC_NAMES[@]}"; do
        local svc="${SVC_NAMES[$i]}"
        if sudo systemctl stop "$svc" 2>/dev/null; then
            [ "$QUIET" != true ] && echo "  Stopped: $svc (${SVC_MBS[$i]} MB)"
            stopped=$((stopped + 1))
            freed=$((freed + SVC_MBS[$i]))
        else
            [ "$QUIET" != true ] && echo "  FAILED:  $svc"
        fi
    done

    [ "$QUIET" != true ] && echo "Stopped $stopped services, freed ~${freed} MB."
    return 0
}

# Disable services (persistent across reboot)
disable_services() {
    build_audit

    if [ "${#SVC_NAMES[@]}" -eq 0 ]; then
        [ "$QUIET" != true ] && echo "Nothing to disable."
        return 0
    fi

    if [ "$YES" != true ] && [ "$QUIET" != true ]; then
        echo "Will stop AND disable ${#SVC_NAMES[@]} services (freeing ~${TOTAL_RECLAIMABLE} MB):"
        for i in "${!SVC_NAMES[@]}"; do
            echo "  - ${SVC_NAMES[$i]} (${SVC_MBS[$i]} MB)"
        done
        echo ""
        echo "Services will NOT start on next boot. Re-enable with: $0 --enable"
        echo ""
        read -r -p "Proceed? [y/N] " answer
        [[ "$answer" =~ ^[Yy] ]] || { echo "Aborted."; return 1; }
    fi

    local count=0
    for i in "${!SVC_NAMES[@]}"; do
        local svc="${SVC_NAMES[$i]}"
        if sudo systemctl disable --now "$svc" 2>/dev/null; then
            [ "$QUIET" != true ] && echo "  Disabled: $svc"
            count=$((count + 1))
        else
            [ "$QUIET" != true ] && echo "  FAILED:   $svc"
        fi
    done

    [ "$QUIET" != true ] && echo "Disabled $count services."
    return 0
}

# Re-enable all managed services (except kept ones)
enable_services() {
    if [ "${#MANAGED_SERVICES[@]}" -eq 0 ]; then
        load_services_conf || { echo "No services.conf found."; return 1; }
    fi

    local count=0
    for entry in "${MANAGED_SERVICES[@]}"; do
        local svc="${entry%%:*}"
        is_kept "$svc" && continue

        local enabled
        enabled=$(systemctl is-enabled "$svc" 2>/dev/null)
        if [ "$enabled" = "disabled" ] || [ "$enabled" = "masked" ]; then
            if sudo systemctl enable --now "$svc" 2>/dev/null; then
                [ "$QUIET" != true ] && echo "  Enabled: $svc"
                count=$((count + 1))
            fi
        fi
    done
    [ "$QUIET" != true ] && echo "Re-enabled $count services."
    return 0
}

# ── Preflight hook (called from other scripts) ────────────────

# Quick check: returns 0 if enough memory, 1 if services should be stopped.
# Usage: preflight_memory_check 4200   # need 4200 MB for model
preflight_memory_check() {
    local need_mb="${1:-4200}"
    local avail_mb
    avail_mb=$(awk '/^MemAvailable:/ {printf "%d", $2/1024}' /proc/meminfo)

    build_audit

    if [ "$avail_mb" -ge "$need_mb" ]; then
        return 0
    fi

    local after_reclaim=$(( avail_mb + TOTAL_RECLAIMABLE ))
    if [ "$after_reclaim" -ge "$need_mb" ]; then
        echo "[check_services] Need ${need_mb} MB, have ${avail_mb} MB."
        echo "[check_services] Can reclaim ~${TOTAL_RECLAIMABLE} MB by stopping ${#SVC_NAMES[@]} services."
        return 1
    else
        echo "[check_services] Need ${need_mb} MB, have ${avail_mb} MB (${after_reclaim} MB after reclaim)."
        echo "[check_services] WARNING: May not have enough memory even after stopping services."
        return 1
    fi
}

# ── Argument parsing ──────────────────────────────────────────

if [[ "${1:-}" == "--source-only" ]]; then
    SOURCE_ONLY=true
    # Load configs when sourced
    load_services_conf 2>/dev/null || true
    load_keep_conf
fi

if [ "$SOURCE_ONLY" = false ]; then
    # Load configs
    load_services_conf 2>/dev/null || true
    load_keep_conf

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --report)    ACTION="report"; shift ;;
            --stop)      ACTION="stop"; shift ;;
            --disable)   ACTION="disable"; shift ;;
            --enable)    ACTION="enable"; shift ;;
            --init)      generate_default_configs; exit 0 ;;
            --keep)
                IFS=',' read -ra extras <<< "$2"
                for e in "${extras[@]}"; do
                    [[ "$e" == *.service ]] || e="${e}.service"
                    KEEP_LIST+=("$e")
                done
                shift 2
                ;;
            --min-mb)    MIN_MB="$2"; shift 2 ;;
            --quiet)     QUIET=true; shift ;;
            --yes)       YES=true; shift ;;
            --help|-h)
                head -30 "$0" | grep '^#' | sed 's/^# \?//'
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Run $0 --help for usage."
                exit 1
                ;;
        esac
    done

    # Guard: refuse to stop/disable without a config
    if [[ "$ACTION" =~ ^(stop|disable)$ ]] && [ "${#MANAGED_SERVICES[@]}" -eq 0 ]; then
        echo "No services.conf found. Run '$0 --init' to create one."
        echo "The script will not stop any services without explicit configuration."
        exit 1
    fi

    case "$ACTION" in
        report)  audit_services ;;
        stop)    stop_services ;;
        disable) disable_services ;;
        enable)  enable_services ;;
    esac
fi
