#!/usr/bin/env bash
# Auto-supervise a Claude Code session in Cursor: poll every N seconds and
# approve permission prompts (Bash "Do you want to proceed?" etc.).
#
# Usage:
#   ./scripts/supervise_claude.sh              # foreground, 10s interval
#   ./scripts/supervise_claude.sh --bg         # background + log file
#   ./scripts/supervise_claude.sh --stop       # stop background supervisor
#   ./scripts/supervise_claude.sh --status     # show running/stopped
#
# Screen lock grabs the keyboard, so xdotool cannot type. This script holds a
# systemd idle/sleep inhibitor so the session should not lock while it runs.
# Unlock once if already locked; after that, keep the session open.

set -euo pipefail

INTERVAL="${SUPERVISOR_INTERVAL:-10}"
LOG="${SUPERVISOR_LOG:-/tmp/claude_supervisor.log}"
PIDFILE="${SUPERVISOR_PIDFILE:-/tmp/claude_supervisor.pid}"
TERMINALS_DIR="${CURSOR_TERMINALS_DIR:-$HOME/.cursor/projects/home-yash-Desktop-Sih/terminals}"

session_locked() {
    local sid
    sid="${XDG_SESSION_ID:-}"
    if [[ -z "$sid" ]]; then
        sid="$(loginctl --no-legend | awk '/seat/ {print $1; exit}')"
    fi
    [[ -n "$sid" ]] || return 1
    loginctl show-session "$sid" -p LockedHint --value 2>/dev/null | grep -qx yes
}

# Write bytes into the PTY *master* that Cursor holds for Claude's terminal.
# Writing the slave (/dev/pts/N) only prints on screen; it does not type.
# This works while the screen is locked because it never uses X11.
inject_pty_keys() {
    python3 - <<'PY'
import os, sys

keys = b"\r"

def claude_pid():
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            comm = open(f"/proc/{name}/comm").read().strip()
        except OSError:
            continue
        if comm == "claude":
            return int(name)
    return None

def tty_index_of_pid(pid):
    try:
        target = os.readlink(f"/proc/{pid}/fd/0")
    except OSError:
        return None
    prefix = "/dev/pts/"
    if target.startswith(prefix) and target[len(prefix):].isdigit():
        return int(target[len(prefix):])
    return None

def find_master_path(index):
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        fd_dir = f"/proc/{name}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd_num in fds:
            path = f"{fd_dir}/{fd_num}"
            try:
                if os.readlink(path) != "/dev/ptmx":
                    continue
            except OSError:
                continue
            try:
                info = open(f"/proc/{name}/fdinfo/{fd_num}").read()
            except OSError:
                continue
            for line in info.splitlines():
                if line.startswith("tty-index:"):
                    try:
                        got = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        continue
                    if got == index:
                        return path
    return None

pid = claude_pid()
if pid is None:
    print("no claude process", file=sys.stderr)
    sys.exit(1)
idx = tty_index_of_pid(pid)
if idx is None:
    print(f"claude {pid} has no pts tty", file=sys.stderr)
    sys.exit(1)
master = find_master_path(idx)
if master is None:
    print(f"no ptmx master for pts/{idx}", file=sys.stderr)
    sys.exit(1)
fd = os.open(master, os.O_WRONLY | os.O_NOCTTY)
try:
    n = os.write(fd, keys)
finally:
    os.close(fd)
print(f"wrote {n} bytes to pts/{idx} via {master}")
PY
}

approve_prompt() {
    # xdotool is the only method that Claude's TUI actually accepts.
    # It cannot type while the lock screen has the keyboard.
    if session_locked; then
        log_line "WARN: screen is locked — unlock once; idle lock is inhibited while this script runs"
        return 1
    fi

    local wid title best_wid="" best_len=0
    while read -r wid; do
        [[ -n "$wid" ]] || continue
        title="$(xdotool getwindowname "$wid" 2>/dev/null || true)"
        [[ "$title" == *Cursor* ]] || continue
        if ((${#title} > best_len)); then
            best_wid="$wid"
            best_len=${#title}
        fi
    done < <(xdotool search --name "Cursor" 2>/dev/null || true)

    if [[ -z "$best_wid" ]]; then
        log_line "WARN: Cursor window not found"
        return 1
    fi

    xdotool windowactivate --sync "$best_wid" 2>/dev/null || true
    sleep 0.25
    # Do not send ctrl+grave — that toggles the terminal panel closed.
    xdotool key --window "$best_wid" 1 Return 2>/dev/null || true
    log_line "xdotool 1+Enter sent to window $best_wid"
    return 0
}

log_line() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >>"$LOG"
}

find_claude_terminal() {
    local f
    for f in "$TERMINALS_DIR"/*.txt; do
        [[ -f "$f" ]] || continue
        if head -5 "$f" | grep -q 'active_command: claude'; then
            echo "$f"
            return 0
        fi
    done
    return 1
}

needs_approval() {
    local file="$1"
    # Only the live tail — older prompts stay in the terminal log forever.
    local tail_txt
    tail_txt="$(tail -n 40 "$file")"
    printf '%s\n' "$tail_txt" | grep -q 'Do you want to proceed?' || return 1
    printf '%s\n' "$tail_txt" | grep -q '❯ 1. Yes' || return 1
    return 0
}

prompt_fingerprint() {
    local file="$1"
    # Hash the bash command block above the permission dialog (unique per prompt).
    awk '
        /Bash command/ { capture=1; block="" }
        capture { block = block $0 "\n" }
        /Do you want to proceed\?/ { print block; exit }
    ' "$file" | md5sum | awk '{print $1}'
}

run_loop() {
    log_line "Supervisor started (interval=${INTERVAL}s, terminals=$TERMINALS_DIR)"
    local last_approved=""
    local last_attempt_epoch=0

    while true; do
        local term_file=""
        term_file="$(find_claude_terminal || true)"

        if [[ -z "$term_file" ]]; then
            log_line "No active 'claude' terminal found — waiting"
        elif needs_approval "$term_file"; then
            local fingerprint now_epoch
            fingerprint="$(prompt_fingerprint "$term_file")"
            now_epoch="$(date +%s)"
            # Re-try if command changed, or same prompt still visible after 15s.
            if [[ "$fingerprint" != "$last_approved" ]] \
                || (( now_epoch - last_attempt_epoch >= 15 )); then
                log_line "Approval needed in $(basename "$term_file") — sending Enter (Yes)"
                last_attempt_epoch="$now_epoch"
                if approve_prompt; then
                    last_approved="$fingerprint"
                    log_line "Approved ($fingerprint)"
                else
                    log_line "Approve attempt failed"
                fi
            fi
        else
            last_approved=""
        fi

        # Keep DE screensaver from locking even if idle-inhibit is ignored.
        xdg-screensaver reset >/dev/null 2>&1 || true

        sleep "$INTERVAL"
    done
}

stop_supervisor() {
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid="$(cat "$PIDFILE")"
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            rm -f "$PIDFILE"
            echo "Stopped supervisor (pid $pid)"
        else
            rm -f "$PIDFILE"
            echo "Supervisor not running (stale pidfile removed)"
        fi
    else
        echo "Supervisor not running"
    fi
}

status_supervisor() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "Running (pid $(cat "$PIDFILE"))"
        echo "Log: $LOG"
        tail -5 "$LOG" 2>/dev/null || true
    else
        echo "Stopped"
    fi
}

case "${1:-}" in
    --stop)
        stop_supervisor
        ;;
    --status)
        status_supervisor
        ;;
    --bg|--background)
        if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Already running (pid $(cat "$PIDFILE"))"
            exit 0
        fi
        nohup systemd-inhibit \
            --what=idle:sleep \
            --who=claude-supervisor \
            --why="Claude Code permission approvals need an unlocked session" \
            --mode=block \
            "$0" >/dev/null 2>&1 &
        echo $! >"$PIDFILE"
        echo "Supervisor started in background (pid $(cat "$PIDFILE"))"
        echo "Idle/sleep lock is inhibited while this runs. Do not lock the screen."
        echo "Log: $LOG"
        ;;
    -h|--help)
        sed -n '2,12p' "$0"
        ;;
    *)
        run_loop
        ;;
esac
