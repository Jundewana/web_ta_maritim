#!/bin/bash
# Buka relay TCP dari node interaktif ini (tempat VSCode Remote-SSH nyambung)
# ke frontend yang jalan di node compute SLURM (lihat web/run_web.sh).
# Diperlukan karena node compute (mis. "a100") tidak bisa diakses SSH
# langsung -- relay ini lewat `srun --overlap` (port_relay.py) yang memang
# diizinkan untuk job milik sendiri.
#
# Setelah relay ini jalan, buka panel "Ports" di VSCode -> Forward a Port
# -> isi 5173 -> klik URL yang muncul.
#
# Usage:
#   bash web/open_relay.sh <jobid>       # default port 5173
#   bash web/open_relay.sh <jobid> stop

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEB_DIR="$REPO_ROOT/web"
LOG_DIR="$WEB_DIR/logs"
PID_FILE="$LOG_DIR/relay.pid"
LOG_FILE="$LOG_DIR/relay.log"
PORT=5173

mkdir -p "$LOG_DIR"

stop_relay() {
    if [[ -f "$PID_FILE" ]]; then
        OLD_PID=$(cat "$PID_FILE")
        kill "$OLD_PID" 2>/dev/null
        rm -f "$PID_FILE"
    fi
    pkill -f "web/port_relay.py" 2>/dev/null
}

JOBID="${1:-}"
if [[ -z "$JOBID" ]]; then
    echo "Usage: bash web/open_relay.sh <jobid_run_web> [stop]"
    exit 1
fi

stop_relay
if [[ "${2:-}" == "stop" ]]; then
    echo "Relay dihentikan."
    exit 0
fi

source /home/mdewana/ENTER/etc/profile.d/conda.sh
conda activate minirag

nohup python3 "$WEB_DIR/port_relay.py" --jobid "$JOBID" --port "$PORT" > "$LOG_FILE" 2>&1 &
disown
echo $! > "$PID_FILE"
sleep 2
echo "Relay jalan (PID $(cat "$PID_FILE")) -- 127.0.0.1:$PORT di node ini -> job $JOBID."
echo "Buka panel 'Ports' di VSCode, forward port $PORT, lalu klik URL yang muncul."
