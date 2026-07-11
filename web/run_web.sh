#!/bin/bash

#SBATCH --job-name=web_maritimchat
#SBATCH --partition=long
#SBATCH --nodelist=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=39G
#SBATCH --time=3-00:00:00

# ==============================================================================
# Menjalankan web app MaritimChat (FastAPI backend + Vite frontend) sebagai
# SBATCH job, supaya tetap hidup walau terminal/sesi SSH yang men-submit
# terputus (beda dengan nohup biasa di node interaktif, yang bisa ikut mati
# kalau node itu sendiri di-restart/di-reclaim).
#
# Backend & frontend dijalankan DI NODE YANG SAMA (satu job) karena
# web/frontend/vite.config.js sudah proxy /api -> http://localhost:8000,
# jadi keduanya harus co-located.
#
# Submit : sbatch web/run_web.sh        (dari root repo)
# Log    : slurm-<jobid>.out, atau web/logs/backend.log & web/logs/frontend.log
# Stop   : scancel <jobid>
# ==============================================================================

REPO_ROOT="/home/mdewana/GraphRAG/MiniRAG[gemma3]"
WEB_DIR="$REPO_ROOT/web"
LOG_DIR="$WEB_DIR/logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

echo "Initializing Conda..."
source /home/mdewana/ENTER/etc/profile.d/conda.sh
conda activate minirag

unset SSL_CERT_FILE

export OLLAMA_HOST="http://127.0.0.1:11500"
export OLLAMA_CONTEXT_LENGTH=131072
OLLAMA_PATH="/home/mdewana/.local/bin/bin/ollama"

echo "Starting Ollama server on $OLLAMA_HOST..."
$OLLAMA_PATH serve > "$LOG_DIR/ollama.log" 2>&1 &
OLLAMA_PID=$!

echo "Waiting 15 seconds for Ollama server (PID: $OLLAMA_PID) to start..."
sleep 15

if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "Ollama server GAGAL dimulai."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    conda deactivate
    exit 1
fi

echo "Ollama server berjalan. Memulai backend FastAPI..."
cd "$WEB_DIR/backend"
python3 main.py > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

echo "Menunggu backend siap..."
for i in $(seq 1 30); do
    if curl -s -o /dev/null http://localhost:8000/docs; then
        echo "Backend siap (PID: $BACKEND_PID)."
        break
    fi
    sleep 2
done

echo "Memulai frontend (Vite dev server)..."
cd "$WEB_DIR/frontend"
if [[ ! -d node_modules ]]; then
    echo "node_modules belum ada -- menjalankan npm install..."
    npm install
fi
npm run dev -- --host 0.0.0.0 > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

echo ""
echo "=================================================="
echo "Backend  PID $BACKEND_PID  -> log: $LOG_DIR/backend.log"
echo "Frontend PID $FRONTEND_PID -> log: $LOG_DIR/frontend.log (cari baris 'Local:'/'Network:' utk port & host)"
echo "Node SLURM: $(hostname)"
echo "=================================================="

# Kalau salah satu proses utama mati, matikan semua supaya job tidak
# menggantung memegang GPU tanpa guna -- lalu job selesai (SLURM bebaskan slot).
cleanup() {
    echo "Menghentikan semua proses..."
    kill "$FRONTEND_PID" "$BACKEND_PID" "$OLLAMA_PID" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

wait -n "$BACKEND_PID" "$FRONTEND_PID" "$OLLAMA_PID"
echo "Salah satu proses (backend/frontend/ollama) berhenti -- job selesai."
