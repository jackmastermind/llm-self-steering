#!/usr/bin/env bash
# Start one or more vllm-lens servers (one per GPU) for the model you want to
# run the experiments against. Each server runs in its own tmux session named
# `vllm0..vllm{N-1}` on ports BASE_PORT..BASE_PORT+N-1, so logs are inspectable
# (`tmux attach -t vllm0`). Re-running kills and replaces existing sessions.
#
# Usage:
#   bash scripts/start_vllm.sh                                  # 1 server, Qwen3-8B, port 8000
#   N_SERVERS=8 bash scripts/start_vllm.sh                      # 8 servers (8 GPUs), ports 8000..8007
#   MODEL=Qwen/Qwen3-32B bash scripts/start_vllm.sh            # 32B on a single GPU
#   MODEL=Qwen/Qwen3-32B TP=2 N_SERVERS=4 bash scripts/start_vllm.sh   # 32B, 2 GPUs/server
#
# Env knobs (all optional):
#   MODEL          HF model id served by vllm           (default Qwen/Qwen3-8B)
#   N_SERVERS      how many servers / GPU groups         (default 1)
#   BASE_PORT      first port; server i listens on +i    (default 8000)
#   TP             tensor-parallel size per server       (default 1)
#   MAX_MODEL_LEN  context length                        (default 32768)
#   GPU_UTIL       vllm --gpu-memory-utilization         (default 0.90)
#
# The script waits until /v1/models is reachable on every port before exiting.
# Then run the experiments with: uv run python scripts/run_experiments.py ...
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-8B}
N_SERVERS=${N_SERVERS:-1}
BASE_PORT=${BASE_PORT:-8000}
TP=${TP:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.90}

mkdir -p logs

echo "Launching $N_SERVERS vllm-lens server(s) serving $MODEL (TP=$TP)"
echo "  Ports: $BASE_PORT..$((BASE_PORT + N_SERVERS - 1))"
echo

for i in $(seq 0 $((N_SERVERS - 1))); do
  port=$((BASE_PORT + i))
  session="vllm$i"
  # Assign TP consecutive GPUs to this server.
  gpus=$(seq -s, $((i * TP)) $((i * TP + TP - 1)))
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "  $session: existing session — killing"
    tmux kill-session -t "$session"
  fi
  tmux new-session -d -s "$session" \
    "CUDA_VISIBLE_DEVICES=$gpus uv run vllm serve $MODEL \
       --port $port --max-model-len $MAX_MODEL_LEN \
       --enable-auto-tool-choice --tool-call-parser hermes \
       --tensor-parallel-size $TP --gpu-memory-utilization $GPU_UTIL \
       2>&1 | tee logs/vllm_$i.log"
  echo "  $session: launched on GPU(s) $gpus, port $port"
done

echo
echo "Waiting for all servers to become ready (polling /v1/models)..."
echo "Tail a server log with: tail -f logs/vllm_<i>.log"
echo

for i in $(seq 0 $((N_SERVERS - 1))); do
  port=$((BASE_PORT + i))
  start=$(date +%s)
  while ! curl -fs "http://localhost:$port/v1/models" >/dev/null 2>&1; do
    if ! tmux has-session -t "vllm$i" 2>/dev/null; then
      echo "  port $port: tmux session vllm$i died — check logs/vllm_$i.log" >&2
      exit 1
    fi
    sleep 5
  done
  echo "  port $port: ready (after $(( $(date +%s) - start ))s)"
done

echo
echo "All $N_SERVERS server(s) ready."
echo "Stop with: for i in \$(seq 0 $((N_SERVERS-1))); do tmux kill-session -t vllm\$i; done"
