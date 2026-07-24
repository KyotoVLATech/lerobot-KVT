#!/usr/bin/env bash
# Occupy GPUs (memory + utilization) to keep an allocation alive against idle reclaim.
#
# Usage:
#   bash scripts/occupy_gpus.sh                   # all visible GPUs
#   bash scripts/occupy_gpus.sh 0,1,2,3           # specific GPUs
#   MEM_GB=18 MAT_DIM=8192 bash scripts/occupy_gpus.sh
#   PYTHON_BIN=/path/to/python bash scripts/occupy_gpus.sh
#
# Background (survives logout):
#   nohup bash scripts/occupy_gpus.sh >/tmp/occupy_gpus.out 2>&1 &
#   tail -f /tmp/occupy_gpus/gpu_0.log
#
# Stop:
#   Ctrl+C  in foreground, or:   pkill -f occupy_gpus.sh

set -uo pipefail

GPUS="${1:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | paste -sd ',')}"

: "${CONDA_ENV:=dp3}"
: "${PYTHON_BIN:=}"
: "${MEM_GB:=18}"             # memory to reserve per GPU (GB)
: "${MAT_DIM:=8192}"          # matmul size (square)
: "${BURST_MS:=600}"          # ms of compute per cycle
: "${SLEEP_MS:=80}"           # idle gap (with jitter) to look less synthetic
: "${HEARTBEAT_SEC:=30}"      # how often each worker prints status
: "${LOGDIR:=/tmp/occupy_gpus}"

# Resolve python: explicit PYTHON_BIN > conda env > current python
if [ -z "$PYTHON_BIN" ]; then
    for cand in \
        "$HOME/miniconda3/envs/$CONDA_ENV/bin/python" \
        "$HOME/anaconda3/envs/$CONDA_ENV/bin/python" \
        "/opt/conda/envs/$CONDA_ENV/bin/python" \
        "/root/miniconda3/envs/$CONDA_ENV/bin/python"; do
        if [ -x "$cand" ]; then PYTHON_BIN="$cand"; break; fi
    done
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! "$PYTHON_BIN" -c "import torch" 2>/dev/null; then
    echo "[occupy] ERROR: $PYTHON_BIN cannot import torch."
    echo "         Set PYTHON_BIN=/path/to/python or CONDA_ENV=<env_with_torch>."
    exit 1
fi

mkdir -p "$LOGDIR"

echo "[occupy] python=$PYTHON_BIN"
echo "[occupy] GPUs=$GPUS  MEM_GB=$MEM_GB  MAT_DIM=$MAT_DIM"
echo "[occupy] burst=${BURST_MS}ms  sleep=${SLEEP_MS}ms (jittered)  logs=$LOGDIR"

IFS=',' read -ra GPU_ARRAY <<<"$GPUS"
pids=()

cleanup() {
    echo
    echo "[occupy] releasing ${#pids[@]} workers..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "[occupy] done."
    exit 0
}
trap cleanup INT TERM

for gpu in "${GPU_ARRAY[@]}"; do
    log="$LOGDIR/gpu_${gpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" \
    MEM_GB="$MEM_GB" MAT_DIM="$MAT_DIM" \
    BURST_MS="$BURST_MS" SLEEP_MS="$SLEEP_MS" \
    HEARTBEAT_SEC="$HEARTBEAT_SEC" GPU_LABEL="$gpu" \
    "$PYTHON_BIN" -u -c '
import os, sys, time, signal, random
import torch

GPU       = os.environ["GPU_LABEL"]
MEM_GB    = float(os.environ["MEM_GB"])
MAT_DIM   = int(os.environ["MAT_DIM"])
BURST     = float(os.environ["BURST_MS"]) / 1000.0
SLEEP     = float(os.environ["SLEEP_MS"]) / 1000.0
HEARTBEAT = float(os.environ["HEARTBEAT_SEC"])

dev = torch.device("cuda:0")
torch.cuda.set_device(dev)

# Reserve memory in 1 GB float32 chunks, stopping just before OOM.
chunks = []
chunk_elems = (1 << 30) // 4
target_bytes = int(MEM_GB * (1 << 30))
while sum(c.element_size() * c.numel() for c in chunks) + (1 << 30) <= target_bytes:
    try:
        chunks.append(torch.empty(chunk_elems, dtype=torch.float32, device=dev))
    except RuntimeError as e:
        print(f"[gpu {GPU}] mem reserve stopped at {len(chunks)} GB: {e}", flush=True)
        break

a = torch.randn(MAT_DIM, MAT_DIM, device=dev)
b = torch.randn(MAT_DIM, MAT_DIM, device=dev)
acc = torch.zeros(MAT_DIM, MAT_DIM, device=dev)   # keep result live so kernels are not elided

print(f"[gpu {GPU}] reserved ~{torch.cuda.memory_allocated()/1e9:.1f} GB, "
      f"matmul {MAT_DIM}x{MAT_DIM}, burst={BURST*1000:.0f} ms", flush=True)

signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))

step = 0
t_last = time.time()
while True:
    t0 = time.time()
    while time.time() - t0 < BURST:
        torch.mm(a, b, out=acc)
        a.add_(acc, alpha=1e-12)        # tiny in-place op so JIT cannot drop it
    torch.cuda.synchronize()
    if SLEEP > 0:
        time.sleep(SLEEP * (0.5 + random.random()))
    step += 1
    if time.time() - t_last > HEARTBEAT:
        print(f"[gpu {GPU}] step={step} mem={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
        t_last = time.time()
' >"$log" 2>&1 &
    pids+=($!)
    echo "[occupy] gpu $gpu -> pid $!  log=$log"
done

echo "[occupy] all ${#pids[@]} workers up. Ctrl+C to release."
wait
