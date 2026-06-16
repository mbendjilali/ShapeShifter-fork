#!/bin/bash
# Usage:
#   bash scripts/train_dales.sh 0      # single GPU
#   bash scripts/train_dales.sh 0 1    # dual GPU
#
# Logs: logs/dales_*.log  — dashboard refreshes every 15s
#
# GPU distribution (dual mode):
#   Phase 1 — GPU_A: diffusion 0          GPU_B: upsamplers 1-4
#   Phase 2 — GPU_A: diffusion 1, 2       GPU_B: diffusion 3, 4

GPU_A=${1:-0}
GPU_B=${2:-$GPU_A}

cd "$(dirname "$0")/.."
mkdir -p logs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

start() {
    # start <label> <gpu> <python args...>
    local label=$1 gpu=$2; shift 2
    CUDA_VISIBLE_DEVICES=$gpu python "$@" >"logs/${label}.log" 2>&1 &
    echo $!
}

progress() {
    local f="logs/$1.log"
    [ -f "$f" ] || { printf "  %-72s\n" "waiting..."; return; }
    local line
    line=$(grep -a '\[' "$f" 2>/dev/null | grep -a '/' | tail -1 | tr '\r' '\n' | tail -1)
    if [ -n "$line" ]; then
        printf "  %-72.72s\n" "$line"
    else
        local last
        last=$(grep -av '^$' "$f" 2>/dev/null | tail -1)
        printf "  %-72.72s\n" "${last:-starting...}"
    fi
}

alive() { kill -0 "$1" 2>/dev/null; }

status_of() {
    local pid=$1
    if alive "$pid"; then
        echo "RUNNING"
    else
        wait "$pid" 2>/dev/null
        [ $? -eq 0 ] && echo "DONE   " || echo "FAILED "
    fi
}

dashboard() {
    local -n _labels=$1
    local -n _pids=$2
    local -n _gpus=$3

    while true; do
        local all_done=true
        for pid in "${_pids[@]}"; do
            alive "$pid" && { all_done=false; break; }
        done

        clear
        if [ "$GPU_A" = "$GPU_B" ]; then
            printf "╔══ DALES — GPU %s — %s ══╗\n" "$GPU_A" "$(date '+%H:%M:%S')"
        else
            printf "╔══ DALES — GPU %s / %s — %s ══╗\n" "$GPU_A" "$GPU_B" "$(date '+%H:%M:%S')"
        fi
        printf "%-28s  %-9s  %-5s  %s\n" "Model" "Status" "GPU" "Progress / ETA"
        printf "%.0s─" {1..80}; echo

        for i in "${!_labels[@]}"; do
            printf "%-28s  %s  %-5s\n" "${_labels[$i]}" "$(status_of "${_pids[$i]}")" "${_gpus[$i]}"
            progress "${_labels[$i]}"
        done

        $all_done && break
        sleep 15
    done

    printf "%.0s─" {1..80}; echo
}

# ---------------------------------------------------------------------------
# Phase 1 — Diffusion level 0 + upsamplers 1–4
# ---------------------------------------------------------------------------

echo "[$(date '+%H:%M:%S')] Phase 1: launching level 0 + upsamplers 1–4…"

PID_D0=$(start dales_diff_0 $GPU_A \
    src/diffusion/train_diffusion.py \
    -config configs/train_dales_diffusion_0.yaml -dataset dales -level 0)

PID_U1=$(start dales_up_1 $GPU_A \
    src/diffusion/train_upsamplers.py \
    -config configs/train_dales_upsampler.yaml -dataset dales -level 1)

PID_U2=$(start dales_up_2 $GPU_B \
    src/diffusion/train_upsamplers.py \
    -config configs/train_dales_upsampler.yaml -dataset dales -level 2)

PID_U3=$(start dales_up_3 $GPU_B \
    src/diffusion/train_upsamplers.py \
    -config configs/train_dales_upsampler.yaml -dataset dales -level 3)

PID_U4=$(start dales_up_4 $GPU_B \
    src/diffusion/train_upsamplers.py \
    -config configs/train_dales_upsampler.yaml -dataset dales -level 4)

PHASE1_LABELS=("diffusion  level 0" "upsampler  level 1" "upsampler  level 2" "upsampler  level 3" "upsampler  level 4")
PHASE1_PIDS=($PID_D0 $PID_U1 $PID_U2 $PID_U3 $PID_U4)
PHASE1_GPUS=($GPU_A $GPU_B $GPU_B $GPU_B $GPU_B)

dashboard PHASE1_LABELS PHASE1_PIDS PHASE1_GPUS

# ---------------------------------------------------------------------------
# Phase 2 — Diffusion levels 1–4 (requires upsampler checkpoints)
# ---------------------------------------------------------------------------

echo "[$(date '+%H:%M:%S')] Phase 2: launching diffusion levels 1–4…"

PID_D1=$(start dales_diff_1 $GPU_A \
    src/diffusion/train_diffusion.py \
    -config configs/train_dales_diffusion_up.yaml -dataset dales -level 1)

PID_D2=$(start dales_diff_2 $GPU_A \
    src/diffusion/train_diffusion.py \
    -config configs/train_dales_diffusion_up.yaml -dataset dales -level 2)

PID_D3=$(start dales_diff_3 $GPU_B \
    src/diffusion/train_diffusion.py \
    -config configs/train_dales_diffusion_up.yaml -dataset dales -level 3)

PID_D4=$(start dales_diff_4 $GPU_B \
    src/diffusion/train_diffusion.py \
    -config configs/train_dales_diffusion_up.yaml -dataset dales -level 4)

PHASE2_LABELS=("diffusion  level 1" "diffusion  level 2" "diffusion  level 3" "diffusion  level 4")
PHASE2_PIDS=($PID_D1 $PID_D2 $PID_D3 $PID_D4)
PHASE2_GPUS=($GPU_A $GPU_A $GPU_B $GPU_B)

dashboard PHASE2_LABELS PHASE2_PIDS PHASE2_GPUS

echo "[$(date '+%H:%M:%S')] Training complete. Checkpoints in checkpoints/"
