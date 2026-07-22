#!/usr/bin/env bash
# Thermal/resource guard for WP5b neural baselines (run_neural_baselines.py).
# Soft warn >=85C, hard pause >=90C (stop train, keep completed row JSONs).
set -u
LOG=/home/titoisalive/colide/logs/thermal_guard_neural_baselines.log
PROJ=/home/titoisalive/colide
SOFT=85
HARD=90
INTERVAL=90
TAG=baselines_neural
PROC_PATTERN='run_neural_baselines.py'

echo "thermal_guard_neural_baselines start $(date -u -Iseconds) soft=${SOFT} hard=${HARD} tag=${TAG}" | tee -a "$LOG"

while true; do
  ts=$(date -u +%H:%M:%S)
  read -r TEMP UTIL PWR MEMU <<< "$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | tr ',' ' ')"
  TEMP=${TEMP%.*}
  UTIL=${UTIL%.*}
  LOAD=$(cut -d' ' -f1 /proc/loadavg)
  MEMAVAIL=$(free -m | awk '/Mem:/{print $7}')
  SWAPUSED=$(free -m | awk '/Swap:/{print $3}')
  ROWS=$(ls "$PROJ"/benchmarks/results/${TAG}/G*_seed*.json 2>/dev/null | wc -l)
  TRAIN=$(pgrep -c -f "$PROC_PATTERN" 2>/dev/null || echo 0)

  line="$ts temp=${TEMP}C util=${UTIL}% pwr=${PWR}W vram=${MEMU}MiB load=${LOAD} mem_avail=${MEMAVAIL}M swap=${SWAPUSED}M rows=${ROWS}/7 train_procs=${TRAIN}"
  echo "$line" | tee -a "$LOG"

  if [ "${TRAIN:-0}" -eq 0 ] && [ -f "$PROJ/benchmarks/results/${TAG}/summary.json" ]; then
    echo "$ts JOB_COMPLETE summary present rows=$ROWS" | tee -a "$LOG"
    exit 0
  fi
  if [ "${TRAIN:-0}" -eq 0 ]; then
    echo "$ts JOB_IDLE no $PROC_PATTERN processes rows=$ROWS" | tee -a "$LOG"
    if [ ! -f "$PROJ/benchmarks/results/${TAG}/summary.json" ]; then
      echo "$ts WARN job died without summary rows=$ROWS" | tee -a "$LOG"
    fi
    exit 0
  fi

  if [ -n "${TEMP:-}" ] && [ "$TEMP" -ge "$HARD" ] 2>/dev/null; then
    echo "$ts HARD_TRIP temp=${TEMP}>=${HARD} pausing neural baselines" | tee -a "$LOG"
    pkill -TERM -f "$PROC_PATTERN" 2>/dev/null || true
    sleep 2
    pkill -KILL -f "$PROC_PATTERN" 2>/dev/null || true
    echo "$ts PAUSED completed_rows_kept=$ROWS (re-run remaining later with --skip-existing)" | tee -a "$LOG"
    echo "PAUSED_THERMAL temp=$TEMP rows=$ROWS $(date -u -Iseconds)" > "$PROJ/logs/THERMAL_PAUSE.flag"
    exit 2
  fi

  if [ -n "${TEMP:-}" ] && [ "$TEMP" -ge "$SOFT" ] 2>/dev/null; then
    echo "$ts SOFT_WARN temp=${TEMP}>=${SOFT} continuing" | tee -a "$LOG"
  fi

  if [ -n "${MEMAVAIL:-}" ] && [ "$MEMAVAIL" -lt 150 ] 2>/dev/null; then
    echo "$ts HARD_MEM_TRIP avail=${MEMAVAIL}M pausing" | tee -a "$LOG"
    pkill -TERM -f "$PROC_PATTERN" 2>/dev/null || true
    echo "PAUSED_MEM avail=$MEMAVAIL $(date -u -Iseconds)" > "$PROJ/logs/THERMAL_PAUSE.flag"
    exit 3
  fi

  sleep "$INTERVAL"
done
