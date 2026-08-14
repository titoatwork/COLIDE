#!/usr/bin/env bash
# Wait until multirun parent exits, then start imbalance loss compare (GPU free).
set -euo pipefail
cd /home/titoisalive/colide
LOG=/tmp/post_multirun_queue.log
echo "[$(date -Is)] watcher start" >> "$LOG"
while pgrep -f 'scripts/run_baseline_multirun.py' >/dev/null 2>&1; do
  sleep 60
done
echo "[$(date -Is)] multirun finished; starting imbalance loss compare" >> "$LOG"
export PYTHONPATH=/home/titoisalive/colide
exec .venv/bin/python scripts/run_imbalance_loss_compare.py --epochs 5 --seed 42 \
  >> /tmp/imbalance_loss_compare.log 2>&1
