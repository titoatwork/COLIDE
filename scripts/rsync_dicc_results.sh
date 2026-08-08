#!/usr/bin/env bash
# Pull UM DICC campaign artifacts to the laptop.
# Requires: VPN + working SSH (Host dicc or ibteshamulhaque@login01.dicc.um.edu.my)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/benchmarks/results/dicc"
REMOTE_HOST="${DICC_SSH_HOST:-dicc}"
REMOTE_PATH="${DICC_REMOTE_PATH:-/home/user/ibteshamulhaque/colide/benchmarks/results/dicc/}"

mkdir -p "${DEST}"
echo "Syncing ${REMOTE_HOST}:${REMOTE_PATH} -> ${DEST}"
rsync -avz --progress \
  -e "ssh -o BatchMode=yes -o ConnectTimeout=20" \
  "${REMOTE_HOST}:${REMOTE_PATH}" \
  "${DEST}/"

echo "=== SUCCESS markers ==="
find "${DEST}" -name SUCCESS | sort
echo "=== done ==="
du -sh "${DEST}"
