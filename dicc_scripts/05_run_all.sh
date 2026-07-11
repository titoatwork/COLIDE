#!/usr/bin/env bash
# =============================================================================
# Back-compat entrypoint. Prefer submit_session.sh (absolute logs, --chdir,
# session manifest, Nsight opt-in). This wrapper only submits the default
# V100+A100 core session.
# =============================================================================
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/submit_session.sh" "$@"
