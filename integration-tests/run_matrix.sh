#!/usr/bin/env bash
# Run the integration debug matrix — manually, not in CI.
#
# Usage
# -----
#   ./run_matrix.sh                              # all available infras
#   ./run_matrix.sh local                        # one infra
#   ./run_matrix.sh terminalbench toolkit        # one cube × one infra
#   ./run_matrix.sh terminalbench toolkit overfull-hbox   # one task
#
# The script always cd's into the integration-tests directory first so you
# can call it from the repo root:
#   bash integration-tests/run_matrix.sh terminalbench local

set -euo pipefail
cd "$(dirname "$0")"

CUBE="${1:-}"
INFRA="${2:-}"
TASK="${3:-}"

# Build the -k filter expression
K=""
if [[ -n "$CUBE" && -n "$INFRA" && -n "$TASK" ]]; then
    K="${CUBE} and ${TASK} and ${INFRA}"
elif [[ -n "$CUBE" && -n "$INFRA" ]]; then
    K="${CUBE} and ${INFRA}"
elif [[ -n "$CUBE" ]]; then
    K="${CUBE}"
elif [[ -n "$INFRA" ]]; then
    K="${INFRA}"
fi

# Determine which uv dependency groups to sync based on requested infra
# (or all local+cloud groups if no infra specified).
case "${INFRA:-all}" in
    local)   GROUPS="--group local" ;;
    daytona) GROUPS="--group daytona" ;;
    toolkit) GROUPS="--group toolkit" ;;
    modal)   GROUPS="--group modal" ;;
    *)       GROUPS="--group local --group daytona --group toolkit --group modal" ;;
esac

echo "==> Syncing deps ($GROUPS --group dev)…"
uv sync $GROUPS --group dev -q

PYTEST_ARGS=(
    test_debug_matrix.py
    -v
    --log-cli-level=INFO
    --tb=short
    # -p no:timeout   # uncomment to disable the timeout backstop temporarily
)
[[ -n "$K" ]] && PYTEST_ARGS+=(-k "$K")

echo "==> Running: uv run $GROUPS --group dev pytest ${PYTEST_ARGS[*]}"
echo ""
uv run $GROUPS --group dev pytest "${PYTEST_ARGS[@]}"
