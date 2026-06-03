#!/usr/bin/env bash
# status.py — quick GPU + container status summary
#
# Usage:
#   ./scripts/status.py
#   ./scripts/status.py --container rocm-llama
#
# Shows:
#   - Container status (docker compose ps)
#   - ROCm GPU info (rocminfo)
#   - ROCm SMI (rocm-smi)
#   - Container GPU visibility check

set -euo pipefail

CONTAINER="${1:-rocm-llama}"

echo "=== Docker containers ==="
docker compose ps 2>/dev/null || echo "docker compose not found or not in project dir"
echo

echo "=== ROCm GPU info (host) ==="
if command -v rocminfo &>/dev/null; then
    rocminfo 2>/dev/null | grep -E "Name:|gfx|Profile:" || echo "rocminfo returned no GPU data"
else
    echo "rocminfo not available on host"
fi
echo

echo "=== ROCm SMI (host) ==="
if command -v rocm-smi &>/dev/null; then
    rocm-smi 2>/dev/null | head -30 || echo "rocm-smi returned no data"
else
    echo "rocm-smi not available on host"
fi
echo

echo "=== Container GPU check ==="
if docker compose ps 2>/dev/null | grep -q "$CONTAINER"; then
    docker compose exec "$CONTAINER" rocminfo 2>/dev/null | grep -E "Name:|gfx" || \
        echo "container running but rocminfo failed (GPU passthrough issue?)"
else
    echo "Container '$CONTAINER' not running"
fi
echo

echo "=== HIP_VISIBLE_DEVICES ==="
if docker compose ps 2>/dev/null | grep -q "$CONTAINER"; then
    docker compose exec "$CONTAINER" env | grep HIP || echo "HIP_VISIBLE_DEVICES not set"
else
    echo "Container '$CONTAINER' not running"
fi
