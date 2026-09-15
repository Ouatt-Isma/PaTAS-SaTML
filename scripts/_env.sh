# Sourced by every run script. Runs from the artifact root so that the
# relative paths the experiment scripts use (results/, data/) resolve.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-}"          # cuda / cpu for the convolutional scripts; auto when empty
DEV_FLAG=""; [ -n "$DEVICE" ] && DEV_FLAG="--device $DEVICE"
echo "[run] root $ROOT  python $($PY --version 2>&1)"
