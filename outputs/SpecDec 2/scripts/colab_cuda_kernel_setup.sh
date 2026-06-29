#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
from specdec.cuda_extension import build_acceptance_rejection_so

path = build_acceptance_rejection_so(arch="sm_75", force=True)
print(path)
PY

python scripts/benchmark_acceptance_kernel.py \
  --arch sm_75 \
  --depths 1 2 \
  --iters 2000 \
  --warmup 200
