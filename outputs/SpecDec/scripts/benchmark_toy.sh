#!/usr/bin/env bash
set -euo pipefail

python3 -m specdec toy-benchmark \
  --max-new-tokens "${MAX_NEW_TOKENS:-20000}" \
  --depths 1 2 4 8 \
  --repeats "${REPEATS:-3}"
