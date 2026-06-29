#!/usr/bin/env bash
set -euo pipefail

python3 -m specdec hf-benchmark \
  --draft-model "${DRAFT_MODEL:-gpt2}" \
  --target-model "${TARGET_MODEL:-gpt2-medium}" \
  --prompt "${PROMPT:-Speculative decoding is}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-64}" \
  --depths 1 2 4 6 8 \
  --repeats "${REPEATS:-1}" \
  --device cpu
