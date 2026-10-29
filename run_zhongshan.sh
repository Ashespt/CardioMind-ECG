#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
mkdir -p outputs/zs
if [[ ! -f models/C1/model.safetensors ]]; then
  echo "Place the C1 model.safetensors checkpoint in models/C1/ (see README.md)" >&2
  exit 1
fi
"$PYTHON_BIN" prepare_manifest.py
export TOKENIZERS_PARALLELISM=false
"$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC_PER_NODE" --master_port="${MASTER_PORT:-29611}" \
  predict_c1.py --checkpoint models/C1 --thresholds config/youden_thresholds.json \
  --subset zs --manifest data/manifest.jsonl --pulse_dir data/tools/PULSE \
  --featuredb data/tools/measurement.jsonl --clear_cls data/tools/clear_predictions.jsonl \
  --include_clear_cls --output_dir outputs/zs --batch_size 1 --num_workers 2 \
  --min_pixels 200704 --max_pixels 501760 2>&1 | tee outputs/inference.log
"$PYTHON_BIN" evaluate_zhongshan.py
