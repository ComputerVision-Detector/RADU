#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OCR_PYTHON="$PROJECT_ROOT/.venv/bin/python"
OCR_CONFIG="$PROJECT_ROOT/src/cv/ocr/custom_rec_korean.yml"
export OMP_NUM_THREADS=1
export FLAGS_allocator_strategy=auto_growth
"$OCR_PYTHON" "$PROJECT_ROOT/scripts/check_ocr_training.py" --config "$OCR_CONFIG" --device gpu:0 --batch-size 64
if [[ "${1:-}" == "--check-only" ]]; then
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo 'Usage: bash scripts/train_ocr_words.sh [--check-only]' >&2
    exit 2
fi
cd "$PROJECT_ROOT/src/cv/ocr/PaddleOCR"
"$OCR_PYTHON" tools/train.py -c "$OCR_CONFIG"
"$OCR_PYTHON" "$PROJECT_ROOT/scripts/export_ocr_model.py" --config "$OCR_CONFIG"
"$OCR_PYTHON" "$PROJECT_ROOT/scripts/evaluate_ocr_checkpoint.py" --config "$OCR_CONFIG" --device gpu:0
