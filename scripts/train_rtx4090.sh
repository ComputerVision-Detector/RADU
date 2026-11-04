#!/usr/bin/env bash
# RADU LoRA training on the RTX 4090 workstation (Linux). Run from the repository root.
#   scripts/train_rtx4090.sh setup      create the environment and fetch the pinned model
#   scripts/train_rtx4090.sh prepare    validate data and build train/validation/test
#   scripts/train_rtx4090.sh preflight  tokenize every sample and check LoRA targets
#   scripts/train_rtx4090.sh smoke      one real optimizer step into a separate directory
#   scripts/train_rtx4090.sh train      full training
# data/raw is not in Git. Copy data/raw/radu/{fc_a,hdbpmn} from the prepared machine first,
# including normalized.jsonl. Run the stages in order; smoke must pass before train.
set -euo pipefail

CONFIG="configs/qwen3_vl_4b_lora.json"
VENV=".cache/venvs/train"
PYTHON="$VENV/bin/python"
export HF_HOME="$PWD/.cache/huggingface"

case "${1:-}" in
  setup)
    python3.11 -m venv "$VENV"
    "$PYTHON" -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements-training.lock.txt
    "$PYTHON" -c "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))"
    "$PYTHON" -c "import json; from huggingface_hub import snapshot_download; c = json.load(open('$CONFIG')); print(snapshot_download(c['model_id'], revision=c['model_revision']))"
    ;;
  prepare)
    # Paths inside prepared files are absolute, so they must be built on this machine.
    # --allow-partial: remove it once all six sources have normalized.jsonl.
    "$PYTHON" -m src.training.prepare --output-dir data/processed/radu --allow-partial
    ;;
  preflight)
    HF_HUB_OFFLINE=1 "$PYTHON" -m src.training.train --config "$CONFIG" --preflight-only
    ;;
  smoke)
    HF_HUB_OFFLINE=1 "$PYTHON" -m src.training.train --config "$CONFIG" --max-steps 1 --output-dir "models/radu/smoke-$(date +%Y%m%d-%H%M%S)"
    ;;
  train)
    HF_HUB_OFFLINE=1 "$PYTHON" -m src.training.train --config "$CONFIG"
    ;;
  *)
    sed -n '2,9p' "$0"
    exit 2
    ;;
esac
