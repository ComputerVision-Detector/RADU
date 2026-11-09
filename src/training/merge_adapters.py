"""Combine LoRA adapters of the same shape into one adapter whose weight change is their weighted sum.

The result stacks the adapters along the rank dimension, so it is exact, not an approximation:
merged B @ A == sum(weight_i * scaling_i * B_i @ A_i). Source adapters are only read.

  python -m src.training.merge_adapters --adapter models/radu/run/checkpoint-6000 0.5 \
      --adapter models/radu/run/checkpoint-18000 0.5 --output models/radu/merged/w55
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

COPIED = ("chat_template.jinja", "processor_config.json", "tokenizer.json", "tokenizer_config.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", nargs=2, action="append", required=True, metavar=("DIR", "WEIGHT"))
    parser.add_argument("--output", required=True, help="New directory")
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file, save_file

    output = Path(args.output)
    if output.exists():
        raise SystemExit("output exists; use a new directory")
    sources = [(Path(path), float(weight)) for path, weight in args.adapter]
    configs = [json.loads((path / "adapter_config.json").read_text(encoding="utf-8")) for path, _ in sources]
    for config in configs:
        if config["rank_pattern"] or config["alpha_pattern"] or config.get("use_rslora"):
            raise SystemExit("per-module ranks, alphas and rsLoRA are not supported")
        if config["target_modules"] != configs[0]["target_modules"]:
            raise SystemExit("adapters target different modules")
    states = [load_file(str(path / "adapter_model.safetensors")) for path, _ in sources]
    if any(set(state) != set(states[0]) for state in states):
        raise SystemExit("adapters have different tensors")
    merged = {}
    for key in states[0]:
        if ".lora_A." in key:
            merged[key] = torch.cat([state[key] for state in states], dim=0).contiguous()
        elif ".lora_B." in key:
            # The merged adapter uses scaling 1, so each part carries its own weight and alpha/r.
            merged[key] = torch.cat([state[key] * (weight * config["lora_alpha"] / config["r"])
                                     for state, (_, weight), config in zip(states, sources, configs)], dim=1).contiguous()
        else:
            raise SystemExit(f"unexpected tensor {key}")
    rank = sum(config["r"] for config in configs)
    output.mkdir(parents=True)
    save_file(merged, str(output / "adapter_model.safetensors"))
    (output / "adapter_config.json").write_text(json.dumps(dict(configs[0], r=rank, lora_alpha=rank, lora_dropout=0.0), indent=2), encoding="utf-8")
    for name in COPIED:
        if (sources[0][0] / name).exists():
            shutil.copy(sources[0][0] / name, output / name)
    (output / "merge.json").write_text(json.dumps({"adapters": [[str(path), weight] for path, weight in sources], "rank": rank}, indent=2), encoding="utf-8")
    print(f"{len(merged)} tensors, rank {rank} -> {output}")


if __name__ == "__main__":
    main()
