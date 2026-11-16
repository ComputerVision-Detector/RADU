"""Qwen3-VL LoRA structure inference for one image. Runs in the training environment.

  .cache/venvs/train/bin/python -m src.cv.structure_qwen_worker IMAGE OUTPUT --groups diagram,formula --device cuda:0

Each group is one prompt with the adapter fixed in configs/structure_qwen_lora.lock.json.
The raw model text is written as-is; parsing happens in src.cv.structure_qwen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("output")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    try:
        import torch
        from peft import PeftModel
        from PIL import Image

        from src.training.prepare import build_prompt
        from src.training.train import load_base_model

        binding = json.loads((ROOT / "configs/structure_qwen_lora.lock.json").read_text(encoding="utf-8"))
        config = json.loads((ROOT / binding["config"]).read_text(encoding="utf-8"))
        groups = [name for name in args.groups.split(",") if name]
        unknown = set(groups) - set(binding["groups"])
        if unknown or not groups:
            raise ValueError(f"unknown structure groups: {sorted(unknown)}")
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        width, height = image.size
        started = time.perf_counter()
        processor, model = load_base_model(config)
        model.to(args.device)
        adapters = {}
        for name in groups:
            path = binding["groups"][name]["adapter"]
            if path in adapters:
                continue
            adapters[path] = f"a{len(adapters)}"
            if isinstance(model, PeftModel):
                model.load_adapter(str(ROOT / path), adapter_name=adapters[path])
            else:
                model = PeftModel.from_pretrained(model, str(ROOT / path), adapter_name=adapters[path])
        model.eval()
        load_seconds = time.perf_counter() - started
        results = {}
        for name in groups:
            group = binding["groups"][name]
            model.set_adapter(adapters[group["adapter"]])
            prompt_text = build_prompt({"tasks": group["tasks"], "width": width, "height": height})
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(args.device)
            tick = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(**inputs, max_new_tokens=binding["max_new_tokens"], do_sample=False)
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            results[name] = {
                "text": processor.tokenizer.decode(new_tokens, skip_special_tokens=True),
                "tasks": group["tasks"], "adapter": group["adapter"],
                "generated_tokens": len(new_tokens),
                "truncated": len(new_tokens) >= binding["max_new_tokens"],
                "seconds": time.perf_counter() - tick,
            }
        payload = {"width": width, "height": height, "groups": results, "load_seconds": load_seconds,
                   "model": config["model_id"], "model_revision": config.get("model_revision"),
                   "device": args.device, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as error:
        output.write_text(json.dumps({"error": f"{type(error).__name__}: {error}"}, ensure_ascii=False), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
