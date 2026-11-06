"""Run the LoRA model on one image and save the structure as JSON and an annotated PNG.

  python -m src.training.infer_image --config configs/qwen3_vl_4b_lora.json \
      --adapter models/radu/exp_hd_epochs12/final --image diagram.png --output out/diagram

Writes out/diagram.png and out/diagram.json. With --data, a prepared JSONL holding this image,
the ground truth is drawn beside the prediction. Omit --adapter to run the base model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .evaluate import score_record, valid_box, valid_point
from .prepare import build_prompt, read_jsonl
from .visualize_predictions import salvage_elements

NODE_COLOR = (21, 101, 192)
ARROW_COLOR = (216, 27, 96)
HEADER_HEIGHT = 44


def draw_structure(image, nodes, arrows, title):
    """Return a copy of the image with elements drawn and a title band on top."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = image.size
    line = max(2, round(max(width, height) / 450))
    panel = Image.new("RGB", (width, height + HEADER_HEIGHT), "white")
    panel.paste(image.convert("RGB"), (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=max(14, round(max(width, height) / 70)))
    draw.text((8, 10), title, fill="black", font=ImageFont.load_default(size=20))
    for node in nodes:
        box = node.get("bbox") if isinstance(node, dict) else None
        if valid_box(box):
            x0, y0, x1, y1 = box[0], box[1] + HEADER_HEIGHT, box[2], box[3] + HEADER_HEIGHT
            draw.rectangle((x0, y0, x1, y1), outline=NODE_COLOR, width=line)
            draw.text((x0 + line, y0 + line), str(node.get("kind", "")), fill=NODE_COLOR, font=font, stroke_width=2, stroke_fill="white")
    head = line * 6
    for arrow in arrows:
        start, end = (arrow.get("start"), arrow.get("end")) if isinstance(arrow, dict) else (None, None)
        if valid_point(start) and valid_point(end):
            x0, y0, x1, y1 = start[0], start[1] + HEADER_HEIGHT, end[0], end[1] + HEADER_HEIGHT
            draw.line((x0, y0, x1, y1), fill=ARROW_COLOR, width=line)
            angle = math.atan2(y1 - y0, x1 - x0)
            wings = [(x1 - head * math.cos(angle + turn), y1 - head * math.sin(angle + turn)) for turn in (-0.45, 0.45)]
            draw.polygon([(x1, y1), *wings], fill=ARROW_COLOR)
    return panel


def side_by_side(panels):
    from PIL import Image

    gap = 12
    canvas = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1), max(p.height for p in panels)), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gap
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", help="LoRA adapter directory; omit to run the base model")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="Output path without extension")
    parser.add_argument("--data", help="Prepared JSONL containing this image, to draw ground truth and score")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()
    import torch
    from PIL import Image

    from .train import load_base_model

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    image_path = str(Path(args.image).resolve())
    row = None
    if args.data:
        row = next((item for item in read_jsonl(args.data) if str(Path(item["image"]).resolve()) == image_path), None)
        if row is None:
            raise ValueError(f"{args.image} is not in {args.data}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    prompt_text = row["prompt"] if row else build_prompt({"tasks": ["nodes", "arrows", "relations"], "width": width, "height": height})
    processor, model = load_base_model(config)
    if not config.get("load_in_4bit", False):
        model.to("cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)

    nodes, arrows, relations, truncated = salvage_elements(text)
    title = f"Prediction: {len(nodes)} nodes, {len(arrows)} arrows, {len(relations)} relations"
    if truncated:
        title += " (invalid JSON, partial)"
    result = {"image": image_path, "adapter": args.adapter, "generated_tokens": len(new_tokens), "valid_json": not truncated,
              "nodes": nodes, "arrows": arrows, "relations": relations}
    panels = [draw_structure(image, nodes, arrows, title)]
    if row:
        gold = json.loads(row["completion"])
        score = score_record(text, gold, width, height)
        result["score"] = score
        gold_title = f"Ground truth: {len(gold.get('nodes', []))} nodes, {len(gold.get('arrows', []))} arrows"
        panels.insert(0, draw_structure(image, gold.get("nodes", []), gold.get("arrows", []), gold_title))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    side_by_side(panels).save(output.with_suffix(".png"))
    output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("generated_tokens", "valid_json")} | ({"score": result["score"]} if row else {}), ensure_ascii=False))
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
