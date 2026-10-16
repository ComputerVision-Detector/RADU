"""Qwen2.5-VL direct-structure baseline; requires a local model checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROMPT = '''Read this hand-drawn flowchart and return only valid JSON using this schema:
{"version":"1.0","image":{"width":0,"height":0},"elements":[{"id":"node-1","type":"Box|Text|Arrow","bbox":[x1,y1,x2,y2],"text":"","shape":"process|decision|terminator|unknown","start":[x,y],"end":[x,y]}],"relations":[{"id":"edge-1","type":"flows_to","source":"node-1","target":"node-2","direction":"right|left|up|down|unknown"}],"warnings":[]}
Use pixel coordinates relative to the input image. Each text, box, and arrow is a separate element. Include explicit arrow start and end points when visible. Do not infer missing connections. Put uncertain items in warnings.'''


def run(image_path: str | Path, model_path: str, max_new_tokens: int = 2048) -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError("Install transformers, torch, and qwen-vl-utils") from exc
    from PIL import Image
    image=Image.open(image_path).convert("RGB")
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,torch_dtype="auto",device_map="auto")
    processor=AutoProcessor.from_pretrained(model_path)
    messages=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":PROMPT}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    image_inputs,video_inputs=process_vision_info(messages)
    inputs=processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt").to(model.device)
    generated=model.generate(**inputs,max_new_tokens=max_new_tokens)
    output=processor.batch_decode([out[len(inp):] for inp,out in zip(inputs.input_ids,generated)],skip_special_tokens=True)[0]
    start=output.find("{"); end=output.rfind("}")
    if start<0 or end<start: raise ValueError(f"Model did not return a JSON object: {output[:500]}")
    result=json.loads(output[start:end+1])
    result.setdefault("image",{}).update({"width":image.width,"height":image.height})
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("image"); parser.add_argument("--model",default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output",required=True); args=parser.parse_args()
    result=run(args.image,args.model)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__": main()
