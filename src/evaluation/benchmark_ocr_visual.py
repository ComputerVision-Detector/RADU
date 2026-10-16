"""Sample a Hugging Face image dataset and render OCR outputs for visual review.

No labels are expected. This is a qualitative side-by-side benchmark, not an
accuracy/CER evaluation. Dataset streaming avoids downloading the full corpus.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.cv.ocr_engine import recognize, release_paddleocrv6


DEFAULT_DATASET = "hakuuung/HybridOCR_CustomNotes"
DEFAULT_ENGINES = ("paddleocrv6", "paddleocrvl16")


def load_random_records(dataset_id: str, split: str | None, count: int, seed: int) -> tuple[list[dict[str, Any]],str]:
    """Load a seeded random sample from a streaming HF dataset split."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Hugging Face dataset loading needs `pip install datasets`") from exc
    try:
        dataset = load_dataset(dataset_id, streaming=True)
        if hasattr(dataset,"keys"):
            available=list(dataset.keys())
            if not available: raise RuntimeError("Dataset has no splits")
            actual_split=split or ("train" if "train" in available else available[0])
            if actual_split not in dataset:
                raise RuntimeError(f"Split {actual_split!r} not found; available: {available}")
            stream=dataset[actual_split]
        else:
            actual_split=split or "default"
            stream=dataset
        # Streaming shuffle uses a bounded buffer and does not fetch the full dataset.
        sampled = list(stream.shuffle(seed=seed, buffer_size=max(100, count * 20)).take(count))
    except Exception as exc:
        raise RuntimeError(f"Could not read {dataset_id!r} split {split or 'auto'!r}: {exc}") from exc
    if not sampled:
        raise RuntimeError(f"Dataset {dataset_id!r} split {split!r} returned no records")
    return sampled,actual_split


def _image_from_value(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            import io
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            try: return Image.open(value["path"]).convert("RGB")
            except (OSError, TypeError): return None
    if isinstance(value, np.ndarray) and value.ndim in (2, 3):
        arr=value
        if arr.dtype != np.uint8:
            arr=np.clip(arr,0,255).astype(np.uint8)
        if arr.ndim==2: return Image.fromarray(arr).convert("RGB")
        if arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4): arr=np.moveaxis(arr,0,-1)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(value, str):
        try: return Image.open(value).convert("RGB")
        except OSError: return None
    return None


def extract_image(record: dict[str, Any]) -> tuple[str, Image.Image]:
    """Find an image feature in a generic dataset record."""
    candidates=[]
    for key,value in record.items():
        image=_image_from_value(value)
        if image is not None:
            score=0 if key.lower() in {"image","img","picture","scan"} else 1
            candidates.append((score,key,image))
    if not candidates:
        raise ValueError(f"No decodable image column found. Available fields: {list(record)}")
    _,key,image=min(candidates,key=lambda item:item[0])
    return key,image


def _font(size: int) -> ImageFont.ImageFont:
    candidates=("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    for path in candidates:
        if Path(path).exists():
            try: return ImageFont.truetype(path,size=size)
            except OSError: pass
    return ImageFont.load_default()


def _bbox_points(bbox: Any) -> list[tuple[int,int]]:
    try:
        values=np.asarray(bbox,dtype=float)
        if values.ndim==2 and values.shape[1]>=2:
            return [(int(x),int(y)) for x,y,*_ in values.tolist()]
        values=values.reshape(-1)
        if values.size==4:
            x1,y1,x2,y2=map(int,values.tolist())
            return [(x1,y1),(x2,y2)]
    except (TypeError,ValueError): pass
    return []


def annotate_result(image: Image.Image, engine: str, text: str, details: dict[str,Any],
                    latency: float | None, error: str | None = None,
                    inference_size: tuple[int,int] | None = None) -> Image.Image:
    """Create a review image with OCR text above the source and boxes overlaid."""
    base=image.convert("RGB")
    original_w,original_h=base.size
    scale=min(1.0,1200/max(original_w,original_h))
    if scale<1:
        base=base.resize((int(original_w*scale),int(original_h*scale)),Image.Resampling.LANCZOS)
    w,h=base.size
    inference_scale_x=original_w/(inference_size[0] if inference_size else original_w)
    inference_scale_y=original_h/(inference_size[1] if inference_size else original_h)
    title_font=_font(22); text_font=_font(17); margin=18; max_chars=max(28,(w-2*margin)//18)
    transcript=("ERROR: "+error) if error else (text.strip() or "(모델이 텍스트를 반환하지 않았습니다)")
    wrapped=[]
    for paragraph in transcript.splitlines() or [transcript]:
        wrapped.extend([paragraph[i:i+max_chars] for i in range(0,max(1,len(paragraph)),max_chars)])
    shown=wrapped[:12]
    if len(wrapped)>len(shown): shown.append(f"… ({len(wrapped)-len(shown)}줄 생략)")
    banner_h=62+len(shown)*25
    canvas=Image.new("RGB",(w,h+banner_h),"white")
    canvas.paste(base,(0,banner_h))
    draw=ImageDraw.Draw(canvas)
    label=engine.upper().replace("PADDLEOCRVL16","PaddleOCR-VL 1.6").replace("PADDLEOCRV6","PaddleOCR v6")
    timing=f" · {latency:.2f}s" if latency is not None else ""
    draw.text((margin,10),label+timing,font=title_font,fill="#12345a")
    y=42
    for line in shown:
        draw.text((margin,y),line,font=text_font,fill="#172033"); y+=25
    colors={"paddleocrv6":"#e11d48","paddleocrvl16":"#2563eb"}
    color=colors.get(engine,"#059669"); stroke=max(2,int(min(w,h)*.003))
    for line in details.get("lines",[]):
        points=_bbox_points(line.get("bbox"))
        if not points: continue
        points=[(int(x*inference_scale_x*scale),int(y*inference_scale_y*scale)+banner_h) for x,y in points]
        if len(points)>=3: draw.polygon(points,outline=color,width=stroke)
        elif len(points)==2:
            (x1,y1),(x2,y2)=points
            draw.rectangle((min(x1,x2),min(y1,y2),max(x1,x2),max(y1,y2)),outline=color,width=stroke)
    return canvas


def run_benchmark(dataset_id: str=DEFAULT_DATASET, split: str|None=None, count: int=3,
                  seed: int=42, output_dir: str|Path="data/benchmark/hf_visual",
                  engines: tuple[str,...]=DEFAULT_ENGINES) -> dict[str,Any]:
    records,actual_split=load_random_records(dataset_id,split,count,seed)
    print(f"데이터셋 로드 완료: {dataset_id} / {actual_split} ({len(records)}장)",flush=True)
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    report={"dataset":dataset_id,"split":actual_split,"seed":seed,"requested_samples":count,
            "sample_count":len(records),"labels_available":False,"engines":list(engines),"samples":[]}
    prepared=[]
    for index,record in enumerate(records,1):
        try: image_key,image=extract_image(record)
        except Exception as exc:
            report["samples"].append({"sample_index":index,"error":str(exc)}); continue
        source_id=str(record.get("id",record.get("file_name",record.get("filename",f"sample-{index:03d}"))))
        safe_id=re.sub(r"[^A-Za-z0-9_.-]+","_",Path(source_id).name)[:60] or f"sample-{index:03d}"
        sample={"sample_index":index,"source_id":source_id,"image_column":image_key,
                "image_width":image.width,"image_height":image.height,"results":[]}
        inference_image=image.copy()
        inference_image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
        inference_size=inference_image.size
        image_array=np.asarray(inference_image)[:,:,::-1].copy()  # Paddle adapters expect OpenCV BGR.
        prepared.append({"index":index,"safe_id":safe_id,"sample":sample,"image":image,
                         "image_array":image_array,"inference_size":inference_size,
                         "panels":[("원본",image.convert("RGB"))]})

    # Paddle keeps its CUDA context alive after a model is closed. Run every
    # isolated VL worker before initializing PP-OCRv6 in this parent process.
    engine_order=tuple(sorted(engines,key=lambda name: 0 if name in {"paddleocrvl16","paddleocrvl1.6","ppocrvl16"} else 1))
    for engine in engine_order:
        for item in prepared:
            index=item["index"]; safe_id=item["safe_id"]; sample=item["sample"]
            started=time.perf_counter()
            try:
                print(f"[{index}/{len(records)}] {engine}: 추론 시작 ({sample['image_width']}x{sample['image_height']})",flush=True)
                faulthandler.dump_traceback_later(30,repeat=True)
                text,details=recognize(item["image_array"],engine)
                faulthandler.cancel_dump_traceback_later()
                latency=time.perf_counter()-started
                annotated=annotate_result(item["image"],engine,text,details,latency,
                                          inference_size=item["inference_size"])
                error=None
                result={"engine":engine,"latency_seconds":round(latency,3),"text":text,
                        "line_count":len(details.get("lines",[])),"error":None}
            except Exception as exc:
                faulthandler.cancel_dump_traceback_later()
                latency=time.perf_counter()-started
                text=""; details={}; error=f"{type(exc).__name__}: {exc}"
                print(f"[{index}/{len(records)}] {engine}: 오류: {error}",file=sys.stderr,flush=True)
                annotated=annotate_result(item["image"],engine,text,details,latency,error)
                result={"engine":engine,"latency_seconds":round(latency,3),"text":"",
                        "line_count":0,"error":error}
            finally:
                # Do not keep PP-OCRv6's CUDA context/model resident while the
                # isolated PaddleOCR-VL process creates its own GPU context.
                if engine == "paddleocrv6":
                    release_paddleocrv6()
            result_path=output/f"{index:02d}_{safe_id}_{engine}.png"
            annotated.save(result_path)
            result["visualization"]=str(result_path)
            sample["results"].append(result)
            item["panels"].append((engine,annotated))
    for item in prepared:
        index=item["index"]; safe_id=item["safe_id"]; sample=item["sample"]
        sample_path=output/f"{index:02d}_{safe_id}_comparison.png"
        _save_contact_sheet(item["panels"],sample_path)
        sample["comparison_image"]=str(sample_path)
        report["samples"].append(sample)
    report_path=output/"benchmark_report.json"
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


def _save_contact_sheet(panels: list[tuple[str,Image.Image]],path:Path) -> None:
    width=1000; gap=14; title_h=42
    resized=[]
    for title,image in panels:
        factor=min(1.0,(width-2*gap)/image.width)
        view=image.resize((int(image.width*factor),int(image.height*factor)),Image.Resampling.LANCZOS)
        resized.append((title,view))
    height=gap+sum(title_h+im.height+gap for _,im in resized)
    sheet=Image.new("RGB",(width,height),"#e9eef5"); draw=ImageDraw.Draw(sheet); font=_font(20); y=gap
    for title,image in resized:
        draw.rounded_rectangle((gap,y,width-gap,y+title_h-4),radius=7,fill="#dce7f6")
        draw.text((gap+10,y+7),title,font=font,fill="#153150"); y+=title_h
        sheet.paste(image,(gap,y)); y+=image.height+gap
    path.parent.mkdir(parents=True,exist_ok=True); sheet.save(path)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",default=DEFAULT_DATASET)
    parser.add_argument("--split",default=None,help="Dataset split (auto: train if present, otherwise the first split)")
    parser.add_argument("--samples",type=int,default=3)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",default="gpu:0",help="Paddle 장치 (예: gpu:0 또는 cpu)")
    parser.add_argument("--output-dir",default="data/benchmark/hf_visual")
    parser.add_argument("--engines",nargs="+",default=list(DEFAULT_ENGINES),choices=("paddleocrv6","paddleocrvl16","clova"))
    args=parser.parse_args()
    if args.samples<1: parser.error("--samples must be at least 1")
    import os
    os.environ["RADU_PADDLE_DEVICE"]=args.device
    try:
        report=run_benchmark(args.dataset,args.split,args.samples,args.seed,args.output_dir,tuple(args.engines))
    except RuntimeError as exc:
        print(str(exc),file=sys.stderr); raise SystemExit(2)
    print(json.dumps({"dataset":report["dataset"],"sample_count":report["sample_count"],
                      "report":str(Path(args.output_dir)/"benchmark_report.json"),
                      "comparisons":[sample.get("comparison_image") for sample in report["samples"]]},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
