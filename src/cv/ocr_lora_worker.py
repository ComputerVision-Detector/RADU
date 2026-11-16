"""Separate existing Paddle detector and Torch/PEFT fixed LoRA workers."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def make_crops(image_path: Path, polygons, directory: Path) -> list:
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    regions=[]
    with Image.open(image_path) as image:
        width,height=image.size
        boxes=[]
        for polygon in polygons:
            values=[float(v) for point in polygon for v in point]
            if len(values)<6 or len(values)%2 or not all(math.isfinite(v) for v in values):
                raise ValueError('검출기가 유효하지 않은 좌표를 반환했습니다.')
            xs,ys=values[::2],values[1::2]
            box=[max(0,math.floor(min(xs))),max(0,math.floor(min(ys))),min(width,math.ceil(max(xs))),min(height,math.ceil(max(ys)))]
            if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('검출 박스의 크기가 유효하지 않습니다.')
            boxes.append(box)
        boxes.sort(key=lambda b:(b[1],b[0],b[3],b[2]))
        for i,box in enumerate(boxes):
            path=directory/f'{i:04d}.png';image.crop(box).save(path)
            regions.append({'region_id':f'det-{i:04d}','bbox':box,'image':str(path),'crop_sha256':sha(path)})
    return regions


def detect(image,output,device):
    from PIL import Image
    from src.cv.ocr_lora import ROOT
    os.environ['PADDLE_PDX_CACHE_HOME']=str(ROOT/'.cache/paddlex')
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    from paddlex import create_model
    from src.cv.ocr_engine import _result_payload
    tick=time.perf_counter()
    model=create_model(model_name='PP-OCRv6_medium_det',model_dir=str(ROOT/'models/detection/PP-OCRv6_medium_det'),device=device)
    payload=_result_payload(list(model.predict(str(image)))[0])
    regions=make_crops(image,payload['dt_polys'],output.parent/'crops')
    with Image.open(image) as source:width,height=source.size
    return {'regions':regions,'width':width,'height':height,'image_sha256':sha(image),'seconds':time.perf_counter()-tick}


def recognize(image,output,device,detections):
    from src.cv.ocr_lora import ROOT, fixed_binding, gpu_device
    binding=fixed_binding();data=json.loads(detections.read_text())
    if sha(image)!=data['image_sha256']:raise ValueError('검출·인식 입력 이미지가 다릅니다.')
    for row in data['regions']:
        if sha(row['image'])!=row['crop_sha256']:raise ValueError('검출 crop이 변경됐습니다.')
    import numpy as np
    import torch
    from peft import PeftModel
    from src.training.ocr_vl_lora import load, encode, generation_limit
    config=dict(binding['inference'],model=str(ROOT/binding['model']),dataset=str(output.parent),device=gpu_device(device))
    tick=time.perf_counter();base,processor,loading=load(config)
    model=PeftModel.from_pretrained(base,ROOT/binding['adapter'],is_trainable=False)
    model.eval();processor.tokenizer.padding_side='left';lines=[];limits=[]
    torch.cuda.reset_peak_memory_stats(config['device'])
    for row in data['regions']:
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        batch=encode(processor,[dict(row,text='')],config,False).to(config['device'])
        with torch.inference_mode():
            tokens=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=128,use_cache=True)
        generated=tokens[:,batch['input_ids'].shape[1]:]
        text=processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        if generation_limit(generated[0],128,processor.tokenizer.eos_token_id):limits.append(row['region_id'])
        lines.append({'id':row['region_id'],'text':text,'bbox':row['bbox'],'crop_sha256':row['crop_sha256'],'confidence':None})
    return {'lines':lines,'metadata':{'device':config['device'],'recognizer_called':True,
        'config':binding['inference'],'binding_sha256':sha(ROOT/'configs/ocr_lora_backend.lock.json'),
        'generation_limit_regions':limits,'recognizer_seconds':time.perf_counter()-tick,
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(config['device']),'strict_loading':loading}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['detect','recognize'])
    parser.add_argument('image',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--device',required=True);parser.add_argument('--detections',type=Path)
    args=parser.parse_args()
    try:
        if args.stage=='detect':result=detect(args.image,args.output,args.device)
        else:result=recognize(args.image,args.output,args.device,args.detections)
        args.output.write_text(json.dumps(result,ensure_ascii=False));return 0
    except Exception as error:
        args.output.write_text(json.dumps({'error':f'{type(error).__name__}: {error}'},ensure_ascii=False));return 1


if __name__=='__main__':raise SystemExit(main())
