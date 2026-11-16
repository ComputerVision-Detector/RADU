"""Infer a handwriting region crop with unchanged VL 1.6 and its saved LoRA."""
import argparse
import json
from pathlib import Path

from src.training.ocr_vl_lora import load,infer
from src.training.ocr_finetune import save
from src.ocr_data.audit import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='configs/ocr_v2_vl_lora.json');p.add_argument('--adapter');a=p.parse_args()
    from peft import PeftModel
    c=json.loads(Path(a.config).read_text());image=Path(a.image).resolve();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    adapter=Path(a.adapter) if a.adapter else Path(c['output'])/'train/best-adapter'
    row=dict(page_id=image.stem,region_id='input-crop',image=str(image),text='')
    model,proc,info=load(c);before,_=infer(model,proc,[row],c,out/'before.json')
    model=PeftModel.from_pretrained(model,adapter);after,_=infer(model,proc,[row],c,out/'after.json')
    result=dict(image=str(image),image_sha256=sha256(image),before=before[0],after=after[0],adapter_sha256=sha256(adapter/'adapter_model.safetensors'),config=c)
    save(out/'comparison.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
