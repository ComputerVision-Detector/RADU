"""Compare a crop with the parent and a clean-train adapter bundle."""
import argparse
import gc
import json
from pathlib import Path

from src.training.ocr_clean_vision import load_arm,effective_config,verify_inputs
from src.training.ocr_vl_lora import infer
from src.training.ocr_finetune import save
from src.ocr_data.audit import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--image',required=True);p.add_argument('--output',required=True);p.add_argument('--config',default='configs/ocr_clean_vision.json');p.add_argument('--arm',choices=['selected','language','vision','parent'],default='selected');a=p.parse_args()
    import torch
    r=json.loads(Path(a.config).read_text());root=Path(r['output']);r=json.loads((root/'round-config.json').read_text());verify_inputs(r)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);image=Path(a.image).resolve();row=dict(page_id=image.stem,region_id='input-crop',image=str(image),text='');c=effective_config(r)
    arm=json.loads((root/'selection-lock.json').read_text())['chosen']['arm'] if a.arm=='selected' else a.arm
    model,proc,_,_=load_arm(r,'parent',False);before,_=infer(model,proc,[row],c,out/'before.json')
    if arm=='parent':after=before;active=list(model.active_adapters)
    else:
        del model;gc.collect();torch.cuda.empty_cache();model,proc,_,_=load_arm(r,arm,False,root/arm/'best');after,_=infer(model,proc,[row],c,out/'after.json');active=list(model.active_adapters)
    result=dict(image=str(image),image_sha256=sha256(image),arm=arm,before=before[0],after=after[0],active_adapters=active,parent_language_adapter=r['parent_adapter'],checkpoint=str(root/arm/'best') if arm!='parent' else r['parent_adapter'])
    save(out/'comparison.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
