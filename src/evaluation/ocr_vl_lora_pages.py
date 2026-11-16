"""Paired VL/LoRA inference on the previous experiment's frozen detector crops.

Scope deliberately remains the prior CTC-screened GT (199 test references), so
the exact cached detection geometry/exclusion policy can be reused unchanged.
This scope differs from the VL GT-crop set (235 references).
"""
import argparse
import copy
import json
from pathlib import Path

from src.training.ocr_vl_lora import load,infer
from src.training.ocr_finetune import save
from src.evaluation.ocr_finetune_page import pages_for
from src.evaluation.ocr_v2 import evaluate
from src.ocr_data.audit import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/ocr_v2_vl_lora.json');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());root=Path(c['output']);out=root/'pages';out.mkdir(parents=True,exist_ok=False)
    old=json.loads(Path('configs/ocr_v2_korean_ctc.json').read_text())
    from peft import PeftModel
    model,proc,info=load(c)
    for name in ['before','after']:
        if name=='after':model=PeftModel.from_pretrained(model,root/'train/best-adapter')
        for split in ['validation','test']:
            cached=Path(f'data/experiments/ocr_v2_korean/det-{split}/detections.json');det=json.loads(cached.read_text())
            if det['dataset_manifest_sha256']!=sha256(Path(c['dataset'])/'manifest.json'):raise ValueError('frozen detector dataset drift')
            rows=[r for page in det['pages'] for r in page['regions']]
            for r in rows:
                if sha256(r['image'])!=r['crop_sha256']:raise ValueError('frozen detector crop drift')
            texts,rt=infer(model,proc,rows,c,out/f'{name}.{split}.recognition.json')
            keyed={(r['page_id'],r['region_id']):t for r,t in zip(rows,texts)}
            pred=dict(mode='page',pages=copy.deepcopy(det['pages']))
            for page in pred['pages']:
                for r in page['regions']:r['text']=keyed[r['page_id'],r['region_id']]
            metric=evaluate(pages_for(old,split),pred,.5)
            metric.update(runtime=rt,detector_sha256=sha256(cached),scope='prior CTC-screened full-page text scope; same fixed detector boxes; 199 test references',test_status=c['test_status'])
            save(out/f'{name}.{split}.predictions.json',pred);save(out/f'{name}.{split}.metrics.json',metric)
            print(json.dumps(dict(model=name,split=split,end_to_end=metric['end_to_end'],detection=metric['detection']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
