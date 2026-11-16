"""Optional operational VL baseline, validation only, bounded isolated workers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from src.evaluation.ocr_finetune_page import pages_for,mask_policy
from src.evaluation.ocr_v2 import evaluate
from src.ocr_data.audit import sha256
from src.training.ocr_finetune import save


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--timeout',type=int,default=90);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    pages=pages_for(config,'validation');predictions=[]
    from src.evaluation.text_box_candidates import _xyxy
    for page in pages:
        if not page['eligible']:continue
        rawpath=out/f"{page['id']}.json";log=out/f"{page['id']}.log";row=dict(id=page['id'],regions=[]);start=time.perf_counter()
        try:
            if sha256(page['image'])!=page['source_sha256']:raise ValueError('original hash mismatch')
            with log.open('w') as handle:
                subprocess.run([sys.executable,'-m','src.evaluation.paddleocr_vl_worker',page['image'],str(rawpath),'--device',config['device']],stdout=handle,stderr=subprocess.STDOUT,timeout=a.timeout,check=True)
            payload=json.loads(rawpath.read_text())
            if 'error' in payload:raise RuntimeError(payload['error'])
            raw=payload['raw']['spotting_res'];texts=raw['rec_texts'];polys=raw['rec_polys']
            if len(texts)!=len(polys):raise ValueError('VL text/box mismatch')
            valid=[r for r in page['regions'] if not r['hold_reasons']];held=[r for r in page['regions'] if r['hold_reasons']]
            row['ignored']=0
            for text,poly in zip(texts,polys):
                box=_xyxy(poly,page['width'],page['height'],page['width'],page['height'])
                if box is None:raise ValueError('invalid VL box')
                policy=mask_policy(box,valid,held)
                if policy=='ambiguous':raise ValueError('VL overlaps held and eligible GT')
                if policy=='exclude':row['ignored']+=1;continue
                row['regions'].append(dict(bbox=box,text=text,order=len(row['regions'])+1))
            row['metadata']=payload['metadata']
        except Exception as e:row.update(error=f'{type(e).__name__}: {e}',regions=[])
        row['seconds']=time.perf_counter()-start;predictions.append(row)
        save(out/'progress.json',dict(pages=predictions));print(json.dumps(dict(id=row['id'],error=row.get('error'),seconds=row['seconds'])),flush=True)
    prediction=dict(mode='page',pages=predictions,dataset_manifest_sha256=sha256(Path(config['dataset'])/'manifest.json'))
    save(out/'predictions.validation.json',prediction)
    metrics=evaluate(pages,prediction,.5)
    metrics.update(scope='separate operational PaddleOCR-VL-1.6 spotting baseline; not same-model before/after',
        timeout_seconds=a.timeout,weights_sha256=sha256('models/vision_language/PaddleOCR-VL-1.6/model.safetensors'),
        worker_sha256=sha256('src/evaluation/paddleocr_vl_worker.py'))
    save(out/'metrics.validation.json',metrics)

if __name__=='__main__':main()
