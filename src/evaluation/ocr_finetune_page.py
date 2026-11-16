"""Frozen PP-OCRv6 detector + before/after Korean recognizer experiment.

Original images enter the same detector once. Predictions predominantly within
excluded GT (formula/symbol/unsupported text) are ignored for both recognizers.
Ambiguous overlaps fail the whole page and keep its GT in the CER denominator.
This measures the screened natural-text scope, not unrestricted document OCR.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time

from src.evaluation.ocr_v2 import evaluate
from src.ocr_data.audit import sha256
from src.training.ocr_finetune import bootstrap_model, infer, readrows, save,verify_training_binding


def mask_policy(box, valid, held):
    def inside(ref):
        b=ref['bbox'];intersection=max(0,min(box[2],b[2])-max(box[0],b[0]))*max(0,min(box[3],b[3])-max(box[1],b[1]))
        return intersection/((box[2]-box[0])*(box[3]-box[1]))>=.5
    a=any(inside(r) for r in valid);b=any(inside(r) for r in held)
    return 'ambiguous' if a and b else 'exclude' if b else 'keep'


def ordered_boxes(boxes):
    return sorted(boxes,key=lambda b:(b[1],b[0],b[3],b[2]))


def pages_for(config,split):
    accepted=json.loads((Path(config['prepared'])/f'{split}.json').read_text())
    keys={(r['page_id'],r['region_id']) for r in accepted}
    pages=[p for p in readrows(Path(config['dataset'])/'pages.jsonl') if p['split']==split]
    for p in pages:
        for r in p['regions']:
            if (p['id'],r['id']) not in keys:r['hold_reasons']=sorted(set(r['hold_reasons']+['outside_screened_natural_text_scope']))
    return pages


def detect(config,split,out):
    import cv2
    from PIL import Image
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True'
    os.environ['PADDLE_PDX_CACHE_HOME']=str(Path('.cache/paddlex').resolve())
    from paddlex import create_model
    from src.cv.ocr_engine import _result_payload
    model_dir=Path('models/detection/PP-OCRv6_medium_det')
    detector=create_model(model_name='PP-OCRv6_medium_det',model_dir=str(model_dir.resolve()),device=config['device'])
    records=[];cropdir=out/'crops';cropdir.mkdir()
    for p in pages_for(config,split):
        if not p['eligible']:continue
        record=dict(id=p['id'],regions=[],source_sha256=p['source_sha256']);start=time.perf_counter()
        try:
            if sha256(p['image'])!=p['source_sha256']:raise ValueError('original hash changed')
            payload=_result_payload(list(detector.predict(p['image']))[0]);polys=payload['dt_polys']
            boxes=[]
            for poly in polys:
                box=[max(0,math.floor(float(min(x[0] for x in poly)))),max(0,math.floor(float(min(x[1] for x in poly)))),
                     min(p['width'],math.ceil(float(max(x[0] for x in poly)))),min(p['height'],math.ceil(float(max(x[1] for x in poly))))]
                if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('degenerate detector box')
                boxes.append(box)
            valid=[r for r in p['regions'] if not r['hold_reasons']];held=[r for r in p['regions'] if r['hold_reasons']]
            record['raw_detections']=len(boxes);record['ignored']=0
            with Image.open(p['image']) as im:
                for i,box in enumerate(ordered_boxes(boxes)):
                    policy=mask_policy(box,valid,held)
                    if policy=='ambiguous':raise ValueError('detector region overlaps both eligible and excluded GT')
                    if policy=='exclude':record['ignored']+=1;continue
                    target=cropdir/f"{p['id']}-{i:04d}.png";im.crop(box).save(target)
                    record['regions'].append(dict(bbox=box,text='',order=i,image=str(target.resolve()),crop_sha256=sha256(target),page_id=p['id'],region_id=f'det-{i:04d}'))
        except Exception as e:record.update(error=f'{type(e).__name__}: {e}',regions=[])
        record['seconds']=time.perf_counter()-start;records.append(record)
        print(json.dumps(dict(page=p['id'],regions=len(record['regions']),error=record.get('error'),seconds=record['seconds'])),flush=True)
    save(out/'detections.json',dict(split=split,pages=records,detector='PP-OCRv6_medium_det',
         detector_hashes={p.name:sha256(p) for p in model_dir.iterdir() if p.is_file()},
         dataset_manifest_sha256=sha256(Path(config['dataset'])/'manifest.json'),
         detector_settings='local inference.yml defaults; frozen; no recognition-score filter; original pixels',
         order='bbox top y then left x; identical for both models',exclusion='prediction area >=50% in excluded GT; ambiguous page fails closed'))


def recognize(config,split,detections,out):
    import paddle
    verify_training_binding(config)
    frozen=json.loads(Path(detections).read_text())
    if frozen['split']!=split or frozen['dataset_manifest_sha256']!=sha256(Path(config['dataset'])/'manifest.json'):raise ValueError('detector provenance mismatch')
    rows=[r for p in frozen['pages'] for r in p['regions']]
    for r in rows:
        if sha256(r['image'])!=r['crop_sha256']:raise ValueError('detector crop drift')
    pages=pages_for(config,split);model,decoder,_=bootstrap_model(config)
    for name in ['before','after']:
        if name=='after':model.set_state_dict(paddle.load(str(Path(config['trained'])/'best.pdparams')))
        texts,runtime=infer(model,decoder,rows,'.',config)
        keyed={(r['page_id'],r['region_id']):t for r,t in zip(rows,texts)}
        prediction=dict(mode='page',dataset_manifest_sha256=frozen['dataset_manifest_sha256'],pages=copy.deepcopy(frozen['pages']))
        for page in prediction['pages']:
            for r in page['regions']:r['text']=keyed[(r['page_id'],r['region_id'])]
        result=evaluate(pages,prediction,.5);result.update(split=split,scope='screened natural-text full-page OCR with excluded regions ignored',recognizer_runtime=runtime,
            detector_seconds=sum(p['seconds'] for p in frozen['pages']),detector_prediction_sha256=sha256(detections),
            ignored_detections=sum(p.get('ignored',0) for p in frozen['pages']),config=config)
        save(out/f'{name}.{split}.pages.predictions.json',prediction);save(out/f'{name}.{split}.pages.metrics.json',result)
        print(json.dumps(dict(model=name,split=split,detection=result['detection'],end_to_end=result['end_to_end'],failures=result['inference_failures']),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--split',choices=['validation','test'],required=True);p.add_argument('--stage',choices=['detect','recognize'],required=True);p.add_argument('--detections');p.add_argument('--output',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    import paddle
    paddle.set_device(config['device'])
    if a.stage=='detect':detect(config,a.split,out)
    else:recognize(config,a.split,a.detections,out)

if __name__=='__main__':main()
