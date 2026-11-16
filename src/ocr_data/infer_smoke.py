"""Development-only train smoke, with explicit page vs recognition-only modes.

No model download, training or test selection. GPU failures are not CPU fallbacks.
"""
import argparse
import json
import os
from pathlib import Path
import time
from src.ocr_data.audit import sha256,verify_image
from src.ocr_data.build import write


def run(dataset, mode, output, device='gpu:0', limit=2):
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK','True')
    root=Path(dataset)
    result=dict(mode=mode,scope='development smoke only; train split; not benchmark',pages=[] if mode=='page' else None,
                crops=[] if mode=='recognition_only' else None,device=device,dataset_manifest_sha256=sha256(root/'manifest.json'))
    rows=[json.loads(x) for x in (root/('pages.jsonl' if mode=='page' else 'crops.jsonl')).read_text().splitlines()]
    rows=[r for r in rows if r['split']=='train' and r.get('eligible',True) and (mode=='page' or r['type_suggestion']=='text')][:limit]
    model=None
    for r in rows:
        started=time.perf_counter();p=dict(id=r['id']) if mode=='page' else dict(page_id=r['page_id'],region_id=r['region_id'])
        try:
            if mode=='page':
                verify_image(r['image'],dict(width=r['width'],height=r['height'],sha256=r['source_sha256']))
                import cv2
                from src.cv.ocr_engine import recognize
                from src.evaluation.text_box_candidates import _xyxy
                image=cv2.imread(r['image'])
                if image is None:raise ValueError('unreadable original')
                _,raw=recognize(image,'paddleocrvl16',device=device)
                regions=[]
                for line in raw.get('lines',[]):
                    box=_xyxy(line['bbox'],r['width'],r['height'],r['width'],r['height'])
                    if box is None:raise ValueError('invalid model polygon; no silent omission')
                    regions.append(dict(bbox=box,text=line['text'],order=len(regions)+1))
                p['regions']=regions
                result['model']='PaddleOCR-VL-1.6 direct page spotting'
                result['weights_sha256']=sha256('models/vision_language/PaddleOCR-VL-1.6/model.safetensors')
            else:
                # The local PP-OCR recognition model receives the exact GT crop.
                # create_model is a recognizer, not PaddleOCR's detector pipeline.
                if model is None:
                    import paddle
                    if device.startswith('gpu') and paddle.device.cuda.device_count()<1:raise RuntimeError('GPU unavailable; no CPU fallback')
                    from paddlex import create_model
                    local=Path('models/recognition/PP-OCRv6_medium_rec')
                    if not (local/'inference.pdiparams').exists():raise RuntimeError('local recognition weights missing; no download')
                    model=create_model(model_name='PP-OCRv6_medium_rec',model_dir=str(local.resolve()),device=device)
                    result['model']='PP-OCRv6_medium_rec recognition only'
                    result['weights_sha256']=sha256(local/'inference.pdiparams')
                from src.cv.ocr_engine import _result_payload
                path=root/r['image']
                if sha256(path)!=r['crop_sha256']:raise ValueError('crop integrity mismatch')
                res=list(model.predict(str(path)))
                if len(res)!=1:raise RuntimeError('recognizer did not return exactly one result')
                payload=_result_payload(res[0]);p['text']=payload['rec_text']
        except Exception as e:p['error']=f'{type(e).__name__}: {e}'
        p['seconds']=time.perf_counter()-started
        result['pages' if mode=='page' else 'crops'].append(p)
    write(output,{k:v for k,v in result.items() if v is not None})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--mode',choices=['page','recognition_only'],required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='gpu:0');p.add_argument('--limit',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.limit<=5:p.error('development smoke limit must be 1..5; train only')
    r=run(a.dataset,a.mode,a.output,a.device,a.limit)
    print(json.dumps({k:v for k,v in r.items() if k not in ['pages','crops']},ensure_ascii=False))
if __name__=='__main__':main()
