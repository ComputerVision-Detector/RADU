"""Fixed selected adapter, paired crops only; no training or candidate selection."""
import argparse
import copy
import json
import math
from pathlib import Path
import platform
import time

from src.evaluation.ocr_page_audit import ROOT,ADAPTER,DATA,read,write_new,verify_lock,inputs,clear_pairs,intersection
from src.ocr_data.audit import sha256
from src.training.ocr_vl_lora import load,infer,encode


def prepare_rows():
    from PIL import Image
    verify_lock();pages=read(ROOT/'audited-gt-v1.json')['pages'];_,_,_,pred=inputs()
    crops=ROOT/'new-crops';crops.mkdir(exist_ok=False);p1=[];m=[];excluded=[]
    for page in pages:
        refs=sorted([g for g in page['regions'] if g['include']],key=lambda r:r['order']);pp=next(p for p in pred['pages'] if p['id']==page['id']);preds=sorted(pp['regions'],key=lambda r:r['order'])
        allboxes=[g['bbox'] for g in refs];pairs=clear_pairs([r['bbox'] for r in preds],allboxes)
        paired={j:i for i,j in pairs}
        im=Image.open(page['image']).convert('RGB')
        for j,g in enumerate(refs):
            path=crops/f"{page['id']}-{j:04d}.png";b=g['bbox'];box=[math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3])];im.crop(box).save(path)
            row=dict(page_id=page['id'],region_id=g['annotation_id'],image=str(path.resolve()),crop_sha256=sha256(path),text=g['text'],bbox=b,gt_order=g['order']);p1.append(row)
            if j not in paired:excluded.append(dict(page_id=page['id'],annotation_id=g['annotation_id'],reason='not unique IoU>=.5 pair with one >.1 neighbor each'));continue
            pr=preds[paired[j]]
            held_overlap=any(intersection(pr['bbox'],h['bbox'])>0 for h in page['regions'] if not h['include'])
            if g['geometry_provenance']!='original_V2' or held_overlap:
                excluded.append(dict(page_id=page['id'],annotation_id=g['annotation_id'],reason='detector-assisted new GT or ignored-neighbor intersection'));continue
            m.append(dict(**row,prediction_id=page['id']+'/'+pr['region_id'],pred_image=pr['image'],pred_bbox=pr['bbox'],pred_order=pr['order'],pred_crop_sha256=pr['crop_sha256'],cached_C1=pr['text']))
    write_new(ROOT/'P1-rows.json',p1);write_new(ROOT/'M.json',m);write_new(ROOT/'M-excluded.json',excluded)
    write_new(ROOT/'inference-lock.json',dict(gt_sha256=sha256(ROOT/'audited-gt-v1.json'),M_sha256=sha256(ROOT/'M.json'),P1_rows_sha256=sha256(ROOT/'P1-rows.json'),batch=4,dtype='bfloat16',C0_C1_grouping='same M row IDs, same batch4 membership/order; image geometry is sole experimental change',P1_grouping='all included GT ordered page then audited spatial order, batch4; distinct batch context from M so separate diagnostic'))
    print(json.dumps(dict(M=len(m),P1=len(p1),M_pages=len({r['page_id'] for r in m}))))


def tensor_receipt(batch):
    import torch
    result={}
    for k,v in batch.items():
        if isinstance(v,torch.Tensor):
            # bytes through uint8 handles bfloat16.
            import hashlib
            result[k]=dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
    return result


def run():
    import torch
    from peft import PeftModel
    verify_lock();lock=read(ROOT/'inference-lock.json')
    for file,key in [('audited-gt-v1.json','gt_sha256'),('M.json','M_sha256'),('P1-rows.json','P1_rows_sha256')]:
        if sha256(ROOT/file)!=lock[key]:raise ValueError('inference input drift')
    out=ROOT/'inference';out.mkdir(exist_ok=False);start=time.perf_counter();c=read(ROOT/'input-lock.json')['config'];c=copy.deepcopy(c)
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model,proc,info=load(c);model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False);model.eval();proc.tokenizer.padding_side='left'
    if any(p.requires_grad for p in model.parameters()):raise ValueError('model unexpectedly trainable')
    # Repeat first original detector batch to test original cached context, not cherry-picked output.
    _,_,_,pred=inputs();representative=[r for p in pred['pages'] for r in p['regions']][:4]
    b=encode(proc,representative,c,False);bt=tensor_receipt(b)
    r1,_=infer(model,proc,representative,c,out/'repeat1.json');proc.tokenizer.padding_side='left';b2=encode(proc,representative,c,False)
    r2,_=infer(model,proc,representative,c,out/'repeat2.json')
    reproducible=r1==r2 and bt==tensor_receipt(b2)
    if not reproducible:raise ValueError('same batch repeat not reproducible')
    write_new(out/'reproducibility.json',dict(repeat_equal=True,input_tensors_equal=True,tensors=bt,cached_original_texts=[r['text'] for r in representative],new_texts=r1,cached_equal=r1==[r['text'] for r in representative],active_adapters=list(model.active_adapters),eval_mode=not model.training,padding='left',strict_load=info))
    m=read(ROOT/'M.json');p1=read(ROOT/'P1-rows.json');c0=copy.deepcopy(m);c1=copy.deepcopy(m)
    for r in c1:r['image']=r['pred_image'];r['crop_sha256']=r['pred_crop_sha256']
    for rows in [c0,c1,p1]:
        for r in rows:
            if sha256(r['image'])!=r['crop_sha256']:raise ValueError('crop drift')
    for name,rows in [('C0',c0),('C1',c1),('P1',p1)]:
        _,rt=infer(model,proc,rows,c,out/f'{name}.json');print(name,rt,flush=True)
    verify_lock()
    write_new(out/'receipt.json',dict(seconds=time.perf_counter()-start,versions=dict(python=platform.python_version(),torch=torch.__version__,transformers=__import__('transformers').__version__,peft=__import__('peft').__version__),gpu=torch.cuda.get_device_name(0),config=c,active_adapters=list(model.active_adapters),model_dtype=str(next(model.parameters()).dtype),source_hashes={p:sha256(p) for p in ['src/evaluation/ocr_page_audit_infer.py','src/training/ocr_vl_lora.py']},inference_lock_sha256=sha256(ROOT/'inference-lock.json'),optimizer_steps=0))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','infer'],required=True);a=p.parse_args();{'prepare':prepare_rows,'infer':run}[a.stage]()
