"""Recover legacy scope-filtered detections with fixed detector and selected LoRA.
No GT filtering in detector/inference, no optimizer, no parameter tuning.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
import platform
from pathlib import Path
import time
from src.evaluation.ocr_page_audit import read,write_new,sha256,score_scope,normalize

ROOT=Path(os.environ.get('RADU_OCR_SCOPE_OUTPUT','data/experiments/ocr_page_audit_scope_recovery_v1'))
PRIOR=Path('data/experiments/ocr_page_audit_v1')
OLD=Path('data/experiments/ocr_v2_vl_lora/pages/after.test.predictions.json')
DETECTOR=Path('models/detection/PP-OCRv6_medium_det')
ADAPTER=Path('data/experiments/ocr_v2_vl_lora/train/best-adapter')
WEIGHT='ab1bfcfba5c0711ebfc82e83432eca13b47fbb48ca0800d9e7b4d590e15421e5'


def keyed(bank):
    out={}
    for p in bank['pages']:
        for r in p['regions']:
            key=p['id']+'/'+r['region_id']
            if key in out:raise ValueError('duplicate prediction ID')
            out[key]=r
    return out


def recovery_map(old,raw):
    before=keyed(old);after=keyed(raw)
    if set(before)-set(after):raise ValueError('missing old prediction IDs')
    for key,r in before.items():
        if r['bbox']!=after[key]['bbox']:raise ValueError(f'geometry drift: {key}')
        if r['crop_sha256']!=after[key]['crop_sha256']:raise ValueError(f'crop drift: {key}')
    return list(before),[k for k in after if k not in before]


def merge_bank(old,raw,extra_texts):
    shared,extra=recovery_map(old,raw)
    if set(extra_texts)!=set(extra):raise ValueError('extra text coverage/old text overwrite')
    oldrows=keyed(old);result=copy.deepcopy(raw)
    for p in result['pages']:
        for r in p['regions']:
            key=p['id']+'/'+r['region_id'];r['text']=oldrows[key]['text'] if key in oldrows else extra_texts[key]
    return result


def revise_gt(gt,decisions):
    out=copy.deepcopy(gt);rows={(p['id'],r['annotation_id']):r for p in out['pages'] for r in p['regions']};changes=[];seen=set()
    for d in decisions:
        key=d['image_id'],d['annotation_id']
        if key in seen or key not in rows or not d.get('evidence'):raise ValueError('invalid GT review decision/evidence')
        seen.add(key);r=rows[key];before=copy.deepcopy(r)
        for field in ['text','include','ignore_reason','uncertainty']:
            if field in d:r[field]=d[field]
        r['review_status']='AI_rechecked_original_image; no_independent_human_review';r['change_evidence']=d['evidence']
        changes.append(dict(image_id=key[0],annotation_id=key[1],before=before,after=copy.deepcopy(r),evidence=d['evidence']))
    return out,changes


def verify():
    lock=read(ROOT/'input-lock.json')
    for path,h in lock['hashes'].items():
        if sha256(path)!=h:raise ValueError(f'input drift: {path}')
    if sha256(ADAPTER/'adapter_model.safetensors')!=WEIGHT:raise ValueError('adapter drift')
    return lock


def prepare():
    from PIL import Image,ImageDraw
    from src.evaluation.ocr_page_audit import verify_lock
    verify_lock();ROOT.mkdir(parents=True,exist_ok=False)
    c=read('configs/ocr_v2_vl_lora.json');old=read(OLD);pages=read(PRIOR/'source-pages.json');receipt=read('data/experiments/ocr_v2_korean/det-test/detections.json')
    for name,h in receipt['detector_hashes'].items():
        if sha256(DETECTOR/name)!=h:raise ValueError('historical detector file drift: '+name)
    files=[OLD,PRIOR/'audited-gt-v2.json',PRIOR/'fp201-audit-v2.json',PRIOR/'source-pages.json',Path('configs/ocr_v2_vl_lora.json'),Path('data/experiments/ocr_v2_korean/det-test/detections.json')]
    files+=list(DETECTOR.iterdir())+list(ADAPTER.iterdir())+list(Path(c['model']).iterdir())+[Path(p['image']) for p in pages]
    write_new(ROOT/'input-lock.json',dict(config=c,hashes={str(p):sha256(p) for p in files if p.is_file()},detector='same PP-OCRv6_medium_det local inference.yml defaults',recognition=dict(batch=4,dtype='bfloat16',attention='sdpa',prompt='OCR:',beams=1,do_sample=False,max_new_tokens=128),policy='no detector GT filtering; only evaluation ignore; old397 strings and batch contexts unchanged, extra103 fixed batch4 in raw page/order; no score-based choice',primary_GT='prior auditedGTv2 frozen306 refs',no_new_training=True))
    write_new(ROOT/'prior-gt-v2.json',read(PRIOR/'audited-gt-v2.json'))
    review=ROOT/'review';review.mkdir()
    targets=[];gt=read(PRIOR/'audited-gt-v2.json');source={p['id']:p for p in pages}
    fp=read(PRIOR/'fp201-audit-v2.json')
    for r in fp:
        if r['primary']=='uncertain' or r['index']==112:
            targets.append(dict(image_id=r['page_id'],annotation_id=f"audit-{r['page_id']}-{r['index']:03d}",bbox=r['bbox'],kind='held_FP_or_truncated_website',fp_index=r['index']))
    for p in gt['pages']:
        for g in p['regions']:
            if not g['include'] and g['geometry_provenance']=='original_V2' and g['region_type']=='text':targets.append(dict(image_id=p['id'],annotation_id=g['annotation_id'],bbox=g['bbox'],kind='held_source_GT',original_text=g['text']))
    for i,r in enumerate(targets):
        page=source[r['image_id']];b=r['bbox'];box=[max(0,int(b[0])-20),max(0,int(b[1])-20),min(page['width'],math.ceil(b[2])+20),min(page['height'],math.ceil(b[3])+20)]
        with Image.open(page['image']) as im:
            crop=im.crop(box).convert('RGB');crop.thumbnail((2300,600));crop.save(review/f'item-{i:02d}.png')
        r['review_image']=f'review/item-{i:02d}.png'
    write_new(ROOT/'review-queue.json',targets)
    # Viewable contact sheets without recognition strings, source GT only in queue.
    for start in range(0,len(targets),8):
        canvas=Image.new('RGB',(2400,8*190),'#eee');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(targets[start:start+8]):
            im=Image.open(ROOT/r['review_image']);im.thumbnail((2000,170));canvas.paste(im,(380,j*190+10));d.text((5,j*190+8),f"{start+j} {r['image_id']}\n{r['annotation_id'][-30:]}",fill='black')
        canvas.save(review/f'queue-{start//8}.jpg')
    print(dict(review_items=len(targets),old_predictions=len(keyed(old)),expected_recovered=103))


def detect():
    from PIL import Image
    verify();out=ROOT/'raw-detector';out.mkdir(exist_ok=False);cropdir=out/'crops';cropdir.mkdir()
    os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK']='True';os.environ['PADDLE_PDX_CACHE_HOME']=str(Path('.cache/paddlex').resolve())
    from paddlex import create_model
    from src.cv.ocr_engine import _result_payload
    model=create_model(model_name='PP-OCRv6_medium_det',model_dir=str(DETECTOR.resolve()),device='gpu:0')
    pages=read(PRIOR/'source-pages.json');records=[];start=time.perf_counter()
    for p in pages:
        t=time.perf_counter();payload=_result_payload(list(model.predict(p['image']))[0]);boxes=[]
        for poly in payload['dt_polys']:
            b=[max(0,math.floor(float(min(x[0] for x in poly)))),max(0,math.floor(float(min(x[1] for x in poly)))),min(p['width'],math.ceil(float(max(x[0] for x in poly)))),min(p['height'],math.ceil(float(max(x[1] for x in poly))))]
            if b[2]<=b[0] or b[3]<=b[1]:raise ValueError('invalid reconstructed detector box')
            boxes.append(b)
        regions=[]
        with Image.open(p['image']) as im:
            for i,b in enumerate(sorted(boxes,key=lambda b:(b[1],b[0],b[3],b[2]))):
                dest=cropdir/f"{p['id']}-{i:04d}.png";im.crop(b).save(dest)
                regions.append(dict(page_id=p['id'],region_id=f'det-{i:04d}',bbox=b,order=i,image=str(dest.resolve()),crop_sha256=sha256(dest),text=''))
        records.append(dict(id=p['id'],source_sha256=p['source_sha256'],regions=regions,seconds=time.perf_counter()-t));print(p['id'],len(regions),flush=True)
    raw=dict(mode='page',pages=records);write_new(out/'raw500.json',raw)
    shared,extra=recovery_map(read(OLD),raw)
    write_new(out/'recovery.json',dict(shared=shared,extra=extra,shared_count=len(shared),extra_count=len(extra),all_retained_geometry_and_crops_identical=True,seconds=time.perf_counter()-start,versions=dict(python=platform.python_version(),paddle=__import__('paddle').__version__,paddlex=__import__('paddlex').__version__),source_sha256=sha256(__file__)))
    if len(shared)!=397 or len(extra)!=103 or len(keyed(raw))!=500:raise ValueError('historical count drift; do not force recovery')
    extras=[dict(key=key,**keyed(raw)[key]) for key in extra];write_new(ROOT/'extra103-rows.json',extras)
    verify();print(dict(shared=len(shared),extra=len(extra)))

def restored_review():
    from PIL import Image,ImageDraw
    from src.evaluation.ocr_page_audit import ignore_state,intersection,validate_reviews
    verify();rows=read(ROOT/'extra103-rows.json');pages={p['id']:p for p in read(ROOT/'prior-gt-v2.json')['pages']}
    frozen=read('configs/ocr_page_audit/scope_restored_visual.json')['items'];validate_reviews([r['key'] for r in rows],frozen);byid={r['prediction_id']:r for r in frozen};audit=[]
    for r in rows:
        if r['crop_sha256']!=byid[r['key']]['crop_sha256']:raise ValueError('visual evidence crop drift; new review required')
        p=pages[r['page_id']];valid=[g for g in p['regions'] if g['include']];held=[g for g in p['regions'] if not g['include']]
        state=ignore_state(r['bbox'],[g['bbox'] for g in valid],[g['bbox'] for g in held])
        audit.append(dict(prediction_id=r['key'],image_id=r['page_id'],bbox=r['bbox'],scope_policy=state,valid_GT=[g['annotation_id'] for g in valid if intersection(r['bbox'],g['bbox'])>0],held_GT=[g['annotation_id'] for g in held if intersection(r['bbox'],g['bbox'])>0],confidence=None,review_status='frozen AI image review replay; no independent human'))
    write_new(ROOT/'restored103-geometry-scope.json',audit)
    for start in range(0,len(rows),12):
        canvas=Image.new('RGB',(1600,12*155),'#eee');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[start:start+12]):
            im=Image.open(r['image']).convert('RGB');im.thumbnail((1260,145));canvas.paste(im,(330,j*155+4));d.text((5,j*155+10),f"{start+j:03} {r['key']}\n{audit[start+j]['scope_policy']}",fill='black')
        path=ROOT/'review'/f'restored-{start//12:02d}.jpg'
        if path.exists():raise FileExistsError(path)
        canvas.save(path)


def review_gt():
    """Apply frozen image-only review decisions, not a new automatic reviewer."""
    verify();d=read('configs/ocr_page_audit/scope_review_v3.json');old=read(ROOT/'prior-gt-v2.json')
    v3,changes=revise_gt(old,d['decisions']);v3['revision']='v3 original-image enlarged recheck37; no geometry/order edits; explicit text/inclusion changes only'
    write_new(ROOT/'review37-decisions.json',d);write_new(ROOT/'audited-gt-v3.json',v3);write_new(ROOT/'gt-v3-changes.json',changes)
    write_new(ROOT/'review-freeze.json',dict(GTv2=sha256(ROOT/'prior-gt-v2.json'),GTv3=sha256(ROOT/'audited-gt-v3.json'),decisions=sha256(ROOT/'review37-decisions.json'),included_before=sum(g['include'] for p in old['pages'] for g in p['regions']),included_after=sum(g['include'] for p in v3['pages'] for g in p['regions']),before_extra_recognition=not (ROOT/'recognition').exists(),no_independent_human_review=True))


def verify_review_freeze(root=None):
    root=Path(root) if root else ROOT;freeze=read(root/'review-freeze.json')
    for name,key in [('prior-gt-v2.json','GTv2'),('audited-gt-v3.json','GTv3'),('review37-decisions.json','decisions')]:
        if sha256(root/name)!=freeze[key]:raise ValueError('review freeze drift: '+name)
    return freeze


def recognize():
    import torch
    from peft import PeftModel
    from src.training.ocr_vl_lora import load,infer,encode
    from src.evaluation.ocr_page_audit_infer import tensor_receipt
    verify();verify_review_freeze();out=ROOT/'recognition';out.mkdir(exist_ok=False);start=time.perf_counter();rows=read(ROOT/'extra103-rows.json')
    if len(rows)!=103 or len({r['key'] for r in rows})!=103:raise ValueError('extra103 ID mismatch')
    for r in rows:
        if sha256(r['image'])!=r['crop_sha256']:raise ValueError('extra crop drift')
    binding=['raw-detector/raw500.json','raw-detector/recovery.json','extra103-rows.json','prior-gt-v2.json','audited-gt-v3.json','review37-decisions.json','review-freeze.json']
    write_new(out/'start-lock.json',dict(inputs={f:sha256(ROOT/f) for f in binding},source_sha256=sha256(__file__),batch_schedule='cached397 unchanged original batch4; appended extra103 in historical raw page/order batches4; assembly reorders by raw detector order',model_weight_sha256=sha256(ADAPTER/'adapter_model.safetensors'),no_optimizer=True))
    c=read(ROOT/'input-lock.json')['config'];torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model,proc,info=load(c);model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False);model.eval()
    if any(p.requires_grad for p in model.parameters()):raise ValueError('model unexpectedly trainable')
    representative=[r for p in read(OLD)['pages'] for r in p['regions']][:4];proc.tokenizer.padding_side='left';b=encode(proc,representative,c,False);tensors=tensor_receipt(b)
    a,_=infer(model,proc,representative,c,out/'repeat1.json');proc.tokenizer.padding_side='left';b2=encode(proc,representative,c,False);btxt,_=infer(model,proc,representative,c,out/'repeat2.json')
    repeat=dict(equal=a==btxt,tensors_equal=tensors==tensor_receipt(b2),cached_equal=a==[r['text'] for r in representative],active_adapters=list(model.active_adapters),tensors=tensors)
    write_new(out/'repeat-check.json',repeat)
    if not all(repeat[k] for k in ['equal','tensors_equal','cached_equal']):raise ValueError('fixed context/cached repeat drift; do not mix conditions')
    texts,rt=infer(model,proc,rows,c,out/'extra103.json')
    full=merge_bank(read(OLD),read(ROOT/'raw-detector/raw500.json'),{r['key']:t for r,t in zip(rows,texts)})
    write_new(out/'full500.predictions.json',full)
    verify()
    for f,h in read(out/'start-lock.json')['inputs'].items():
        if sha256(ROOT/f)!=h:raise ValueError('new artifact drift during inference')
    write_new(out/'receipt.json',dict(seconds=time.perf_counter()-start,extra_runtime=rt,crop_inputs=111,extra_crop_inputs=103,repeated_crop_inputs=8,failures=rt['failures'],config=c,dtype=str(next(model.parameters()).dtype),active_adapters=list(model.active_adapters),all_frozen=True,optimizer_steps=0,versions=dict(python=platform.python_version(),torch=torch.__version__,transformers=__import__('transformers').__version__,peft=__import__('peft').__version__),strict_load=info))

def paired_scope_compare(pages,before,after):
    shared,extra=recovery_map(before,after);b=keyed(before);a=keyed(after)
    if any(b[key]['text']!=a[key]['text'] for key in shared):raise ValueError('shared recognition drift')
    x=score_scope(pages,before);y=score_scope(pages,after)
    if x['page']['reference_units']!=y['page']['reference_units']:raise ValueError('GT denominator drift')
    return dict(before=x,after=y,shared_strings_identical=True,shared_predictions=len(shared),extra_predictions=len(extra),CER_delta=y['page']['cer_micro']-x['page']['cer_micro'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','review-gt','detect','restored-review','recognize'],required=True);a=p.parse_args();{'prepare':prepare,'review-gt':review_gt,'detect':detect,'restored-review':restored_review,'recognize':recognize}[a.stage]()
