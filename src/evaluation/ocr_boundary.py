"""Fixed-model image-only crop trim and separate provisional GT geometry audit."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
from collections import Counter
from src.evaluation.ocr_page_audit import read,write_new,sha256,intersection,ignore_state,crop_comparison,summarize,score_scope

ROOT=Path(os.environ.get('RADU_OCR_BOUNDARY_OUTPUT','data/experiments/ocr_page_audit_boundary_v1'))
SCOPE=Path('data/experiments/ocr_page_audit_scope_recovery_v1')
AUDIT=Path('data/experiments/ocr_page_audit_v1')
DATA=Path('data/processed/ocr_v2/pilot-v2')
ADAPTER=Path('data/experiments/ocr_v2_vl_lora/train/best-adapter')
POLICY=read('configs/ocr_boundary.json')


def _erase_long_runs(mask,minimum,axis):
    import numpy as np
    result=mask.copy();view=result if axis==1 else result.T
    for row in view:
        transitions=np.diff(np.pad(row.astype('int8'),(1,1)));starts=np.flatnonzero(transitions==1);ends=np.flatnonzero(transitions==-1)
        for a,b in zip(starts,ends):
            if b-a>=minimum:row[a:b]=False
    return result


def ink_bounds(image,policy):
    import numpy as np
    gray=np.asarray(image.convert('L'));mask=gray<policy['dark_threshold'];h,w=mask.shape
    mask=_erase_long_runs(mask,max(policy['horizontal_rule_min_length'],math.ceil(w*policy['horizontal_rule_width_fraction'])),1)
    mask=_erase_long_runs(mask,max(policy['vertical_rule_min_length'],math.ceil(h*policy['vertical_rule_height_fraction'])),0)
    ys,xs=np.nonzero(mask)
    if not len(xs):return [0,0,w,h]
    p=policy['margin_pixels'];return [max(0,int(xs.min())-p),max(0,int(ys.min())-p),min(w,int(xs.max())+1+p),min(h,int(ys.max())+1+p)]


def apply_geometry(gt,decisions):
    out=copy.deepcopy(gt);rows={(p['id'],g['annotation_id']):g for p in out['pages'] for g in p['regions']};trace=[];seen=set()
    for d in decisions:
        key=d['page_id'],d['annotation_id']
        if key in seen or key not in rows or not d.get('evidence') or d.get('decision') not in ['accept','reject','hold']:raise ValueError('invalid geometry decision/evidence')
        seen.add(key)
        if d['decision']!='accept':continue
        r=rows[key];b=d['bbox'];old=r['bbox']
        if not (old[0]<=b[0]<b[2]<=old[2] and old[1]<=b[1]<b[3]<=old[3]):raise ValueError('proposed geometry outside original')
        before=copy.deepcopy(r);r['bbox']=b;r['geometry_provenance']='AI_visual_checked_ink_trim_of_original_GT; provisional';r['geometry_evidence']=d['evidence'];r['review_status']='AI_geometry_review_only; no_human_certification';trace.append(dict(page_id=key[0],annotation_id=key[1],before=before,after=copy.deepcopy(r)))
    return out,trace


def paired_metrics(rows,before,after):
    ids=[(r['page_id'],r['region_id']) for r in rows]
    for pred in [before,after]:
        if [(r['page_id'],r['region_id']) for r in pred]!=ids:raise ValueError('paired ID mismatch')
    metrics=crop_comparison(rows,[r['text'] for r in before],[r['text'] for r in after]);return dict(before=metrics['C0'],after=metrics['C1'],regions=metrics['regions'],pages=metrics['pages'])


def verify():
    lock=read(ROOT/'input-lock.json')
    for p,h in lock['hashes'].items():
        if sha256(p)!=h:raise ValueError('input drift: '+p)
    if read('configs/ocr_boundary.json')!=lock['policy']:raise ValueError('boundary policy drift')
    return lock


def prepare():
    from PIL import Image,ImageDraw
    from src.evaluation.ocr_scope_recovery import verify as verify_scope
    verify_scope();ROOT.mkdir(parents=True,exist_ok=False);cropdir=ROOT/'crops';cropdir.mkdir();review=ROOT/'review';review.mkdir()
    c=read('configs/ocr_v2_vl_lora.json');files=[Path('configs/ocr_boundary.json'),Path('configs/ocr_v2_vl_lora.json'),SCOPE/'audited-gt-v3.json',SCOPE/'recognition/full500.predictions.json',AUDIT/'M.json',DATA/'recognition_candidates.validation.jsonl']
    files+=[Path(p['image']) for p in read(SCOPE/'audited-gt-v3.json')['pages']]
    files+=list(ADAPTER.iterdir())+list(Path(c['model']).iterdir())
    write_new(ROOT/'input-lock.json',dict(config=c,policy=POLICY,hashes={str(p):sha256(p) for p in files if p.is_file()},no_training=True,primary='paired validation244 then predeclared diagnosticM138; GT geometry changes separately scored with fixed raw500'))
    val=[json.loads(x) for x in (DATA/'recognition_candidates.validation.jsonl').read_text().splitlines() if json.loads(x)['page_id'] not in ['hf-0019','hf-0095']]
    diagnostic=read(AUDIT/'M.json');before_diag=[]
    for r in diagnostic:
        q=copy.deepcopy(r);q['image']=r['pred_image'];q['crop_sha256']=r['pred_crop_sha256'];before_diag.append(q)
    reviewrows=[]
    for split,rows in [('validation',val),('diagnostic',before_diag)]:
        before=[];after=[]
        for i,r in enumerate(rows):
            p=Path(r['image']);p=p if p.is_absolute() else DATA/p
            if sha256(p)!=r['crop_sha256']:raise ValueError('source crop drift')
            im=Image.open(p).convert('RGB');box=ink_bounds(im,POLICY);dest=cropdir/f'{split}-{i:04d}.png';im.crop(box).save(dest)
            a=copy.deepcopy(r);a['image']=str(p.resolve());b=copy.deepcopy(a);b['image']=str(dest.resolve());b['crop_sha256']=sha256(dest)
            before.append(a);after.append(b);reviewrows.append(dict(split=split,index=i,page_id=r['page_id'],region_id=r['region_id'],original=str(p.resolve()),trimmed=str(dest.resolve()),original_size=list(im.size),local_trim_bbox=box,changed=box!=[0,0,*im.size],text=r['text']))
        write_new(ROOT/f'{split}.before.rows.json',before);write_new(ROOT/f'{split}.after.rows.json',after)
    write_new(ROOT/'crop-review-index.json',reviewrows)
    # First8 changed validation cases, chosen by row order, not errors.
    samples=[r for r in reviewrows if r['split']=='validation' and r['changed']][:8]
    for start in [0,4]:
        canvas=Image.new('RGB',(2000,4*220),'#eee');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(samples[start:start+4]):
            d.text((5,j*220+3),f"{start+j} {r['page_id']} {r['region_id']}",fill='black')
            for x,key in [(0,'original'),(1000,'trimmed')]:
                im=Image.open(r[key]);im.thumbnail((990,180));canvas.paste(im,(x,j*220+30))
        canvas.save(review/f'validation-{start//4}.jpg')
    write_new(ROOT/'validation-visual-samples.json',samples)
    # Propose ONLY GT implicated in mixed decisions. No recognition correctness used.
    gt=read(SCOPE/'audited-gt-v3.json');bank=read(SCOPE/'recognition/full500.predictions.json');pages={p['id']:p for p in bank['pages']};proposals=[]
    for p in gt['pages']:
        valid=[g for g in p['regions'] if g['include']];ignored=[g for g in p['regions'] if not g['include']];mixed=[r for r in pages[p['id']]['regions'] if ignore_state(r['bbox'],[g['bbox'] for g in valid],[g['bbox'] for g in ignored])=='mixed_keep'];image=Image.open(p['image']).convert('RGB')
        for g in valid:
            touching=[r['region_id'] for r in mixed if intersection(g['bbox'],r['bbox'])>0]
            if not touching:continue
            b=g['bbox'];frame=[math.floor(b[0]),math.floor(b[1]),math.ceil(b[2]),math.ceil(b[3])];crop=image.crop(frame);local=ink_bounds(crop,POLICY);proposal=[max(b[0],frame[0]+local[0]),max(b[1],frame[1]+local[1]),min(b[2],frame[0]+local[2]),min(b[3],frame[1]+local[3])];index=len(proposals)
            proposals.append(dict(index=index,page_id=p['id'],annotation_id=g['annotation_id'],old_bbox=b,bbox=proposal,text=g['text'],linked_predictions=touching,source_sha256=p['source_sha256'],changed=proposal!=b))
            oldpath=review/f'geometry-{index:03d}-old.png';newpath=review/f'geometry-{index:03d}-new.png';crop.save(oldpath);image.crop([math.floor(proposal[0]),math.floor(proposal[1]),math.ceil(proposal[2]),math.ceil(proposal[3])]).save(newpath)
    write_new(ROOT/'geometry-proposals.json',proposals)
    for start in range(0,len(proposals),6):
        canvas=Image.new('RGB',(2200,6*220),'#eee');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(proposals[start:start+6]):
            d.text((5,j*220+3),f"{r['index']} {r['page_id']} {r['annotation_id']} changed={r['changed']}",fill='black')
            for x,kind in [(0,'old'),(1100,'new')]:
                im=Image.open(review/f"geometry-{r['index']:03d}-{kind}.png");im.thumbnail((1090,180));canvas.paste(im,(x,j*220+30))
        canvas.save(review/f'geometry-sheet-{start//6:02d}.jpg')
    print(dict(validation=len(val),diagnostic=len(diagnostic),geometry_candidates=len(proposals)))

def verify_method():
    lock=read(ROOT/'validation-method-lock.json')
    if lock['policy']!=POLICY or lock['policy_sha256']!=sha256('configs/ocr_boundary.json'):raise ValueError('locked boundary method drift')
    for side,h in lock['validation_rows_sha256'].items():
        if sha256(ROOT/f'validation.{side}.rows.json')!=h:raise ValueError('validation row drift')
    return lock


def lock_method():
    verify();frozen=read('configs/ocr_page_audit/boundary_validation_review.json');rows={}
    for a,b in zip(read(ROOT/'validation.before.rows.json'),read(ROOT/'validation.after.rows.json')):rows[a['page_id'],a['region_id']]=(a,b)
    for r in frozen['items']:
        a,b=rows[r['page_id'],r['region_id']]
        if a['crop_sha256']!=r['original_sha256'] or b['crop_sha256']!=r['trim_sha256']:raise ValueError('validation visual evidence changed; review required')
    ids=sorted(rows_ for rows_ in [(r['page_id'],r['region_id']) for r in frozen['items']])
    write_new(ROOT/'validation-method-lock.json',dict(policy=POLICY,policy_sha256=sha256('configs/ocr_boundary.json'),validation_rows_sha256={k:sha256(ROOT/f'validation.{k}.rows.json') for k in ['before','after']},AI_visual_samples=[list(k) for k in ids],AI_review_count=len(ids),represented_pages=len({r[0] for r in ids}),human_reviews=0,review=frozen['review'],locked_before_validation_inference=not (ROOT/'validation').exists(),no_hyperparameter_sweep=True))


def select_method(before,after):
    return 'image_trim' if after['cer_micro']<before['cer_micro'] else 'baseline'


def infer_split(split):
    import torch
    from peft import PeftModel
    from src.training.ocr_vl_lora import load,infer,encode
    from src.evaluation.ocr_page_audit_infer import tensor_receipt
    verify();lock=verify_method()
    if lock['policy']!=POLICY or lock['policy_sha256']!=sha256('configs/ocr_boundary.json'):raise ValueError('locked boundary method drift')
    if split=='diagnostic':
        selection=read(ROOT/'validation/selection-lock.json')
        if selection['policy_sha256']!=sha256('configs/ocr_boundary.json'):raise ValueError('selection drift')
    rows={k:read(ROOT/f'{split}.{k}.rows.json') for k in ['before','after']}
    if [(r['page_id'],r['region_id'],r['text']) for r in rows['before']]!=[(r['page_id'],r['region_id'],r['text']) for r in rows['after']]:raise ValueError('paired crop scope drift')
    for side in rows.values():
        for r in side:
            if sha256(r['image'])!=r['crop_sha256']:raise ValueError('crop drift')
    out=ROOT/split;out.mkdir(exist_ok=False);start=time.perf_counter();c=read(ROOT/'input-lock.json')['config']
    binding={f'{split}.{k}.rows.json':sha256(ROOT/f'{split}.{k}.rows.json') for k in rows}
    write_new(out/'start-lock.json',dict(row_hashes=binding,policy_sha256=sha256('configs/ocr_boundary.json'),source_sha256=sha256(__file__),comparison='same IDs/text/batch4 grouping, original pixels versus image-only trimmed pixels'))
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);model,proc,info=load(c);model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False);model.eval()
    if any(p.requires_grad for p in model.parameters()):raise ValueError('unexpected trainable model')
    repeats=0
    if split=='validation':
        part=rows['before'][:4];proc.tokenizer.padding_side='left';b=encode(proc,part,c,False);h=tensor_receipt(b)
        t1,_=infer(model,proc,part,c,out/'repeat1.json');proc.tokenizer.padding_side='left';b2=encode(proc,part,c,False);t2,_=infer(model,proc,part,c,out/'repeat2.json')
        if t1!=t2 or h!=tensor_receipt(b2):raise ValueError('fixed batch repeat mismatch')
        write_new(out/'repeat-check.json',dict(texts_equal=True,tensors_equal=True,tensors=h,active_adapters=list(model.active_adapters)));repeats=8
    runtimes={}
    for name in ['before','after']:
        _,rt=infer(model,proc,rows[name],c,out/f'{name}.json');runtimes[name]=rt
    bp=read(out/'before.json')['predictions'];ap=read(out/'after.json')['predictions'];metrics=paired_metrics(rows['before'],bp,ap);write_new(out/'comparison.json',metrics)
    if split=='validation':
        chosen=select_method(metrics['before'],metrics['after'])
        write_new(out/'selection-lock.json',dict(chosen=chosen,criterion=POLICY['selection'],policy_sha256=sha256('configs/ocr_boundary.json'),validation_metrics=metrics,diagnostic_used=False,method_lock_sha256=sha256(ROOT/'validation-method-lock.json')))
    verify()
    for f,h in binding.items():
        if sha256(ROOT/f)!=h:raise ValueError('row drift during inference')
    write_new(out/'receipt.json',dict(seconds=time.perf_counter()-start,runtimes=runtimes,crop_inputs=2*len(rows['before'])+repeats,repeat_inputs=repeats,active_adapters=list(model.active_adapters),dtype=str(next(model.parameters()).dtype),optimizer_steps=0,strict_load=info,versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__,peft=__import__('peft').__version__),batch_size=4))
    print(json.dumps(dict(split=split,**metrics),ensure_ascii=False))

def geometry():
    verify();review=read('configs/ocr_page_audit/boundary_geometry_v4.json');gt=read(SCOPE/'audited-gt-v3.json')
    if sha256(SCOPE/'audited-gt-v3.json')!=review['prior_GT_sha256'] or sha256(ROOT/'geometry-proposals.json')!=review['proposal_source_sha256']:raise ValueError('geometry review evidence drift')
    expected={(r['page_id'],r['annotation_id']) for r in read(ROOT/'geometry-proposals.json')};actual=[(r['page_id'],r['annotation_id']) for r in review['decisions']]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('geometry review coverage drift')
    revised,trace=apply_geometry(gt,review['decisions']);revised['revision']='v4 bbox-only AI original-pixel review; no text/include/order changes; not human-certified'
    write_new(ROOT/'geometry-review.json',review);write_new(ROOT/'audited-gt-v4.json',revised);write_new(ROOT/'geometry-changes.json',trace)
    # Recreate evidence assets for manual proposals on a fresh run.
    from PIL import Image
    proposals={(r['page_id'],r['annotation_id']):r for r in read(ROOT/'geometry-proposals.json')};pages={p['id']:p for p in gt['pages']};manual=[]
    for d in review['decisions']:
        key=d['page_id'],d['annotation_id'];proposal=proposals[key]
        if d['decision']=='accept' and d['bbox']!=proposal['bbox']:
            i=d['proposal_index'];manual.append(dict(index=i,old_bbox=proposal['old_bbox'],bbox=d['bbox'],page_id=key[0],annotation_id=key[1],basis='image-only original review'))
            path=ROOT/'review'/f'manual-{i:03d}.png'
            if path.exists():raise FileExistsError(path)
            Image.open(pages[key[0]]['image']).crop(d['bbox']).save(path)
    write_new(ROOT/'manual-boundary-proposals.json',manual)
    pred=read(SCOPE/'recognition/full500.predictions.json');before=score_scope(gt['pages'],pred);after=score_scope(revised['pages'],pred)
    if before['page']['reference_units']!=after['page']['reference_units']:raise ValueError('geometry changed character denominator')
    write_new(ROOT/'geometry-comparison.json',dict(before=before,after=after,GT_regions=sum(g['include'] for p in gt['pages'] for g in p['regions']),bbox_changes=len(trace),same_predictions_sha256=sha256(SCOPE/'recognition/full500.predictions.json'),interpretation='evaluation GT geometry sensitivity only; no recognizer/pipeline improvement'))
    print(json.dumps(dict(changes=len(trace),before=before['page'],after=after['page'],detection_before=before['detection'],detection_after=after['detection']),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','method-lock','validation','diagnostic','geometry'],required=True);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='geometry':geometry()
    elif a.stage=='method-lock':lock_method()
    else:infer_split(a.stage)
