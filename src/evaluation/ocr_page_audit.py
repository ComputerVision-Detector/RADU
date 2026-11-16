"""Fixed-model page diagnostics. No optimizer, model selection or threshold search."""
from __future__ import annotations
import argparse
import copy
import csv
import html
import json
import os
from collections import Counter
from pathlib import Path

from src.ocr_data.audit import sha256
from src.evaluation.evaluate import _iou, _prf
from src.evaluation.ocr_metrics import normalize
from src.evaluation.ocr_v2 import match_boxes, summarize, evaluate
from src.evaluation.ocr_finetune_page import pages_for

ROOT=Path(os.environ.get('RADU_PAGE_AUDIT_ROOT','data/experiments/ocr_page_audit_v1'))
DATA=Path('data/processed/ocr_v2/pilot-v2')
PRED=Path('data/experiments/ocr_v2_vl_lora/pages/after.test.predictions.json')
ADAPTER=Path('data/experiments/ocr_v2_vl_lora/train/best-adapter')
EXPECTED='ab1bfcfba5c0711ebfc82e83432eca13b47fbb48ca0800d9e7b4d590e15421e5'


def read(p):return json.loads(Path(p).read_text())

def write_new(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')

def intersection(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def ignore_state(box,valid,ignored):
    area=(box[2]-box[0])*(box[3]-box[1])
    if area<=0:raise ValueError('invalid box')
    held=any(intersection(box,g)/area>=.5 for g in ignored)
    valid_hit=any(intersection(box,g)>0 for g in valid)
    return 'mixed_keep' if held and valid_hit else 'ignore' if held else 'keep'

def validate_reviews(ids,reviews):
    keys=[r['prediction_id'] for r in reviews]
    if len(keys)!=len(set(keys)) or set(keys)!=set(ids):raise ValueError('FP coverage mismatch')

def clear_pairs(preds,refs):
    # Fixed geometric correspondence: one IoU>.1 neighbor each AND IoU>=.5.
    pn=[[j for j,g in enumerate(refs) if _iou(p,g)>.1] for p in preds]
    gn=[[i for i,p in enumerate(preds) if _iou(p,g)>.1] for g in refs]
    return [(i,j) for i,j,s in match_boxes(preds,refs,.5) if len(pn[i])==1 and len(gn[j])==1]

def crop_comparison(rows,c0,c1):
    if len(rows)!=len(c0) or len(rows)!=len(c1):raise ValueError('paired denominator mismatch')
    if len({(r['page_id'],r['region_id']) for r in rows})!=len(rows):raise ValueError('duplicate M')
    return dict(regions=len(rows),pages=len({r['page_id'] for r in rows}),C0=summarize([(r['text'],t) for r,t in zip(rows,c0)]),C1=summarize([(r['text'],t) for r,t in zip(rows,c1)]))

def order_diagnostic(rows,reference):
    if len({r['prediction_id'] for r in rows})!=len(rows):raise ValueError('duplicate O prediction')
    ordered=sorted(rows,key=lambda r:r['gt_order'])
    assert Counter((r['prediction_id'],r['text']) for r in rows)==Counter((r['prediction_id'],r['text']) for r in ordered)
    orders=[r['gt_order'] for r in rows];inv=sum(a>b for i,a in enumerate(orders) for b in orders[i+1:]);pairs=len(rows)*(len(rows)-1)//2
    ref='\n'.join(reference)
    return dict(regions=len(rows),inversions=inv,possible_pairs=pairs,order_error_rate=inv/pairs if pairs else None,O0=summarize([(ref,'\n'.join(r['text'] for r in rows))]),O1=summarize([(ref,'\n'.join(r['text'] for r in ordered))]))

def inputs():
    c=read('configs/ocr_v2_vl_lora.json');old=read('configs/ocr_v2_korean_ctc.json');legacy=pages_for(old,'test');ids={p['id'] for p in legacy}
    pages=[json.loads(x) for x in (DATA/'pages.jsonl').read_text().splitlines() if json.loads(x)['id'] in ids]
    pred=read(PRED)
    return c,legacy,pages,pred

def verify_lock():
    lock=read(ROOT/'input-lock.json')
    for p,h in lock['hashes'].items():
        if sha256(p)!=h:raise ValueError(f'input drift: {p}')
    if read('configs/ocr_v2_vl_lora.json')!=lock['config']:raise ValueError('config drift')
    return lock

def prepare():
    from PIL import Image,ImageDraw
    ROOT.mkdir(parents=True,exist_ok=False)
    c,legacy,pages,pred=inputs()
    files=[PRED,DATA/'pages.jsonl',DATA/'manifest.json',Path('configs/ocr_v2_vl_lora.json'),Path('configs/ocr_v2_korean_ctc.json'),Path('data/experiments/ocr_v2_korean/det-test/detections.json')]
    files+=list(ADAPTER.glob('*'))+[p for p in Path(c['model']).iterdir() if p.is_file()]
    files += [Path(p['image']) for p in pages]
    files += [Path(r['image']) for p in pred['pages'] for r in p['regions']]
    if sha256(ADAPTER/'adapter_model.safetensors')!=EXPECTED:raise ValueError('wrong adapter')
    write_new(ROOT/'input-lock.json',dict(config=c,hashes={str(p):sha256(p) for p in files if p.is_file()},dtype='bfloat16',attention='sdpa',batch=4,decode='greedy beams1 max_new_tokens128',prompt='OCR:',model_frozen=True,diagnostic_set_not_independent=True))
    result=evaluate(legacy,pred,.5);write_new(ROOT/'legacy-reproduced.json',result)
    write_new(ROOT/'legacy-gt.json',legacy);write_new(ROOT/'source-pages.json',pages)
    fps=[]
    for case in result['cases']:
        pp=next(p for p in pred['pages'] if p['id']==case['id'])
        for i in case['false_positives']:
            r=copy.deepcopy(pp['regions'][i]);r.update(prediction_id=case['id']+'/'+r['region_id'],legacy_index=i,legacy_match=None,confidence=None,confidence_note='detector confidence and autoregressive score not saved')
            fps.append(r)
    assert len(fps)==201 and len({r['prediction_id'] for r in fps})==201
    write_new(ROOT/'frozen-fp201.json',fps)
    paneldir=ROOT/'review';paneldir.mkdir()
    for page in pages:
        im=Image.open(page['image']).convert('RGB');dr=ImageDraw.Draw(im)
        for i,r in enumerate(page['regions']):dr.rectangle(r['bbox'],outline='blue',width=3);dr.text(tuple(r['bbox'][:2]),f'G{i}',fill='red')
        for r in next(p for p in pred['pages'] if p['id']==page['id'])['regions']:
            dr.rectangle(r['bbox'],outline='green',width=2)
        im.thumbnail((1550,2100));im.save(paneldir/f"{page['id']}.jpg")
    # All FP crops, no inference text shown during first visual classification.
    for k in range(0,len(fps),12):
        canvas=Image.new('RGB',(1400,12*140),'#eee');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(fps[k:k+12]):
            im=Image.open(r['image']).convert('RGB');im.thumbnail((1170,125));canvas.paste(im,(220,j*140+5));d.text((5,j*140+8),f"{k+j:03} {r['page_id']}\n{r['region_id']}",fill='black')
        canvas.save(paneldir/f'fp-{k//12:02d}.jpg')
    print(json.dumps(dict(detection=result['detection'],metrics=result['end_to_end'],pages=[(p['id'],len(p['regions'])) for p in pages])))

def make_scope(pages,holds,additions):
    out=copy.deepcopy(pages);changes=[]
    for p in out:
        for i,r in enumerate(p['regions']):
            reason=holds.get(p['id'],{}).get(i)
            include=r.get('type_suggestion')=='text' and not r.get('hold_reasons') and not reason
            r.update(image_id=p['id'],annotation_id=r['id'],source_gt_ids=[r['id']],include=include,
                     region_type=r.get('type_suggestion','unclassified'),ignore_reason=reason or ('non-general-text formula/symbol' if not include else None),review_status='AI_visual_provisional_no_human_review',uncertainty=reason,geometry_provenance='original_V2',change_evidence=reason or 'original page visually compared; transcription preserved')
            if reason:changes.append(dict(image_id=p['id'],annotation_id=r['id'],change='hold_original_unchanged',reason=reason))
        for r in additions:
            if r['image_id']==p['id']:p['regions'].append(copy.deepcopy(r));changes.append(dict(image_id=p['id'],annotation_id=r['annotation_id'],change='add_region',reason=r['change_evidence']))
        # Diagnostic spatial reading order on original canvas, independent of predictions' text.
        # Source GT order is known to have late-added annotations; preserve it as source_order.
        for i,r in enumerate(sorted(p['regions'],key=lambda r:(r['bbox'][1],r['bbox'][0],r['annotation_id']))):
            r['source_order']=r.get('order');r['order']=i
        p['coverage_status']='AI_page_sweep_provisional; unresolved regions explicit; no human completeness certification'
    return out,changes

def score_scope(pages,prediction):
    lookup={p['id']:p for p in prediction['pages']};tp=fp=fn=0;pairs=[];cases=[];failures=[]
    for p in pages:
        refs=sorted([r for r in p['regions'] if r['include']],key=lambda r:r['order']);held=[r for r in p['regions'] if not r['include']]
        pp=lookup.get(p['id'],{});raw=pp.get('regions',[]) if not pp.get('error') else []
        if pp.get('error'):failures.append(p['id'])
        kept=[];ignored=[];mixed=[]
        for r in sorted(raw,key=lambda r:r['order']):
            policy=ignore_state(r['bbox'],[g['bbox'] for g in refs],[g['bbox'] for g in held])
            if policy=='ignore':ignored.append(r['region_id'])
            else:
                kept.append(r)
                if policy=='mixed_keep':mixed.append(r['region_id'])
        matches=match_boxes([r['bbox'] for r in kept],[r['bbox'] for r in refs],.5);tp+=len(matches);fp+=len(kept)-len(matches);fn+=len(refs)-len(matches)
        ref='\n'.join(r['text'] for r in refs);hyp='\n'.join(r['text'] for r in kept);pairs.append((ref,hyp))
        cases.append(dict(page_id=p['id'],reference_regions=len(refs),prediction_regions=len(kept),ignored=ignored,mixed_keep=mixed,
            matches=[dict(prediction_id=p['id']+'/'+kept[i]['region_id'],annotation_id=refs[j]['annotation_id'],iou=s) for i,j,s in matches],
            unmatched_predictions=[p['id']+'/'+r['region_id'] for i,r in enumerate(kept) if i not in {i for i,j,s in matches}],
            missed=[g['annotation_id'] for j,g in enumerate(refs) if j not in {j for i,j,s in matches}],page=summarize([(ref,hyp)])))
    return dict(detection=_prf(tp,fp,fn),page=summarize(pairs),cases=cases,failures=failures,pages=len(pages),ignore_policy='>=.5 of prediction in ignore; ANY intersection with valid GT takes priority; mixed predictions kept',scope='provisional audited general text; formulas/isolated symbols/uncertain spans ignored')

def audit():
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('decisions','configs/ocr_page_audit/visual_decisions.py');d=module_from_spec(spec);spec.loader.exec_module(d)
    verify_lock();c,legacy,pages,pred=inputs();fps=read(ROOT/'frozen-fp201.json');review=[];additions=[]
    assigned={i:cat for cat,indices in d.CATEGORIES.items() for i in indices}
    assert sum(map(len,d.CATEGORIES.values()))==len(assigned)
    for i,r in enumerate(fps):
        p=next(p for p in pages if p['id']==r['page_id']);near=[g for g in p['regions'] if _iou(r['bbox'],g['bbox'])>.1]
        cat=assigned.get(i,'gt_missing');tags=[]
        if i in d.NEIGHBOR_TAGS:tags.append('adjacent_line_or_neighbor_ink')
        if cat=='gt_missing' and near:tags.append('V2_present_but_outside_CTC_or_fragmented')
        if i in [35,36,37,38,39,45,46,47,49,50,52,53,79,82,87,112]:tags.append('printed_stationery_text')
        note={'gt_missing':'Readable normal text visible; no matching legacy199 reference. Image-based transcription or hold, not OCR copied.',
              'split_merge_iou':'Visible text relates to existing GT but annotation granularity/crop/IoU prevents extra one-to-one TP.',
              'excluded_region':'Formula/tensor/table digit/standalone connector or canceled text; outside general-text scope.',
              'duplicate':'Overlapping prediction includes already represented text; partial duplicate with split/merge ambiguity tagged.',
              'actual_nontext':'Crosshatched non-text scribble, no legible prose.',
              'uncertain':'Visible text/overwriting but reliable literal transcription or granularity not established; independent review needed.'}[cat]
        if cat=='duplicate':tags.append('partial_overlap_not_pure_whole_line_duplicate')
        # Freeze review for every FP, including unresolved.
        rr=copy.deepcopy(r);rr.update(primary=cat,tags=tags,original_gt_links=[g['id'] for g in near],reason=note,uncertainty=cat in ['uncertain','duplicate'],review_status='AI_visual_reviewed; no_independent_human',visual_evidence=f'review/fp-{i//12:02d}.jpg row{i%12}',new_match=None,index=i)
        review.append(rr)
        # Add missing regions plus ignore masks for newly observed out-of-scope/unreadable ink.
        if cat=='split_merge_iou' or (near and cat not in ['uncertain','actual_nontext']):continue
        include=cat=='gt_missing' and i in d.TRANSCRIPTS
        text=d.TRANSCRIPTS.get(i,'') if include else ''
        additions.append(dict(image_id=p['id'],annotation_id=f"audit-{p['id']}-{i:03d}",id=f"audit-{p['id']}-{i:03d}",bbox=r['bbox'],text=text,order=0,include=include,region_type='text' if include else 'nontext' if cat=='actual_nontext' else 'uncertain' if cat in ['uncertain','gt_missing'] else 'formula_symbol_or_canceled',ignore_reason=None if include else note,source_gt_ids=[],source_prediction_geometry=r['prediction_id'],geometry_provenance='detector_assisted_box_visually_compared_to_original; not_independent_geometry',review_status='AI_visual_provisional_no_human_review',uncertainty=None if include else note,change_evidence=f'original page + contact sheet FP index{i}; independently image-read text; no prediction transcription copied'))
    # NO.+year header merged in prediction but missing year in source; explicitly transcribe image.
    for idx in [36,47]:
        r=fps[idx];additions.append(dict(image_id=r['page_id'],annotation_id=f'audit-header-{idx}',id=f'audit-header-{idx}',bbox=r['bbox'],text='NO. year month day ( )',order=0,include=True,region_type='text',ignore_reason=None,source_gt_ids=[],source_prediction_geometry=r['prediction_id'],geometry_provenance='detector_assisted',review_status='AI_visual_provisional_no_human_review',uncertainty='NO overlaps separate small detector box',change_evidence='printed header visible in original; combined header granularity'))
    new,changes=make_scope(pages,d.HOLD_SOURCE,additions)
    # Avoid counting same NO label twice: combined header replaces small added NO region.
    for p in new:
        for g in p['regions']:
            if g['annotation_id'] in ['audit-hf-0013-035','audit-hf-0048-046']:
                g.update(include=False,ignore_reason='covered by combined header reference',uncertainty='overlapping header fragments')
    result=score_scope(new,pred);newmatches={m['prediction_id']:m for p in result['cases'] for m in p['matches']}
    for rr in review:rr['new_match']=newmatches.get(rr['prediction_id'])
    validate_reviews([r['prediction_id'] for r in fps],review)
    write_new(ROOT/'fp201-audit.json',review);write_new(ROOT/'audited-gt-v1.json',dict(policy='all readable general text incl printed stationery and flowchart prose/code labels; formulas, tensor-only shapes, table cells, isolated symbols, canceled/unreadable spans ignored; no OCR-based correctness filtering',pages=new))
    write_new(ROOT/'gt-changes.json',changes);write_new(ROOT/'page-review-notes.json',d.PAGE_NOTES);write_new(ROOT/'audited-scope-metrics.json',result)
    write_new(ROOT/'audit-freeze.json',dict(primary_counts=dict(Counter(r['primary'] for r in review)),reviewed_fp=201,reviewed_original_pages=17,human_reviewed=0,new_gt_sha256=sha256(ROOT/'audited-gt-v1.json'),decisions_sha256=sha256('configs/ocr_page_audit/visual_decisions.py'),review_sha256=sha256(ROOT/'fp201-audit.json'),note='frozen before new crop inference; scope partial provisional, not human certified; additions detector-assisted and cannot prove independent detector recall'))
    fields=['prediction_id','page_id','bbox','text','confidence','legacy_match','new_match','primary','tags','reason','uncertainty','review_status','original_gt_links','visual_evidence']
    with (ROOT/'fp201-audit.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(list,dict)) else r[k] for k in fields} for r in review)
    # gallery uses relative paths to original detector crops; overlay and source links included.
    cards=[]
    for rr in review:
        import os
        link=os.path.relpath(rr['image'],ROOT)
        cards.append(f'<article id="{rr["index"]}"><h3>{rr["index"]}: {rr["prediction_id"]} — {rr["primary"]}</h3><a href="review/{rr["page_id"]}.jpg">原본 overlay</a><img src="{link}"><pre>{html.escape(json.dumps({k:rr[k] for k in fields},ensure_ascii=False,indent=2))}</pre></article>')
    (ROOT/'fp201-gallery.html').write_text('<meta charset="utf-8"><title>AI visual audit FP201</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}article{border:1px solid #aaa;padding:12px;margin:12px}img{display:block;max-width:100%;max-height:200px}pre{white-space:pre-wrap}</style><h1>201/201 AI review; no human certification</h1>'+''.join(cards))
    print(json.dumps(dict(counts=Counter(r['primary'] for r in review),detection=result['detection'],page=result['page']),ensure_ascii=False))

def revise_order(pages,groups):
    out=copy.deepcopy(pages)
    for page in out:
        refs={r['annotation_id']:r for r in page['regions']}
        for ids in groups.get(page['id'],[]):
            slots=sorted(refs[id_]['order'] for id_ in ids)
            for id_,slot in zip(ids,slots):refs[id_]['order']=slot
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','audit'],required=True);args=ap.parse_args()
    {'prepare':prepare,'audit':audit}[args.stage]()
