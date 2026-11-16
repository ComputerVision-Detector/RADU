"""Summarize actual fixed-model audit; all result writes are exclusive."""
import copy
import csv
import json
from collections import Counter
from pathlib import Path
from src.evaluation.ocr_page_audit import *
from src.evaluation.ocr_metrics import error_counts


def summarize_results(output=None):
    out=Path(output) if output else ROOT/'summary';out.mkdir(parents=True,exist_ok=False)
    verify_lock();c,legacy,source,pred=inputs();new=read(ROOT/'audited-gt-v2.json')['pages'];m=read(ROOT/'M.json');p1=read(ROOT/'P1-rows.json')
    orders={(p['id'],g['annotation_id']):g['order'] for p in new for g in p['regions']}
    for r in m:r['gt_order']=orders[r['page_id'],r['region_id']]
    inferdir=ROOT/'inference'
    def texts(file,rows):
        d=read(inferdir/file)
        if not d['complete'] or d['failures']:raise ValueError('incomplete inference')
        if [(r['page_id'],r['region_id']) for r in rows]!=[(r['page_id'],r['region_id']) for r in d['predictions']]:raise ValueError('prediction row drift')
        return [r['text'] for r in d['predictions']]
    c0=texts('C0.json',m);c1=texts('C1.json',m);p1texts=texts('P1.json',p1)
    crop=crop_comparison(m,c0,c1);cropcases=[]
    for r,t0,t1 in zip(m,c0,c1):
        a=summarize([(r['text'],t0)]);b=summarize([(r['text'],t1)]);cropcases.append(dict(**r,C0=t0,C1=t1,C0_errors=a['errors'],C1_errors=b['errors'],delta=b['errors']-a['errors'],gt_area=(r['bbox'][2]-r['bbox'][0])*(r['bbox'][3]-r['bbox'][1]),pred_area=(r['pred_bbox'][2]-r['pred_bbox'][0])*(r['pred_bbox'][3]-r['pred_bbox'][1])))
    crop['changes']=dict(improved_C1=sum(x['delta']<0 for x in cropcases),worsened_C1=sum(x['delta']>0 for x in cropcases),unchanged_errors=sum(x['delta']==0 for x in cropcases))
    crop['paired_batch_context_vs_cached_C1']=dict(changed_strings=sum(t!=r['cached_C1'] for r,t in zip(m,c1)),regions=len(m),note='same model/settings but paired batch membership differs from original397 batch; crop comparison uses new pairedC0/C1 exclusively')
    order=[];o0pairs=[];o1pairs=[]
    for page in new:
        rows=sorted([r for r in m if r['page_id']==page['id']],key=lambda r:r['pred_order'])
        if not rows:continue
        rows=[dict(prediction_id=r['prediction_id'],text=r['cached_C1'],gt_order=r['gt_order'],reference=r['text']) for r in rows]
        reference=[r['reference'] for r in sorted(rows,key=lambda r:r['gt_order'])]
        q=order_diagnostic(rows,reference);q['page_id']=page['id'];order.append(q)
        o0pairs.append(('\n'.join(reference),'\n'.join(r['text'] for r in rows)));o1pairs.append(('\n'.join(reference),'\n'.join(r['text'] for r in sorted(rows,key=lambda r:r['gt_order']))))
    O=dict(pages=len(order),regions=sum(x['regions'] for x in order),inversions=sum(x['inversions'] for x in order),possible_pairs=sum(x['possible_pairs'] for x in order),O0=summarize(o0pairs),O1=summarize(o1pairs),cases=order,scope='M-only original savedC1 strings; unmatched/duplicate/uncertain/split-merge/detector-assisted regions excluded; not full page CER',order_method='AI-reviewed spatial GT with explicit left-right captions/headers in v2; formula/flowchart semantic order not certified; no text similarity optimization')
    O['order_error_rate']=O['inversions']/O['possible_pairs'] if O['possible_pairs'] else None
    # P1 assembled from all included GT with all reference failures preserved by infer assertions.
    p1lookup={(r['page_id'],r['region_id']):t for r,t in zip(p1,p1texts)};P1pairs=[]
    for page in new:
        refs=sorted([g for g in page['regions'] if g['include']],key=lambda g:g['order'])
        P1pairs.append(('\n'.join(g['text'] for g in refs),'\n'.join(p1lookup[page['id'],g['annotation_id']] for g in refs)))
    broadraw=copy.deepcopy(new)
    for p in broadraw:p['regions']=[g for g in p['regions'] if g['include']]
    raw=score_scope(broadraw,pred);scored=read(ROOT/'audited-scope-metrics-v2.json')
    # Full change trace compares original V2 fields and legacy inclusion for every region.
    src={(p['id'],r['id']):r for p in source for r in p['regions']};leg={(p['id'],r['id']):not r['hold_reasons'] for p in legacy for r in p['regions']};trace=[]
    for p in new:
        for g in p['regions']:
            before=src.get((p['id'],g['annotation_id']));fields={}
            if before:
                for field in ['bbox','text','order']:
                    if before[field]!=g[field]:fields[field]=dict(before=before[field],after=g[field])
                if leg[p['id'],g['annotation_id']]!=g['include']:fields['legacy_inclusion']=dict(before=leg[p['id'],g['annotation_id']],after=g['include'])
            else:fields['added_region']=g
            if fields:trace.append(dict(page_id=p['id'],annotation_id=g['annotation_id'],changes=fields,evidence=g['change_evidence'],review_status=g['review_status']))
    write_new(out/'gt-full-change-trace.json',trace)
    # Legacy235 recognition replay from original stored predictions, no new model inference.
    from src.training.ocr_finetune import readrows
    rows=readrows(DATA/'recognition_candidates.test.jsonl');saved=read('data/experiments/ocr_v2_vl_lora/final/after.test.predictions.json')['predictions'];lookup={(r['page_id'],r['region_id']):r['text'] for r in saved}
    rec=summarize([(r['text'],lookup[r['page_id'],r['region_id']]) for r in rows]);assert rec['errors']==670 and rec['reference_units']==4401
    # Restore exact eligible region sets for basis counts and FN diagnosis.
    newvalid={(p['id'],g['annotation_id']) for p in new for g in p['regions'] if g['include']};sourcevalid={(p['id'],g['id']) for p in source for g in p['regions'] if g['type_suggestion']=='text'}
    legacyvalid={k for k,v in leg.items() if v};restored=newvalid & sourcevalid - legacyvalid
    missed=[(p['page_id'],id_) for p in scored['cases'] for id_ in p['missed']]
    upstream=read('data/experiments/ocr_v2_korean/det-test/detections.json')
    scope=dict(original_pages=17,original_V2_regions=len(src),legacy_GT=len(legacyvalid),VL_crop_GT=len(rows),audited_GT=len(newvalid),audited_original_GT=len(newvalid&set(src)),added_included_GT=len(newvalid-set(src)),restored_CTC_excluded_normal_GT=len(restored),ignored_or_uncertain_GT=sum(not g['include'] for p in new for g in p['regions']),new_FN=len(missed),new_FN_restored_CTC=sum(k in restored for k in missed),new_FN_other=len(missed)-sum(k in restored for k in missed),upstream_raw_detections=sum(p['raw_detections'] for p in upstream['pages']),upstream_already_ignored=sum(p['ignored'] for p in upstream['pages']),fixed_predictions=397,post_audit_ignored=sum(len(p['ignored']) for p in scored['cases']),mixed_predictions_kept=sum(len(p['mixed_keep']) for p in scored['cases']),coverage='provisional AI scope, no independent human; added boxes detection-assisted; raw103 dropped geometry unavailable so FN not pure detector miss')
    result=dict(legacy=read(ROOT/'legacy-reproduced.json'),legacy_crop235=rec,scope=scope,FP=read(ROOT/'review-revision-v2.json')['counts'],new_GT_no_new_ignore=raw,new_GT_with_ignore=scored,crop=crop,order=O,P0=scored['page'],P1=summarize(P1pairs),P1_recognition=summarize([(r['text'],t) for r,t in zip(p1,p1texts)]),inference=read(inferdir/'receipt.json'),repeat=read(inferdir/'reproducibility.json'),note='all score changes diagnostic; same adapter, no model improvement or additive cause shares claimed')
    write_new(out/'results.json',result);write_new(out/'crop-cases.json',cropcases)
    with (out/'comparisons.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=['condition','pages','samples','reference_units','CER','no_space_CER','exact','S','D','I']);w.writeheader()
        for name,metrics,np in [('legacy_page',result['legacy']['end_to_end'],17),('new_GT_raw397',raw['page'],17),('P0_audited_scope',result['P0'],17),('P1_GT_boxes_order',result['P1'],17),('C0_M_GT_crop',crop['C0'],crop['pages']),('C1_M_detected_crop',crop['C1'],crop['pages']),('O0_M_original_order',O['O0'],O['pages']),('O1_M_GT_order',O['O1'],O['pages'])]:
            w.writerow(dict(condition=name,pages=np,samples=metrics['samples'],reference_units=metrics['reference_units'],CER=metrics['cer_micro'],no_space_CER=metrics['cer_no_space_micro'],exact=metrics['exact_match_rate'],S=metrics['substitutions'],D=metrics['deletions'],I=metrics['insertions']))
    write_new(out/'artifact-hashes.json',{str(p.relative_to(ROOT)):sha256(p) for p in sorted(ROOT.rglob('*')) if p.is_file()})
    print(json.dumps({k:result[k] for k in ['scope','crop','P0','P1']},ensure_ascii=False));print('Order:',json.dumps({k:v for k,v in O.items() if k!='cases'},ensure_ascii=False))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args();summarize_results(args.output)
