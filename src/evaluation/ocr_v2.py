"""Saved-prediction evaluation: detection, GT-crop recognition, full-page OCR."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
from src.evaluation.evaluate import _iou, _prf
from src.evaluation.ocr_metrics import normalize, error_counts, NORMALIZATION
from src.ocr_data.audit import valid_bbox


def match_boxes(predictions, references, threshold=.5):
    """Maximum-cardinality bipartite matching; descending IoU tie preference.

    Augmenting paths maximize TP at the threshold (unlike greedy matching).
    Does not claim maximum total IoU among equal-cardinality matchings.
    """
    if not 0 < threshold <= 1:raise ValueError('IoU threshold must be in (0,1]')
    edges={i:sorted((j for j,g in enumerate(references) if _iou(p,g)>=threshold),
                    key=lambda j:(-_iou(p,references[j]),j)) for i,p in enumerate(predictions)}
    owners={}
    def augment(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owners or augment(owners[j],seen):owners[j]=i;return True
        return False
    for i in sorted(edges,key=lambda x:(len(edges[x]),x)):augment(i,set())
    return sorted((i,j,_iou(predictions[i],references[j])) for j,i in owners.items())


def summarize(pairs):
    if not pairs:return dict(samples=0,cer_micro=None,cer_no_space_micro=None,exact_match_rate=None,substitutions=0,deletions=0,insertions=0,reference_units=0,errors=0)
    normalized=[(normalize(a),normalize(b)) for a,b in pairs]
    c=[error_counts(a,b) for a,b in normalized]
    compact=[error_counts(''.join(a.split()),''.join(b.split())) for a,b in normalized]
    counts={k:sum(x[k] for x in c) for k in c[0]}
    units=counts['reference_units'];units2=sum(x['reference_units'] for x in compact)
    return dict(samples=len(pairs),**counts,cer_micro=counts['errors']/units if units else None,
                cer_no_space_micro=sum(x['errors'] for x in compact)/units2 if units2 else None,
                exact_match_rate=sum(a==b for a,b in normalized)/len(pairs),
                empty_reference_insertions=sum(x['insertions'] for x in c if x['reference_units']==0))


def unique(rows, key):
    result={}
    for row in rows:
        k=key(row)
        if k in result:raise ValueError(f'duplicate prediction ID: {k}')
        result[k]=row
    return result


def evaluate(pages, prediction, threshold=.5):
    mode=prediction.get('mode')
    if mode not in ('page','recognition_only'):raise ValueError('mode must be page or recognition_only; crop detection is not recognition-only')
    page_ids={p['id'] for p in pages}
    if len(page_ids)!=len(pages):raise ValueError('duplicate GT page ID')
    result=dict(mode=mode,normalization=NORMALIZATION,iou_threshold=threshold,
                matching='maximum-cardinality bipartite; descending IoU neighbor order; deterministic ties',
                exclusion_policy='held pages skipped; otherwise ignore predictions >=50% inside held GT regions unless also >=50% inside eligible GT; ambiguous overlaps fail closed',
                ignored_predictions=0,excluded_pages=[],inference_failures=[],missing_predictions=[],empty_results=[],
                no_text_false_positive_pages=0,cases=[],type_metrics_are='rule-suggested strata, not confirmed semantic labels')
    strata=defaultdict(list);tp=fp=fn=0;page_pairs=[]
    if mode=='recognition_only':
        crops=unique(prediction.get('crops',[]),lambda x:(x['page_id'],x['region_id']))
        known={(p['id'],r['id']) for p in pages for r in p['regions']}
        if set(crops)-known:raise ValueError('unknown crop predictions')
    else:
        predicted=unique(prediction.get('pages',[]),lambda x:x['id'])
        if set(predicted)-page_ids:raise ValueError('unknown/out-of-split page predictions')
    for page in pages:
        id_=page['id']
        if not page.get('eligible',False):result['excluded_pages'].append(id_);continue
        refs=sorted([r for r in page['regions'] if not r['hold_reasons']],key=lambda r:(r['order'],r['id']))
        ignored=[r for r in page['regions'] if r['hold_reasons']]
        if mode=='recognition_only':
            for r in refs:
                key=(id_,r['id']);p=crops.get(key,{})
                if not p:result['missing_predictions'].append(list(key))
                if p.get('error'):result['inference_failures'].append(dict(page_id=id_,region_id=r['id'],error=p['error']))
                text='' if p.get('error') else p.get('text','')
                if not isinstance(text,str):raise ValueError('prediction text must be a string')
                strata['all'].append((r['text'],text));strata[r.get('type_confirmed') or r.get('type_suggestion','unclassified')].append((r['text'],text))
            continue
        p=predicted.get(id_,{})
        if not p:result['missing_predictions'].append(id_)
        if p.get('error'):result['inference_failures'].append(dict(page_id=id_,error=p['error']))
        raw=[] if p.get('error') else p.get('regions',[])
        if not raw:result['empty_results'].append(id_)
        preds=[]
        def inside(a,b):
            inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            return inter/((a[2]-a[0])*(a[3]-a[1]))
        for region in raw:
            if not valid_bbox(region.get('bbox'),page.get('width',float('inf')),page.get('height',float('inf'))):raise ValueError(f'invalid predicted original-pixel bbox on {id_}')
            if not isinstance(region.get('text'),str):raise ValueError('prediction text must be a string')
            in_held=any(inside(region['bbox'],g['bbox'])>=.5 for g in ignored)
            in_valid=any(inside(region['bbox'],g['bbox'])>=.5 for g in refs)
            if in_held and in_valid:raise ValueError('prediction overlaps held and valid GT; hold page or mask at inference')
            if in_held:result['ignored_predictions']+=1
            else:preds.append(region)
        if preds and any('order' in r for r in preds):
            if not all(type(r.get('order')) is int for r in preds) or len({r['order'] for r in preds})!=len(preds):raise ValueError('invalid prediction reading order')
            preds=sorted(preds,key=lambda r:r['order'])
        matches=match_boxes([r['bbox'] for r in preds],[r['bbox'] for r in refs],threshold)
        usedp={i for i,j,s in matches};usedg={j for i,j,s in matches}
        tp+=len(matches);fp+=len(preds)-len(matches);fn+=len(refs)-len(matches)
        page_pairs.append(('\n'.join(r['text'] for r in refs),'\n'.join(r['text'] for r in preds)))
        if page.get('no_text') and preds:result['no_text_false_positive_pages']+=1
        # Overlap adjacency records split/merge candidates even when no box reaches IoU .5.
        splits=[dict(reference=r['id'],predictions=[i for i,p in enumerate(preds) if _iou(p['bbox'],r['bbox'])>.1]) for r in refs]
        merges=[dict(prediction=i,references=[r['id'] for r in refs if _iou(p['bbox'],r['bbox'])>.1]) for i,p in enumerate(preds)]
        ordered=[j for i,j,s in sorted(matches)]
        result['cases'].append(dict(id=id_,page_text=summarize([page_pairs[-1]]),matches=[dict(prediction=i,reference=refs[j]['id'],iou=s) for i,j,s in matches],
            missed=[refs[j]['id'] for j in range(len(refs)) if j not in usedg],false_positives=[i for i in range(len(preds)) if i not in usedp],
            split_candidates=[x for x in splits if len(x['predictions'])>1],merge_candidates=[x for x in merges if len(x['references'])>1],
            reading_order_inversions=sum(a>b for i,a in enumerate(ordered) for b in ordered[i+1:])))
    result['evaluated_pages']=len(pages)-len(result['excluded_pages'])
    if mode=='recognition_only':result['recognition']={k:summarize(v) for k,v in sorted(strata.items())};result['recognition'].setdefault('all',summarize([]))
    else:result['detection']=_prf(tp,fp,fn);result['end_to_end']=summarize(page_pairs)
    if mode=='recognition_only':
        for kind,metrics in result['recognition'].items():
            metrics['line_exact_match_rate' if kind=='text' else 'region_exact_match_rate']=metrics['exact_match_rate']
    return result


def evaluate_dataset(dataset, prediction, split, threshold=.5):
    """Bind saved predictions to exact dataset bytes before computing metrics."""
    from src.ocr_data.audit import sha256
    root=Path(dataset)
    if prediction.get('dataset_manifest_sha256')!=sha256(root/'manifest.json'):
        raise ValueError('prediction dataset fingerprint missing or mismatched')
    manifest=json.loads((root/'manifest.json').read_text())
    if sha256(root/'pages.jsonl')!=manifest['artifacts_sha256']['pages.jsonl']:
        raise ValueError('dataset pages hash mismatch')
    pages=[json.loads(x) for x in (root/'pages.jsonl').read_text().splitlines() if x.strip()]
    pages=[x for x in pages if x['split']==split]
    result=evaluate(pages,prediction,threshold)
    result.update(split=split,dataset_manifest_sha256=sha256(root/'manifest.json'),dataset_sha256=sha256(root/'pages.jsonl'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--split',choices=['train','validation','test'],required=True);p.add_argument('--iou',type=float,default=.5)
    a=p.parse_args()
    from src.ocr_data.audit import sha256
    prediction=json.loads(Path(a.predictions).read_text())
    result=evaluate_dataset(a.dataset,prediction,a.split,a.iou)
    result['prediction_sha256']=sha256(a.predictions)
    with Path(a.output).open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2);out.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases',)},ensure_ascii=False))
if __name__=='__main__':main()
