"""Report fixed-cache scope recovery; never select thresholds or recognition outputs."""
import csv
import copy
import html
import json
import os
from collections import Counter
from pathlib import Path
import random
from src.evaluation.ocr_scope_recovery import ROOT,PRIOR,OLD,read,write_new,verify,keyed,paired_scope_compare,verify_review_freeze
from src.evaluation.ocr_page_audit import sha256,intersection,ignore_state


def bootstrap(before,after):
    b={p['page_id']:p for p in before['cases']};a={p['page_id']:p for p in after['cases']};ids=sorted(b);rng=random.Random(42);values=[]
    for _ in range(2000):
        keys=rng.choices(ids,k=len(ids));n=sum(b[k]['page']['reference_units'] for k in keys)
        if n:values.append(sum(b[k]['page']['errors']-a[k]['page']['errors'] for k in keys)/n)
    values.sort();return dict(replicates=2000,seed=42,page_count=len(ids),improvement95=[values[int(.025*len(values))],values[min(len(values)-1,int(.975*len(values)))]],note='diagnostic-set page bootstrap, not independent generalization')


def main():
    verify();verify_review_freeze();out=ROOT/'summary';out.mkdir(exist_ok=False);before=read(OLD);after=read(ROOT/'recognition/full500.predictions.json');gt2=read(ROOT/'prior-gt-v2.json');gt3=read(ROOT/'audited-gt-v3.json')
    primary=paired_scope_compare(gt2['pages'],before,after);secondary=paired_scope_compare(gt3['pages'],before,after)
    legacy=read(PRIOR/'summary/results.json')['P0']
    if primary['before']['page']!=legacy:raise ValueError('prior GTv2 baseline not reproduced')
    primary['bootstrap']=bootstrap(primary['before'],primary['after']);secondary['bootstrap']=bootstrap(secondary['before'],secondary['after'])
    extra=read(ROOT/'extra103-rows.json');geom=read(ROOT/'restored103-geometry-scope.json');full=keyed(after)
    visual={r['prediction_id']:r for r in read('configs/ocr_page_audit/scope_restored_visual.json')['items']}
    if set(visual)!={r['key'] for r in extra}:raise ValueError('visual review ID drift')
    categories=[]
    matched={m['prediction_id']:m for p in primary['after']['cases'] for m in p['matches']};old_missed={p['page_id']:set(p['missed']) for p in primary['before']['cases']};new_missed={p['page_id']:set(p['missed']) for p in primary['after']['cases']}
    recovered_refs=[dict(page_id=p,annotation_id=g) for p in old_missed for g in sorted(old_missed[p]-new_missed[p])]
    for i,r in enumerate(extra):
        if r['crop_sha256']!=visual[r['key']]['crop_sha256']:raise ValueError('visual crop hash drift')
        category=visual[r['key']]['visual_category']
        row=dict(**geom[i],visual_category=category,AI_visual_reviewed=True,independent_human_review=False,index=i,text=full[r['key']]['text'],new_match=matched.get(r['key']),image=r['image'],note='all103 visually inspected in9sheets before extra recognition; geometry policy fixed')
        categories.append(row)
    write_new(ROOT/'restored103-audit.json',categories)
    # Why old CTC screening removed NORMAL references, independent of new predictions.
    held=read('data/experiments/ocr_v2_korean/prepared/held.json');heldmap={(r['page_id'],r['region_id']):r for r in held if r['split']=='test'}
    sourcev2={(p['id'],r['annotation_id']):r for p in gt2['pages'] for r in p['regions']};lost=[]
    for key,g in sourcev2.items():
        if g['include'] and key in heldmap:lost.append(dict(page_id=key[0],annotation_id=key[1],text=g['text'],reasons=heldmap[key]['reasons'],unsupported=heldmap[key]['unsupported'],matched_after=g['annotation_id'] not in new_missed[key[0]]))
    write_new(ROOT/'CTC-normal-text-exclusions.json',lost)
    perpage=[]
    for bp,ap in zip(primary['before']['cases'],primary['after']['cases']):
        assert bp['page_id']==ap['page_id'];perpage.append(dict(page_id=bp['page_id'],reference_units=bp['page']['reference_units'],before=bp['page'],after=ap['page'],errors_delta=ap['page']['errors']-bp['page']['errors']))
    result=dict(primary_GT2=primary,secondary_GT3=secondary,GT_versions=dict(v2_refs=sum(g['include'] for p in gt2['pages'] for g in p['regions']),v3_refs=sum(g['include'] for p in gt3['pages'] for g in p['regions']),changes=read(ROOT/'gt-v3-changes.json')),detector=read(ROOT/'raw-detector/recovery.json'),recognition=read(ROOT/'recognition/receipt.json'),repeat=read(ROOT/'recognition/repeat-check.json'),visual_counts=dict(Counter(r['visual_category'] for r in categories)),extra_scope_counts=dict(Counter(r['scope_policy'] for r in categories)),CTC_normal_refs=len(lost),CTC_normal_matched_after=sum(r['matched_after'] for r in lost),CTC_unsupported_counts=dict(Counter(c for r in lost for c in r['unsupported'])),recovered_FN_references=recovered_refs,still_unmatched=primary['after']['cases'],perpage=perpage,independent_human_review=False)
    write_new(out/'results.json',result)
    with (out/'comparisons.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=['GT','condition','pages','GT_regions','scored_predictions','chars','CER','CER_no_space','S','D','I','TP','FP','FN','P','R','F1']);w.writeheader()
        for ver,pair in [('v2',primary),('v3',secondary)]:
            for condition,r in [('legacy397',pair['before']),('recovered500',pair['after'])]:
                m=r['page'];d=r['detection'];w.writerow(dict(GT=ver,condition=condition,pages=17,GT_regions=d['tp']+d['fn'],scored_predictions=d['tp']+d['fp'],chars=m['reference_units'],CER=m['cer_micro'],CER_no_space=m['cer_no_space_micro'],S=m['substitutions'],D=m['deletions'],I=m['insertions'],TP=d['tp'],FP=d['fp'],FN=d['fn'],P=d['precision'],R=d['recall'],F1=d['f1']))
    with (ROOT/'restored103-audit.csv').open('x') as f:
        fields=['prediction_id','image_id','bbox','text','visual_category','scope_policy','valid_GT','held_GT','new_match','AI_visual_reviewed','independent_human_review'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(list,dict)) else r[k] for k in fields} for r in categories)
    cards=[]
    for r in categories:
        cards.append(f'<article><h3>{r["index"]} {r["prediction_id"]} — {r["visual_category"]}</h3><img loading="lazy" src="{os.path.relpath(r["image"],ROOT)}"><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre></article>')
    with (ROOT/'restored103-gallery.html').open('x') as f:f.write('<meta charset="utf-8"><title>Restored103 audit</title><style>body{font:15px sans-serif;max-width:1300px;margin:auto}article{border:1px solid #aaa;padding:10px;margin:10px}img{max-width:100%;max-height:220px}pre{white-space:pre-wrap}</style><h1>Restored103: all AI visually reviewed, no human certification</h1>'+''.join(cards))
    print(json.dumps(dict(primary=primary['after']['page'],bootstrap=primary['bootstrap'],visual_counts=result['visual_counts'],extra_scope=result['extra_scope_counts'],CTC_normal=len(lost),matched_after=result['CTC_normal_matched_after']),ensure_ascii=False))
    print('perpage',[(p['page_id'],p['errors_delta']) for p in perpage])

if __name__=='__main__':main()
