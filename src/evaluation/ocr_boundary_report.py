"""Summarize actual unselected crop method and independent GT geometry sensitivity."""
import csv
import copy
import html
import json
import os
from pathlib import Path
from collections import Counter
from src.evaluation.ocr_boundary import ROOT,SCOPE,read,write_new,verify,sha256,paired_metrics,select_method
from src.evaluation.ocr_scope_recovery_report import bootstrap
from src.evaluation.ocr_page_audit import score_scope
from src.evaluation.ocr_metrics import normalize


def main():
    verify();out=ROOT/'summary';out.mkdir(exist_ok=False);allcases=read(ROOT/'crop-error-cases.json');splitresults={}
    for split in ['validation','diagnostic']:
        rows=read(ROOT/f'{split}.before.rows.json');before=read(ROOT/split/'before.json');after=read(ROOT/split/'after.json')
        if before['failures'] or after['failures'] or not before['complete'] or not after['complete']:raise ValueError('inference incomplete')
        m=paired_metrics(rows,before['predictions'],after['predictions'])
        if m!=read(ROOT/split/'comparison.json'):raise ValueError('saved metrics drift')
        pages=sorted({r['page_id'] for r in rows});bc=[];ac=[]
        for p in pages:
            cases=[r for r in allcases[split] if r['page_id']==p];chars=sum(len(normalize(r['reference'])) for r in cases)
            bc.append(dict(page_id=p,page=dict(errors=sum(r['before_errors'] for r in cases),reference_units=chars)));ac.append(dict(page_id=p,page=dict(errors=sum(r['after_errors'] for r in cases),reference_units=chars)))
        cases=allcases[split];m.update(bootstrap=bootstrap(dict(cases=bc),dict(cases=ac)),changes=dict(Counter('improved' if x['delta']<0 else 'worsened' if x['delta']>0 else 'same_errors' for x in cases)),changed_crops=sum(r['changed'] for r in cases),unchanged_crop_output_changes=sum(not r['changed'] and r['before']!=r['after'] for r in cases),runtime=read(ROOT/split/'receipt.json'))
        splitresults[split]=m
    selection=read(ROOT/'validation/selection-lock.json')
    if selection['chosen']!=select_method(splitresults['validation']['before'],splitresults['validation']['after']):raise ValueError('validation selection drift')
    gt=read(ROOT/'geometry-comparison.json');recomputed=score_scope(read(ROOT/'audited-gt-v4.json')['pages'],read(SCOPE/'recognition/full500.predictions.json'))
    if recomputed!=gt['after']:raise ValueError('GT geometry score drift')
    old_gt=read(SCOPE/'audited-gt-v3.json');new_gt=read(ROOT/'audited-gt-v4.json')
    for p,q in zip(old_gt['pages'],new_gt['pages']):
        for a,b in zip(p['regions'],q['regions']):
            if any(a[k]!=b[k] for k in ['annotation_id','text','include','order']):raise ValueError('GT scope/text/order changed')
    result=dict(crops=splitresults,selection=selection,GT_geometry=gt,dark_background_failure=read(ROOT/'dark-background-failure.json'),human_review_count=0,independent_test=False,policy_sha256=sha256('configs/ocr_boundary.json'),no_training=True,applied_to_operating_pipeline=False)
    write_new(out/'results.json',result)
    with (out/'comparisons.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=['experiment','condition','pages','regions','reference_units','CER','no_space_CER','exact','S','D','I']);w.writeheader()
        for split,m in splitresults.items():
            for condition in ['before','after']:
                x=m[condition];w.writerow(dict(experiment=split+'_crop',condition=condition,pages=m['pages'],regions=m['regions'],reference_units=x['reference_units'],CER=x['cer_micro'],no_space_CER=x['cer_no_space_micro'],exact=x['exact_match_rate'],S=x['substitutions'],D=x['deletions'],I=x['insertions']))
        for name,r in [('GTv3',gt['before']),('GTv4_bbox_only',gt['after'])]:
            x=r['page'];w.writerow(dict(experiment='same_raw500_GT_geometry',condition=name,pages=17,regions=gt['GT_regions'],reference_units=x['reference_units'],CER=x['cer_micro'],no_space_CER=x['cer_no_space_micro'],exact=x['exact_match_rate'],S=x['substitutions'],D=x['deletions'],I=x['insertions']))
    # Full geometry review view, including held and unchanged proposals.
    decisions=read(ROOT/'geometry-review.json')['decisions'];manual={r['index']:r for r in read(ROOT/'manual-boundary-proposals.json')};cards=[]
    for r in decisions:
        i=r['proposal_index'];new=f'review/manual-{i:03d}.png' if i in manual else f'review/geometry-{i:03d}-new.png'
        cards.append(f'<article><h3>{i} {r["page_id"]} {r["annotation_id"]} — {r["decision"]}</h3><div><img src="review/geometry-{i:03d}-old.png"><img src="{new}"></div><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre></article>')
    with (ROOT/'geometry-review.html').open('x') as f:f.write('<meta charset="utf-8"><title>GT boundary23 AI review</title><style>body{font:15px sans-serif;max-width:1500px;margin:auto}article{border:1px solid #aaa;padding:10px;margin:10px}img{max-width:49%;max-height:250px}pre{white-space:pre-wrap}</style><h1>GT boundary23: AI provisional; independent human0</h1><p>Left old, right reviewed proposal. Holds preserve existing annotation/inclusion. Raw recognition identical.</p>'+''.join(cards))
    examples=read(ROOT/'visual-error-examples.json');ec=[]
    for r in examples:
        ec.append(f'<article><h3>{r["page_id"]} {r["region_id"]}: {r["before_errors"]} → {r["after_errors"]} errors</h3><img src="{os.path.relpath(r["original"],ROOT)}"><img src="{os.path.relpath(r["trimmed"],ROOT)}"><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre></article>')
    with (ROOT/'crop-errors.html').open('x') as f:f.write('<meta charset="utf-8"><style>body{font:15px sans-serif;max-width:1500px;margin:auto}article{border:1px solid #aaa;padding:10px}img{max-width:49%}pre{white-space:pre-wrap}</style><h1>Unselected image trim: failure and improvement cases</h1>'+''.join(ec))
    print(json.dumps(dict(chosen=selection['chosen'],validation=splitresults['validation']['before']['cer_micro'],validation_trim=splitresults['validation']['after']['cer_micro'],diagnostic=splitresults['diagnostic']['before']['cer_micro'],diagnostic_trim=splitresults['diagnostic']['after']['cer_micro'],bootstrap={k:v['bootstrap'] for k,v in splitresults.items()}),ensure_ascii=False))

if __name__=='__main__':main()
