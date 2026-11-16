"""Replay image-based review corrections, preserving every v1 artifact.
No score search: left-right caption order and conservative partial-overlap labels.
"""
import copy
import csv
import html
import json
import os
from src.evaluation.ocr_page_audit import ROOT,PRED,read,write_new,revise_order,score_scope,sha256,Counter,intersection,validate_reviews


def main():
    new=read(ROOT/'audited-gt-v1.json');groups={'hf-0033':[['hf-0033-line-0016','hf-0033-line-0017']],'hf-0094':[['hf-0094-line-0001','hf-0094-line-0002']]}
    v2=copy.deepcopy(new);v2['pages']=revise_order(new['pages'],groups);v2['revision']='v2: image-reviewed left-right captions/headers; order only; no text or geometry changes'
    write_new(ROOT/'audited-gt-v2.json',v2)
    write_new(ROOT/'order-policy-revision.json',dict(before_sha256=sha256(ROOT/'audited-gt-v1.json'),after_sha256=sha256(ROOT/'audited-gt-v2.json'),groups=groups,reason='left-to-right same-row caption/header order from original page, not CER optimization',fields_changed=['order'],model_or_transcription_changed=False))
    metric=score_scope(v2['pages'],read(PRED));write_new(ROOT/'audited-scope-metrics-v2.json',metric)
    matches={m['prediction_id']:m for p in metric['cases'] for m in p['matches']};review=read(ROOT/'fp201-audit.json')
    for r in review:
        r['new_match']=matches.get(r['prediction_id'])
        if r['primary']=='duplicate':r.update(primary='split_merge_iou',uncertainty=True,reason='Partial repeated text in larger crop, not confirmed whole-region duplicate; header fragment or int[] fragment. Preserve partial-overlap tag.')
    validate_reviews([r['prediction_id'] for r in read(ROOT/'frozen-fp201.json')],review)
    write_new(ROOT/'fp201-audit-v2.json',review)
    fields=['prediction_id','page_id','bbox','text','confidence','legacy_match','new_match','primary','tags','reason','uncertainty','review_status','original_gt_links','visual_evidence']
    with (ROOT/'fp201-audit-v2.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r[k],(list,dict)) else r[k] for k in fields} for r in review)
    write_new(ROOT/'review-revision-v2.json',dict(changes=[dict(index=i,before='duplicate',after='split_merge_iou',reason='partial repeated text does not establish a whole duplicate') for i in [36,47,176]],counts=dict(Counter(r['primary'] for r in review)),confirmed_whole_region_duplicate=0,partial_duplicate_candidates=3,model_changed=False))
    gallery(v2['pages'],review)


def gallery(pages,review):
    from PIL import Image,ImageDraw
    cards=[]
    for page in pages:
        im=Image.open(page['image']).convert('RGB');draw=ImageDraw.Draw(im)
        for r in page['regions']:
            draw.rectangle(r['bbox'],outline='blue' if r['include'] else 'orange',width=4);draw.text(tuple(r['bbox'][:2]),r['annotation_id'][-12:],fill='red')
        im.thumbnail((1550,2100));path=ROOT/'review'/f"audited-{page['id']}.jpg"
        if path.exists():raise FileExistsError(path)
        im.save(path)
    for r in review:
        page=next(p for p in pages if p['id']==r['page_id']);data=copy.deepcopy(r);data['new_gt_context']=[g for g in page['regions'] if intersection(r['bbox'],g['bbox'])>0]
        cards.append(f'<article data-cat="{r["primary"]}" id="fp{r["index"]}"><h2>{r["index"]}: {r["prediction_id"]} — {r["primary"]}</h2><a href="review/{r["page_id"]}.jpg">Legacy overlay</a> · <a href="review/audited-{r["page_id"]}.jpg">New GT overlay: blue included / orange ignored</a><img loading="lazy" src="{os.path.relpath(r["image"],ROOT)}"><pre>{html.escape(json.dumps(data,ensure_ascii=False,indent=2))}</pre></article>')
    doc='<meta charset="utf-8"><title>FP201 AI audit</title><style>body{font:15px sans-serif;max-width:1250px;margin:auto}article{border:1px solid #aaa;padding:12px;margin:12px}img{max-width:100%;max-height:240px;display:block}pre{white-space:pre-wrap}</style><h1>201/201 AI visual review; no independent human certification</h1>'+''.join(cards)
    with (ROOT/'review.html').open('x') as f:f.write(doc)

if __name__=='__main__':main()
