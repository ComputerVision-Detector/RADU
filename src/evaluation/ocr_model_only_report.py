"""Summarize frozen recognizer-only runs; no GT/model changes and no reselection."""
import argparse,collections,csv,json,random,time,re,unicodedata
from pathlib import Path
from src.training.ocr_model_only import read,write_new,verify,language,score
from src.evaluation.ocr_v2 import summarize

def compare(rows,parent,chosen):
    a={(r['page_id'],r['region_id']):r['text'] for r in parent};b={(r['page_id'],r['region_id']):r['text'] for r in chosen}
    assert set(a)==set(b)=={(r['page_id'],r['region_id']) for r in rows}
    cases=[]
    for r in rows:
        k=r['page_id'],r['region_id'];before=summarize([(r['text'],a[k])]);after=summarize([(r['text'],b[k])]);cases.append(dict(page_id=k[0],region_id=k[1],reference=r['text'],parent=a[k],chosen=b[k],stratum=language(r['text']),reference_units=before['reference_units'],parent_errors=before['errors'],chosen_errors=after['errors'],delta=after['errors']-before['errors']))
    pages=collections.defaultdict(lambda:[0,0,0])
    for r in cases:
        v=pages[r['page_id']];v[0]+=r['parent_errors'];v[1]+=r['chosen_errors'];v[2]+=r['reference_units']
    values=list(pages.values());rng=random.Random(42);draws=[]
    for _ in range(2000):
        sample=[rng.choice(values) for _ in values];units=sum(x[2] for x in sample);draws.append(sum(x[0]-x[1] for x in sample)/units)
    draws.sort()
    return dict(cases=cases,improved=sum(x['delta']<0 for x in cases),worsened=sum(x['delta']>0 for x in cases),same_errors=sum(x['delta']==0 for x in cases),changed_strings=sum(x['parent']!=x['chosen'] for x in cases),page_bootstrap_CI95_improvement=[draws[49],draws[1949]],bootstrap_draws=2000,bootstrap_seed=42,bootstrap_scope='reused validation/diagnostic pages, post-selection descriptive interval; not independent writer or confirmatory evidence',page_counts={k:dict(parent_errors=v[0],chosen_errors=v[1],reference_units=v[2]) for k,v in pages.items()})

def independent_CER(rows,predictions):
    lookup={(r['page_id'],r['region_id']):r['text'] for r in predictions}
    if len(lookup)!=len(predictions) or len(lookup)!=len(rows):raise ValueError('prediction coverage mismatch')
    errors=units=0
    for r in rows:
        a=re.sub(r'\s+',' ',unicodedata.normalize('NFC',r['text'])).strip()
        b=re.sub(r'\s+',' ',unicodedata.normalize('NFC',lookup[r['page_id'],r['region_id']])).strip()
        previous=list(range(len(b)+1))
        for i,x in enumerate(a,1):
            current=[i]
            for j,y in enumerate(b,1):current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+(x!=y)))
            previous=current
        errors+=previous[-1];units+=len(a)
    return dict(errors=errors,reference_units=units,CER=errors/units)

def main(root):
    root=Path(root);plan=verify(root);selection=read(root/'selection.json');parent=selection['parent'];chosen=selection['chosen'];out=root/'summary';out.mkdir(exist_ok=False)
    rows=read(root/'validation-rows.json');pr=read(root/'parent-validation/predictions.json')['predictions']
    cp=pr if chosen['name']=='parent' else read(Path(chosen['checkpoint']).parent/f'epoch-{chosen["epoch"]}-validation/predictions.json')['predictions']
    comparison=compare(rows,pr,cp);write_new(out/'validation-cases.json',comparison)
    allcandidates=[parent]+selection['candidates'];fields=['name','regions','reference_units','CER','no_space_CER','exact_match','S','D','I','latin_including_mixed_CER','korean_CER']
    with (out/'validation.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in allcandidates:
            m=r['metrics']['all'];w.writerow(dict(name=r['name'],regions=m['samples'],reference_units=m['reference_units'],CER=m['cer_micro'],no_space_CER=m['cer_no_space_micro'],exact_match=m['exact_match_rate'],S=m['substitutions'],D=m['deletions'],I=m['insertions'],latin_including_mixed_CER=r['metrics']['strata']['latin_including_mixed']['cer_micro'],korean_CER=r['metrics']['strata']['korean']['cer_micro']))
    independent={}
    for row in allcandidates:
        path=root/'parent-validation/predictions.json' if row['name']=='parent' else Path(row['checkpoint']).parent/f'epoch-{row["epoch"]}-validation/predictions.json'
        check=independent_CER(rows,read(path)['predictions'])
        if check['errors']!=row['metrics']['all']['errors'] or check['reference_units']!=row['metrics']['all']['reference_units']:raise ValueError('independent CER mismatch')
        independent[row['name']]=check
    write_new(out/'independent-validation-CER.json',independent)
    verification={arm['name']:read(root/arm['name']/'verification.json') for arm in plan['arms']}
    result=dict(parent=parent['metrics'],chosen=chosen['metrics'],chosen_name=chosen['name'],chosen_checkpoint=chosen['checkpoint'],validation_improvement=parent['metrics']['all']['cer_micro']-chosen['metrics']['all']['cer_micro'],validation_changes={k:v for k,v in comparison.items() if k!='cases'},training_verification=verification,optimizer_steps=sum(v['optimizer_steps'] for v in verification.values()),smoke=read(root/'smoke-latin_weighted_parent_KL/verification.json'),inference_policy=plan['inference'],GT_status='existing frozen AI-provisional labels, no new human review',scope='recognizer-only weights; detector/crops/normalization/read-order unchanged; no deployment')
    if (root/'diagnostic235.json').exists():
        result['diagnostic235']=read(root/'diagnostic235.json')
        from src.training.ocr_finetune import readrows
        rr=readrows(Path(plan['base_config']['dataset'])/'recognition_candidates.test.jsonl');rr.sort(key=lambda x:(x['page_id'],x['region_id']))
        diag=compare(rr,read(root/'diagnostic235-parent/predictions.json')['predictions'],read(root/'diagnostic235-chosen/predictions.json')['predictions']);write_new(out/'diagnostic235-cases.json',diag);result['diagnostic235_changes']={k:v for k,v in diag.items() if k!='cases'}
    else:result['diagnostic235_not_run']=read(root/'final-not-run.json')
    write_new(out/'results.json',result);write_new(root/'selected-model.json',dict(name=chosen['name'],adapter=chosen['checkpoint'],original_parent=str(plan['parent']),scope='local research candidate only; existing application selection not modified',criterion=plan['selection']))
    print(json.dumps({k:v for k,v in result.items() if k not in ['parent','chosen','training_verification','smoke']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
