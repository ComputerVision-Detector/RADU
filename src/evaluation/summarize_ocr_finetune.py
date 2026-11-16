"""Collect actual saved measurements; never synthesize missing model results."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

from src.ocr_data.audit import sha256
from src.training.ocr_finetune import save


def summarize(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def load(p):return json.loads((root/p).read_text())
    rec=load('final/comparison.test.json');pb=load('page-test/before.test.pages.metrics.json');pa=load('page-test/after.test.pages.metrics.json')
    split=json.loads(Path('data/processed/ocr_v2/pilot-v2/splits.json').read_text())
    prepared={s:load(f'prepared/{s}.json') for s in ['train','validation','test']}
    exclusions=load('prepared/held.json');cases=rec['cases']
    rows=[]
    for scope,b,a in [('test GT crops',rec['before'],rec['after']),('test fixed-detector pages',pb['end_to_end'],pa['end_to_end'])]:
        for key in ['cer_micro','cer_no_space_micro','exact_match_rate','substitutions','deletions','insertions']:
            rows.append(dict(scope=scope,metric=key,before=b[key],after=a[key],after_minus_before=a[key]-b[key]))
    with (output/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['scope','metric','before','after','after_minus_before']);w.writeheader();w.writerows(rows)
    vlpath=root/'vl-validation/metrics.validation.json'
    summary=dict(recognition_test=rec,fixed_detector_test=dict(before=pb,after=pa),
        counts={s:dict(crops=len(r),pages=len({x['page_id'] for x in r}),normalized_characters=__import__('src.evaluation.ocr_v2',fromlist=['summarize']).summarize([(x['text'],x['text']) for x in r])['reference_units'],split_groups=sum(split['assignments'][c[0]]==s for c in split['components']),recognition_groups=sum(bool(set(c)&{x['page_id'] for x in r}) for c in split['components'])) for s,r in prepared.items()},
        additional_holds=dict(total=len(exclusions),by_split=dict(Counter(x['split'] for x in exclusions)),reasons=dict(Counter(reason for x in exclusions for reason in x['reasons'])),unsupported_characters=dict(Counter(c for x in exclusions for c in x['unsupported']))),
        changed_crops=dict(Counter('improved' if c['after_errors']<c['before_errors'] else 'worsened' if c['after_errors']>c['before_errors'] else 'same_error_count' for c in cases)),
        selection=load('train/selection.json'),training_verification=load('train/verification.json'),fresh_process_resume=load('resume-fresh/receipt.json'),
        vl_validation=json.loads(vlpath.read_text()) if vlpath.exists() else None,
        test_development_policy='No retuning after test; later code edits are integrity guards, receipts, reporting and resume verification only.')
    save(output/'summary.json',summary)
    for name in ['environment.json','environment.freeze.txt','execution-audit.json','final-lock.json','error-examples.json','error-crops.jpg','group-bootstrap.json','training_source.executed.py','train.log','smoke-long.log']:
        shutil.copyfile(root/name,output/Path(name).name)
    for name in ['selection.json','verification.json','history.json','input-receipt.json','checkpoint-receipt.json']:
        shutil.copyfile(root/'train'/name,output/('train-'+name))
    save(output/'artifacts.sha256.json',{str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file()})
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='data/experiments/ocr_v2_korean');p.add_argument('--output',default='docs/ocr_v2_finetune');a=p.parse_args()
    result=summarize(a.root,a.output);print(json.dumps({k:result[k] for k in ['counts','additional_holds','changed_crops']},ensure_ascii=False))

if __name__=='__main__':main()
