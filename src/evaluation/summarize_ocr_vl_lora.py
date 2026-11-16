"""Summarize saved real LoRA measurements and retain compact review evidence."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

from src.ocr_data.audit import sha256
from src.training.ocr_finetune import save,readrows,compare


def main():
    root=Path('data/experiments/ocr_v2_vl_lora');out=Path('docs/ocr_vl_lora');out.mkdir(parents=True,exist_ok=True)
    def load(p):return json.loads((root/p).read_text())
    result=load('final/comparison.test.json');summary=dict(crop_test=result,selection=load('train/selection.json'),
        history=load('train/history.json'),baseline_validation=load('train/before.validation.metrics.json'),
        verification=load('train/verification.json'),lora=load('train/lora-parameters.json'),tokenizer_audit=load('tokenizer-audit.json'),
        changed_crops=dict(Counter('improved' if x['after_errors']<x['before_errors'] else 'worsened' if x['after_errors']>x['before_errors'] else 'same_errors' for x in result['cases'])))
    # A strict 199-crop subset allows reference to the prior study without mixing
    # its different 235-crop VL vocabulary coverage into cross-model comparisons.
    previous=json.loads(Path('data/experiments/ocr_v2_korean/prepared/test.json').read_text())
    preds={(x['page_id'],x['region_id']):x for x in result['cases']}
    summary['prior_199_crop_subset']=compare(previous,[preds[x['page_id'],x['region_id']]['before'] for x in previous],[preds[x['page_id'],x['region_id']]['after'] for x in previous])
    summary['pages']={}
    for split in ['validation','test']:
        if not (root/f'pages/after.{split}.metrics.json').exists():continue
        summary['pages'][split]={name:load(f'pages/{name}.{split}.metrics.json') for name in ['before','after']}
    save(out/'summary.json',summary)
    fields=['cer_micro','cer_no_space_micro','exact_match_rate','substitutions','deletions','insertions']
    with (out/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['scope','metric','before','after','after_minus_before']);writer.writeheader()
        pairs=[('GT crop test235',result['before'],result['after'])]
        if 'test' in summary['pages']:pairs.append(('fixed detector test pages199GT',summary['pages']['test']['before']['end_to_end'],summary['pages']['test']['after']['end_to_end']))
        for scope,b,a in pairs:
            for key in fields:writer.writerow(dict(scope=scope,metric=key,before=b[key],after=a[key],after_minus_before=a[key]-b[key]))
    for name in ['environment.freeze.txt','training_source.executed.py','train.log','final.log','smoke.log','tokenizer-audit.json','group-bootstrap.json','error-examples.json','error-crops.jpg','pages.log']:
        shutil.copyfile(root/name,out/name)
    for name in ['run-receipt.json','selection.json','verification.json','history.json','lora-parameters.json']:
        shutil.copyfile(root/'train'/name,out/('train-'+name))
    shutil.copyfile(root/'final/final-lock.json',out/'final-lock.json')
    save(out/'artifacts.sha256.json',{str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file()})
    save(out/'code.sha256.json',{str(p):sha256(p) for p in [Path('src/training/ocr_vl_lora.py'),Path('src/cv/ocr_vl_lora_infer.py'),Path('src/evaluation/ocr_vl_lora_pages.py'),Path('src/evaluation/summarize_ocr_vl_lora.py'),Path('configs/ocr_v2_vl_lora.json'),Path('tests/test_ocr_vl_lora.py')]})
    print(json.dumps({k:result[k] for k in ['before','after','relative_cer_reduction','paired_page_bootstrap']},ensure_ascii=False))

if __name__=='__main__':main()
