"""Summarize actual clean-train language/vision runs and audit evidence."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

from src.ocr_data.audit import sha256
from src.training.ocr_finetune import save,readrows,compare
from src.evaluation.ocr_metrics import normalize


def main():
    root=Path('data/experiments/ocr_v2_vl_lora_round3');out=Path('docs/ocr_clean_vision');out.mkdir(parents=True,exist_ok=True);audit=Path('data/processed/ocr_v2/train-audit-round3')
    def load(p):return json.loads((root/p).read_text())
    start=load('start-receipt.json');r=start['config'];lock=load('selection-lock.json');test=load('final/comparison.test.json')
    arms={arm:dict(history=load(f'{arm}/history.json'),selection=load(f'{arm}/selection.json'),verification=load(f'{arm}/verification.json'),receipt=load(f'{arm}/run-receipt.json')) for arm in ['language','vision']}
    old_pages=json.loads(Path('data/experiments/ocr_v2_vl_lora/pages/after.test.metrics.json').read_text());new_pages=load('final/selected.pages.metrics.json')
    clean=json.loads(Path(r['clean_train']).read_text());held=json.loads((audit/'held-train.json').read_text());original=readrows(Path(start['effective']['dataset'])/'recognition_candidates.train.jsonl')
    def lang(x):
        s=x['text'];k=any('가'<=c<='힣' for c in s);e=any(c.isascii() and c.isalpha() for c in s)
        return 'mixed' if k and e else 'korean' if k else 'latin_numeric_other'
    corpus={name:dict(crops=len(rows),pages=len({x['page_id'] for x in rows}),normalized_characters=sum(len(normalize(x['text'])) for x in rows),language=dict(Counter(lang(x) for x in rows))) for name,rows in [('original',original),('clean',clean),('held',held)]}
    report=dict(start=start,audit=json.loads((audit/'audit-summary.json').read_text()),clean_manifest=json.loads((audit/'clean-manifest.json').read_text()),decisions=json.loads((audit/'decisions.json').read_text()),corpus=corpus,arms=arms,selection=lock,test=test,pages=dict(previous=old_pages,selected=new_pages))
    save(out/'summary.json',report)
    with (out/'validation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['arm','epoch','step','CER244','no_space_CER244','exact244','CER297','loss']);w.writeheader();b=start['validation']['baseline'];w.writerow(dict(arm='parent',epoch=0,step=0,CER244=b['cer_micro'],no_space_CER244=b['cer_no_space_micro'],exact244=b['exact_match_rate']))
        for arm,measurements in arms.items():
            for h in measurements['history']:
                m=h['validation'];w.writerow(dict(arm=arm,epoch=h['epoch'],step=h['step'],CER244=m['cer_micro'],no_space_CER244=m['cer_no_space_micro'],exact244=m['exact_match_rate'],CER297=h['full_validation_diagnostic']['cer_micro'],loss=h['loss']))
    with (out/'test.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['scope','metric','before','after','delta']);w.writeheader()
        for scope,b,a in [('GTcrop235',test['before'],test['after']),('fixed-detector397_199GT',old_pages['end_to_end'],new_pages['end_to_end'])]:
            for key in ['cer_micro','cer_no_space_micro','exact_match_rate','substitutions','deletions','insertions']:
                w.writerow(dict(scope=scope,metric=key,before=b[key],after=a[key],delta=a[key]-b[key]))
    for name in ['start-receipt.json','round-config.json','selection-lock.json','validation-comparisons.json','language.log','vision.log','smoke-language.log','smoke-vision.log','final.log','ocr_clean_vision.executed.py','train_audit.executed.py','ocr_vl_lora.executed.py']:
        shutil.copyfile(root/name,out/name)
    for name in ['audit-summary.json','review-index.json','decisions.json','clean-manifest.json','clean-train.json','held-train.json']:
        shutil.copyfile(audit/name,out/name)
    for p in root.glob('*.source.py'):shutil.copyfile(p,out/p.name)
    if (root/'demo-vision/comparison.json').exists():shutil.copyfile(root/'demo-vision/comparison.json',out/'demo-vision.json')
    save(out/'artifact.sha256.json',{str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file()})
    save(out/'audit-artifact.sha256.json',{str(p.relative_to(audit)):sha256(p) for p in sorted(audit.rglob('*')) if p.is_file()})
    save(out/'code.sha256.json',{str(p):sha256(p) for p in [Path('src/training/ocr_clean_vision.py'),Path('src/ocr_data/train_audit.py'),Path('src/cv/ocr_clean_vision_infer.py'),Path('src/evaluation/summarize_ocr_clean_vision.py'),Path('configs/ocr_clean_vision.json'),Path('tests/test_ocr_clean_vision.py'),Path('tests/test_ocr_train_audit.py')]})
    save(out/'test-error-examples.json',sorted(test['cases'],key=lambda x:abs(x['before_errors']-x['after_errors']),reverse=True)[:12])
    print(json.dumps(dict(chosen=lock['chosen']['name'],before=test['before'],after=test['after'],relative_reduction=test['relative_cer_reduction']),ensure_ascii=False))

if __name__=='__main__':main()
