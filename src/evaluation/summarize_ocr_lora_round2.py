"""Collect actual validation ablations and locked round-two test measurements."""
import csv
import json
from pathlib import Path
import shutil

from src.ocr_data.audit import sha256
from src.training.ocr_finetune import save,readrows,compare


def main():
    root=Path('data/experiments/ocr_v2_vl_lora_round2');out=Path('docs/ocr_lora_round2');out.mkdir(parents=True,exist_ok=True)
    def load(p):return json.loads((root/p).read_text())
    start=load('start-receipt.json');lock=load('selection-lock.json');test=load('final/comparison.test.json')
    arms={name:dict(history=load(f'{name}/history.json'),selection=load(f'{name}/selection.json'),verification=load(f'{name}/verification.json')) for name in ['continue','highres']}
    probes={name:load(f'probe/{name}.metrics.json') for name in ['continue','highres']}
    pages_before=json.loads(Path('data/experiments/ocr_v2_vl_lora/pages/after.test.metrics.json').read_text())
    pages_after=load('final/selected.test.pages.metrics.json')
    report=dict(start=start,validation_probes=probes,arms=arms,selection=lock,test=test,
        halfstep=load('halfstep/metrics.json'),adaptive_extension=load('adaptive-plan-extension.json'),
        matched_resolution_test=load('final/matched-resolution.test.json'),pages=dict(previous=pages_before,selected=pages_after),
        scope_note='validation244 excludes53 crops on two predeclared visually problematic pages; full297 diagnostic retained; unchanged exploratory test235')
    save(out/'summary.json',report)
    with (out/'validation.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['candidate','epoch','primary_crops','primary_CER','primary_no_space_CER','primary_exact','full297_CER']);writer.writeheader()
        for name,probe in probes.items():
            p=probe['primary'];writer.writerow(dict(candidate='parent_'+name,epoch=0,primary_crops=p['samples'],primary_CER=p['cer_micro'],primary_no_space_CER=p['cer_no_space_micro'],primary_exact=p['exact_match_rate'],full297_CER=probe['full_validation_diagnostic']['cer_micro']))
        for name,arm in arms.items():
            for h in arm['history']:
                p=h['validation'];writer.writerow(dict(candidate='trained_'+name,epoch=h['epoch'],primary_crops=p['samples'],primary_CER=p['cer_micro'],primary_no_space_CER=p['cer_no_space_micro'],primary_exact=p['exact_match_rate'],full297_CER=h['full_validation_diagnostic']['cer_micro']))
        blend=load('halfstep/metrics.json');p=blend['primary'];writer.writerow(dict(candidate='halfstep_continue',epoch='no extra training',primary_crops=p['samples'],primary_CER=p['cer_micro'],primary_no_space_CER=p['cer_no_space_micro'],primary_exact=p['exact_match_rate'],full297_CER=blend['full_validation_diagnostic']['cer_micro']))
    with (out/'test.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['scope','metric','previous','selected','delta']);writer.writeheader()
        for scope,b,a in [('GTcrop235',test['before'],test['after']),('fixed-detector pages199GT',pages_before['end_to_end'],pages_after['end_to_end'])]:
            for metric in ['cer_micro','cer_no_space_micro','exact_match_rate','substitutions','deletions','insertions']:
                writer.writerow(dict(scope=scope,metric=metric,previous=b[metric],selected=a[metric],delta=a[metric]-b[metric]))
    for name in ['start-receipt.json','round-config.json','quarantine.json','quarantine-visual-evidence.jpg','selection-lock.json','continue.log','highres.log','probe.log','smoke.log','final.log','ocr_lora_round2.executed.py','ocr_vl_lora.executed.py','adaptive-plan-extension.json','validation-comparisons.json','processor-preflight.json']:
        shutil.copyfile(root/name,out/name)
    # Save a few paired outcome examples without changing any GT text.
    cases=test['cases'];chosen=[]
    for kind,predicate in [('improved',lambda c:c['after_errors']<c['before_errors']),('worsened',lambda c:c['after_errors']>c['before_errors']),('same',lambda c:c['after_errors']==c['before_errors'])]:
        chosen.extend(dict(c,change=kind) for c in sorted([c for c in cases if predicate(c)],key=lambda c:abs(c['after_errors']-c['before_errors']),reverse=True)[:3])
    save(out/'error-examples.json',chosen)
    for name in ['select.source.py','final.source.py','highres.source.py']:
        if (root/name).exists():shutil.copyfile(root/name,out/name)
    save(out/'code.sha256.json',{str(p):sha256(p) for p in [Path('src/training/ocr_lora_round2.py'),Path('src/training/ocr_vl_lora.py'),Path('src/evaluation/summarize_ocr_lora_round2.py'),Path('configs/ocr_v2_vl_lora_round2.json'),Path('tests/test_ocr_lora_round2.py')]})
    save(out/'artifact.sha256.json',{str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*')) if p.is_file()})
    print(json.dumps(dict(chosen=lock['chosen']['name'],before=test['before'],after=test['after'],relative_reduction=test['relative_cer_reduction']),ensure_ascii=False))

if __name__=='__main__':main()
