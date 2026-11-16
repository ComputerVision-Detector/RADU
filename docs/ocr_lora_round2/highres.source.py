"""Bounded continuation/resolution ablation; select on quarantined validation only."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from src.ocr_data.audit import sha256
from src.training.ocr_finetune import save,readrows,compare
from src.training.ocr_vl_lora import load,infer,data,train
from src.evaluation.ocr_v2 import summarize,evaluate


def primary_validation(rows,quarantine):
    return ([r for r in rows if r['page_id'] not in quarantine],
            [r for r in rows if r['page_id'] in quarantine])


def verify_selected_adapter(directory,selection):
    digest=sha256(Path(directory)/'adapter_model.safetensors')
    if digest!=selection['adapter_sha256']:raise ValueError('selected adapter differs from validation checkpoint')
    return digest


def verify_base_receipt(config,receipt):
    if sha256(Path(config['dataset'])/'manifest.json')!=receipt['dataset_manifest_sha256']:raise ValueError('dataset drift')
    for name,digest in receipt['model_files_sha256'].items():
        if sha256(Path(config['model'])/name)!=digest:raise ValueError(f'base model drift: {name}')
    for split,digest in receipt['splits_sha256'].items():
        if sha256(Path(config['dataset'])/f'recognition_candidates.{split}.jsonl')!=digest:raise ValueError(f'split drift: {split}')


def arm_config(base,round_config,arm):
    c=copy.deepcopy(base);c.update(round_config['arms'][arm])
    c.update(seed=round_config['seed'],epochs=round_config['epochs'],learning_rate=round_config['learning_rate'],
             initial_adapter=round_config['parent_adapter'],output=str(Path(round_config['output'])/arm))
    if 'quarantined_validation_pages' in round_config:c['quarantined_validation_pages']=round_config['quarantined_validation_pages']
    if 'initial_adapter_sha256' in round_config:c['initial_adapter_sha256']=round_config['initial_adapter_sha256']
    return c


def initialize(r):
    root=Path(r['output']);root.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path(r['parent_config']).read_text());r=dict(r,initial_adapter_sha256=sha256(Path(r['parent_adapter'])/'adapter_model.safetensors'))
    val=data(base,'validation');kept,held=primary_validation(val,r['quarantined_validation_pages'])
    parent_train=Path(r['parent_adapter']).parent
    parent_epoch=json.loads((parent_train/'selection.json').read_text())['epoch']
    previous=json.loads((parent_train/f'epoch-{parent_epoch}.validation.predictions.json').read_text())['predictions']
    keyed={(p['page_id'],p['region_id']):p['text'] for p in previous}
    baseline=summarize([(p['text'],keyed[p['page_id'],p['region_id']]) for p in kept])
    save(root/'round-config.json',r);save(root/'quarantine.json',dict(pages=r['quarantined_validation_pages'],regions=held,policy='whole validation page quarantine; unchanged original GT retained only in full diagnostic'))
    for source in [Path(__file__),Path('src/training/ocr_vl_lora.py')]:
        (root/(source.stem+'.executed.py')).write_bytes(source.read_bytes())
    for arm in r['arms']:save(root/f'{arm}.config.json',arm_config(base,r,arm))
    save(root/'start-receipt.json',dict(round_config=r,base_config=base,source_hashes={f:sha256(f) for f in [__file__,'src/training/ocr_vl_lora.py','configs/ocr_v2_vl_lora_round2.json']},
        dataset_manifest_sha256=sha256(Path(base['dataset'])/'manifest.json'),parent_adapter_sha256=r['initial_adapter_sha256'],
        validation_primary=dict(crops=len(kept),pages=len({x['page_id'] for x in kept}),baseline=baseline),
        validation_quarantined=dict(crops=len(held),pages=len({x['page_id'] for x in held})),test_accessed=False))
    print(json.dumps(dict(primary=baseline,quarantined=len(held)),ensure_ascii=False),flush=True)


def probe(r):
    from peft import PeftModel
    root=Path(r['output']);out=root/'probe';out.mkdir(exist_ok=False)
    base=json.loads(Path(r['parent_config']).read_text());model,proc,_=load(base)
    model=PeftModel.from_pretrained(model,r['parent_adapter']);rows=data(base,'validation')
    for arm in r['arms']:
        c=json.loads((root/f'{arm}.config.json').read_text());texts,rt=infer(model,proc,rows,c,out/f'{arm}.predictions.json')
        primary=summarize([(row['text'],text) for row,text in zip(rows,texts) if row['page_id'] not in r['quarantined_validation_pages']])
        save(out/f'{arm}.metrics.json',dict(primary=primary,full_validation_diagnostic=summarize([(row['text'],text) for row,text in zip(rows,texts)]),runtime=rt))
        print(json.dumps(dict(probe=arm,primary=primary),ensure_ascii=False),flush=True)


def select(r):
    root=Path(r['output']);candidates=[]
    for arm in r['arms']:
        metric=json.loads((root/f'probe/{arm}.metrics.json').read_text())['primary']
        c=json.loads((root/f'{arm}.config.json').read_text())
        candidates.append(dict(name='parent_'+arm,kind='parent',adapter=r['parent_adapter'],metrics=metric,config=c))
        selection=json.loads((root/arm/'selection.json').read_text());history=json.loads((root/arm/'history.json').read_text())
        verify_selected_adapter(root/arm/'best-adapter',selection)
        receipt=json.loads((root/arm/'run-receipt.json').read_text())
        if receipt['config']!=c:raise ValueError('arm config drift')
        verify_base_receipt(c,receipt)
        selected=next(x for x in history if x['epoch']==selection['epoch'])
        corrected=dict(selection,criterion='minimum primary non-quarantined validation micro CER',validation_primary_samples=selected['validation']['samples'],validation_full_samples=selected['full_validation_diagnostic']['samples'])
        if corrected!=selection:
            save(root/arm/'selection-description-correction.json',dict(original=selection,corrected=corrected,reason='shared trainer used legacy full-validation label; metric and weights unchanged'))
            save(root/arm/'selection.json',corrected);selection=corrected
        candidates.append(dict(name='trained_'+arm,kind='trained',adapter=str(root/arm/'best-adapter'),metrics=selected['validation'],config=c,selection=selection))
    chosen=min(candidates,key=lambda x:x['metrics']['cer_micro'])
    for c in candidates:
        c['adapter_sha256']=sha256(Path(c['adapter'])/'adapter_model.safetensors')
        c['adapter_config_sha256']=sha256(Path(c['adapter'])/'adapter_config.json')
    save(root/'selection-lock.json',dict(criterion=r['selection'],candidates=candidates,chosen=chosen,test_used=False,
        parent_adapter_sha256=r['initial_adapter_sha256'],dataset_manifest_sha256=sha256(Path(chosen['config']['dataset'])/'manifest.json')))
    print(json.dumps(dict(chosen=chosen['name'],CER=chosen['metrics']['cer_micro'])),flush=True)


def final(r):
    from peft import PeftModel
    root=Path(r['output']);out=root/'final';out.mkdir(exist_ok=False)
    lock=json.loads((root/'selection-lock.json').read_text());chosen=lock['chosen'];base=json.loads(Path(r['parent_config']).read_text())
    if chosen['adapter_sha256']!=sha256(Path(chosen['adapter'])/'adapter_model.safetensors'):raise ValueError('selected adapter drift')
    if chosen['adapter_config_sha256']!=sha256(Path(chosen['adapter'])/'adapter_config.json'):raise ValueError('selected adapter config drift')
    if lock['parent_adapter_sha256']!=sha256(Path(r['parent_adapter'])/'adapter_model.safetensors'):raise ValueError('parent adapter drift')
    if lock['dataset_manifest_sha256']!=sha256(Path(base['dataset'])/'manifest.json'):raise ValueError('dataset drift')
    receipt=json.loads((root/chosen['name'].removeprefix('trained_')/'run-receipt.json').read_text()) if chosen['kind']=='trained' else json.loads(Path('data/experiments/ocr_v2_vl_lora/train/run-receipt.json').read_text())
    verify_base_receipt(chosen['config'],receipt)
    rows=data(base,'test');model,proc,_=load(base);model=PeftModel.from_pretrained(model,r['parent_adapter'])
    previous,pt=infer(model,proc,rows,base,out/'round1.test.predictions.json')
    if chosen['config']['min_pixels']==base['min_pixels'] and chosen['config']['max_pixels']==base['max_pixels']:matched=previous;mt=pt
    else:matched,mt=infer(model,proc,rows,chosen['config'],out/'parent-matched-resolution.test.predictions.json')
    if chosen['kind']=='trained':model.load_adapter(chosen['adapter'],adapter_name='round2');model.set_adapter('round2')
    after,at=infer(model,proc,rows,chosen['config'],out/'selected.test.predictions.json')
    result=compare(rows,previous,after);result.update(selection=chosen,round1_runtime=pt,selected_runtime=at,test_status=r['test_policy'])
    save(out/'comparison.test.json',result);save(out/'matched-resolution.test.json',compare(rows,matched,after))
    print(json.dumps({k:result[k] for k in ['before','after','relative_cer_reduction','paired_page_bootstrap']},ensure_ascii=False),flush=True)
    # Same frozen 397 detections and 199-GT reference scope as preceding rounds.
    from src.evaluation.ocr_finetune_page import pages_for
    old=json.loads(Path('configs/ocr_v2_korean_ctc.json').read_text());frozen=Path('data/experiments/ocr_v2_korean/det-test/detections.json');det=json.loads(frozen.read_text())
    if det['dataset_manifest_sha256']!=lock['dataset_manifest_sha256']:raise ValueError('detector dataset manifest drift')
    detrows=[p for page in det['pages'] for p in page['regions']]
    for row in detrows:
        if sha256(row['image'])!=row['crop_sha256']:raise ValueError('detector crop drift')
    texts,rt=infer(model,proc,detrows,chosen['config'],out/'selected.test.detector-crops.json');lookup={(row['page_id'],row['region_id']):text for row,text in zip(detrows,texts)}
    pred=dict(mode='page',pages=copy.deepcopy(det['pages']))
    for page in pred['pages']:
        for region in page['regions']:region['text']=lookup[region['page_id'],region['region_id']]
    metric=evaluate(pages_for(old,'test'),pred,.5);metric.update(runtime=rt,scope='same frozen detector397 / prior199GT text scope',detector_sha256=sha256(frozen))
    save(out/'selected.test.pages.predictions.json',pred);save(out/'selected.test.pages.metrics.json',metric)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/ocr_v2_vl_lora_round2.json');p.add_argument('--stage',choices=['all','init','probe','smoke','continue','highres','select','final'],default='all');a=p.parse_args()
    r=json.loads(Path(a.config).read_text());root=Path(r['output'])
    if a.stage=='all':
        initialize(r)
        for stage in ['probe','smoke','continue','highres','select','final']:
            with (root/f'{stage}.log').open('x') as log:subprocess.run([sys.executable,'-m','src.training.ocr_lora_round2','--config',a.config,'--stage',stage],stdout=log,stderr=subprocess.STDOUT,check=True)
        return
    if a.stage=='init':initialize(r);return
    r=json.loads((root/'round-config.json').read_text())
    (root/f'{a.stage}.source.py').write_bytes(Path(__file__).read_bytes())
    if a.stage=='probe':probe(r)
    elif a.stage=='smoke':
        c=json.loads((root/'continue.config.json').read_text());out=root/'smoke';out.mkdir(exist_ok=False);train(c,out,smoke=True)
    elif a.stage=='select':select(r)
    elif a.stage=='final':final(r)
    else:
        c=json.loads((root/f'{a.stage}.config.json').read_text());out=root/a.stage;out.mkdir(exist_ok=False);train(c,out)

if __name__=='__main__':main()
