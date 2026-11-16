"""Clean-train language vs visual LoRA, preserving the active parent adapter."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from src.ocr_data.audit import sha256
from src.ocr_data.train_audit import clean_training
from src.training.ocr_finetune import save,readrows,verify_crops,compare
from src.training.ocr_vl_lora import load,infer,encode,data
from src.evaluation.ocr_v2 import summarize,evaluate


def visual_targets(names,last_layers=4):
    found=[]
    for name in names:
        m=re.search(r'\.visual\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)$',name)
        if m:found.append((int(m[1]),name))
    if not found:raise ValueError('visual q/v modules not found')
    layers=sorted({i for i,_ in found})[-last_layers:]
    selected=[n for i,n in found if i in layers]
    if len(selected)!=2*last_layers:raise ValueError('incomplete visual target set')
    return selected


def activate_adapters(model,arm,training=False):
    names=['language','vision'] if arm=='vision' else ['language']
    # PeftModel.set_adapter accepts one name, the underlying LoraModel supports
    # a list. Both branches must stay active; never replace the parent by vision.
    model.base_model.set_adapter(names,inference_mode=not training)
    for name,p in model.named_parameters():
        p.requires_grad_(training and '.lora_' in name and f'.{arm}.' in name)
    if set(model.active_adapters)!=set(names):raise ValueError('active adapter mismatch')


def frozen_digest(model):
    import torch
    h=hashlib.sha256();count=0
    for n,p in model.named_parameters():
        if p.requires_grad:continue
        h.update(n.encode());h.update(str((list(p.shape),str(p.dtype))).encode())
        h.update(p.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes());count+=p.numel()
    return dict(sha256=h.hexdigest(),parameters=count)


def effective_config(r):
    c=json.loads(Path(r['parent_config']).read_text())
    c.update({k:r[k] for k in ['epochs','learning_rate','seed','rank','alpha','dropout','batch_size','accumulation','test_status']})
    return c


def load_arm(r,arm,training=False,checkpoint=None):
    from peft import PeftModel,LoraConfig,TaskType
    c=effective_config(r);base,proc,info=load(c)
    targets=visual_targets([n for n,_ in base.named_modules()],r['visual_last_layers']) if arm=='vision' else []
    path=Path(checkpoint)/'language' if checkpoint and arm=='language' else Path(r['parent_adapter'])
    model=PeftModel.from_pretrained(base,path,adapter_name='language',is_trainable=training and arm=='language')
    if arm=='vision':
        if checkpoint:model.load_adapter(Path(checkpoint)/'vision',adapter_name='vision',is_trainable=training)
        else:model.add_adapter('vision',LoraConfig(r=r['rank'],lora_alpha=r['alpha'],lora_dropout=r['dropout'],target_modules=targets,bias='none',task_type=TaskType.CAUSAL_LM))
    activate_adapters(model,arm,training)
    return model,proc,info,targets


def verify_inputs(r):
    root=Path(r['output']);receipt=json.loads((root/'start-receipt.json').read_text());c=effective_config(r)
    if receipt['config']!=r:raise ValueError('round config drift')
    for path,digest in receipt['input_hashes'].items():
        if sha256(path)!=digest:raise ValueError(f'input drift: {path}')
    if sha256(Path(c['dataset'])/'manifest.json')!=receipt['dataset_manifest_sha256']:raise ValueError('dataset drift')
    return receipt


def initialize(r):
    root=Path(r['output']);root.mkdir(parents=True,exist_ok=False);c=effective_config(r)
    manifest=json.loads(Path(r['clean_manifest']).read_text());clean=json.loads(Path(r['clean_train']).read_text());original=data(c,'train')
    decisions=json.loads(Path(r['audit_decisions']).read_text());expected,held=clean_training(original,decisions['held_pages'],decisions['held_regions'])
    if clean!=expected or sha256(r['clean_train'])!=manifest['clean_train_sha256']:raise ValueError('clean manifest drift')
    verify_crops(clean,c['dataset']);val=data(c,'validation');primary=[x for x in val if x['page_id'] not in r['quarantined_validation_pages']]
    parent_selected=json.loads((Path(r['parent_adapter']).parent/'selection.json').read_text())
    previous=json.loads((Path(r['parent_adapter']).parent/f"epoch-{parent_selected['epoch']}.validation.predictions.json").read_text())['predictions'];lookup={(p['page_id'],p['region_id']):p['text'] for p in previous}
    baseline=summarize([(x['text'],lookup[x['page_id'],x['region_id']]) for x in primary])
    paths=[r['clean_train'],r['clean_manifest'],r['audit_decisions'],str(Path(r['parent_adapter'])/'adapter_model.safetensors'),str(Path(r['parent_adapter'])/'adapter_config.json')]
    paths += [str(p) for p in Path(c['model']).iterdir() if p.is_file() and p.suffix in ['.safetensors','.json','.jinja']]
    save(root/'round-config.json',r);save(root/'start-receipt.json',dict(config=r,effective=c,input_hashes={p:sha256(p) for p in paths},dataset_manifest_sha256=sha256(Path(c['dataset'])/'manifest.json'),
        source_hashes={p:sha256(p) for p in [__file__,'src/ocr_data/train_audit.py','src/training/ocr_vl_lora.py']},
        train=dict(original=len(original),clean=len(clean),held=len(held),pages=len({x['page_id'] for x in clean})),validation=dict(full=len(val),primary=len(primary),baseline=baseline),test_accessed=False))
    for p in [Path(__file__),Path('src/ocr_data/train_audit.py'),Path('src/training/ocr_vl_lora.py')]: (root/(p.stem+'.executed.py')).write_bytes(p.read_bytes())
    print(json.dumps(dict(clean_train=len(clean),held=len(held),validation_baseline=baseline),ensure_ascii=False),flush=True)


def train_arm(r,arm,smoke=False):
    import gc
    import numpy as np
    import torch
    root=Path(r['output']);out=root/(f'smoke-{arm}' if smoke else arm);out.mkdir(exist_ok=False);verify_inputs(r);c=effective_config(r)
    torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed']);random.seed(r['seed']);np.random.seed(r['seed'])
    model,proc,info,targets=load_arm(r,arm,True)
    rows=json.loads(Path(r['clean_train']).read_text());val=data(c,'validation');held=set(r['quarantined_validation_pages'])
    if smoke:
        # Include long and large crops, so smoke tests more than the first page.
        ranked=sorted(rows,key=lambda x:(-len(x['text']),x['region_id']));rows=[*ranked[:4],*rows[:4]];val=val[:4]
    def score(texts):return summarize([(x['text'],t) for x,t in zip(val,texts) if x['page_id'] not in held])
    # Exact parent identity of zero-initialized visual adapter before any update.
    if arm=='vision':
        activate_adapters(model,'language',False);parent,_=infer(model,proc,val[:4],c,out/'identity.parent.json')
        activate_adapters(model,'vision',True);zero,_=infer(model,proc,val[:4],c,out/'identity.zero-vision.json')
        if parent!=zero:raise ValueError('zero vision adapter changes parent generation')
        save(out/'identity.json',dict(zero_vision_equal_parent=True,validation_samples=4))
    before,rt=infer(model,proc,val,c,out/'before.validation.predictions.json');save(out/'before.validation.metrics.json',dict(primary=score(before),full=summarize([(x['text'],t) for x,t in zip(val,before)]),runtime=rt))
    if not smoke and score(before)!=json.loads((root/'start-receipt.json').read_text())['validation']['baseline']:raise ValueError('parent baseline changed before training')
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    if not trainable or any(f'.{arm}.' not in n or '.lora_' not in n for n in trainable):raise ValueError('wrong trainable parameter scope')
    initial={n:p.detach().clone() for n,p in trainable.items()};frozen_before=frozen_digest(model)
    save(out/'run-receipt.json',dict(config=r,effective=c,arm=arm,smoke=smoke,strict_load=info,source_sha256=sha256(__file__),start_receipt_sha256=sha256(root/'start-receipt.json'),clean_train_sha256=sha256(r['clean_train']),
        trainable_names=list(trainable),trainable_parameters=sum(p.numel() for p in trainable.values()),targets=targets,active_adapters=list(model.active_adapters),frozen_before=frozen_before))
    opt=torch.optim.AdamW(list(trainable.values()),lr=r['learning_rate'],weight_decay=c['weight_decay']);history=[];step=0;best=float('inf')
    epochs=1 if smoke else r['epochs']
    for epoch in range(1,epochs+1):
        model.train()
        if arm=='vision':
            # Parent language LoRA is frozen in both weights AND dropout behavior.
            model.base_model.model.model.language_model.eval()
        order=list(range(len(rows)));random.shuffle(order);batches=math.ceil(len(rows)/r['batch_size']);losses=[];opt.zero_grad();start=time.perf_counter()
        for b,i in enumerate(range(0,len(order),r['batch_size'])):
            part=[rows[k] for k in order[i:i+r['batch_size']]];batch=encode(proc,part,c,True).to(c['device'])
            outputs=model(**batch,use_cache=False);loss=float(outputs.loss.detach())
            if not math.isfinite(loss):raise ValueError('nonfinite loss')
            group=min(r['accumulation'],batches-(b//r['accumulation'])*r['accumulation']);(outputs.loss/group).backward();losses.append(loss)
            if (b+1)%r['accumulation']==0 or b+1==batches:
                norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
                opt.step();opt.zero_grad();step+=1
                if step%10==0:print(json.dumps(dict(arm=arm,epoch=epoch,step=step,loss=loss,seconds=time.perf_counter()-start,gpu_peak_bytes=torch.cuda.max_memory_allocated())),flush=True)
            del outputs,batch
        model.save_pretrained(out/'latest',selected_adapters=[arm]);torch.save(dict(optimizer=opt.state_dict(),epoch=epoch,step=step,python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/'optimizer.pt')
        texts,rt=infer(model,proc,val,c,out/f'epoch-{epoch}.validation.predictions.json');metrics=score(texts)
        record=dict(arm=arm,epoch=epoch,step=step,loss=sum(losses)/len(losses),validation=metrics,full_validation_diagnostic=summarize([(x['text'],t) for x,t in zip(val,texts)]),seconds=time.perf_counter()-start)
        history.append(record);save(out/'history.json',history);print(json.dumps(record),flush=True)
        if metrics['cer_micro']<best:
            best=metrics['cer_micro'];model.save_pretrained(out/'best',selected_adapters=[arm])
            save(out/'selection.json',dict(criterion='minimum fixed primary validation CER',epoch=epoch,step=step,cer=best,test_used=False,checkpoint=str(out/'best'/arm),adapter_sha256=sha256(out/'best'/arm/'adapter_model.safetensors'),adapter_config_sha256=sha256(out/'best'/arm/'adapter_config.json')))
    changed=max(float((p-initial[n]).detach().abs().max()) for n,p in trainable.items());frozen_after=frozen_digest(model)
    if changed<=0 or frozen_before!=frozen_after:raise ValueError('no intended update or frozen parameters changed')
    # Check the serialized best bundle on a fresh base+adapter model object.
    selected=json.loads((out/'selection.json').read_text());model.load_adapter(out/'best'/arm,adapter_name='verify')
    model.base_model.set_adapter(['language','verify'] if arm=='vision' else ['verify'],inference_mode=True)
    expected,_=infer(model,proc,val[:4],c,out/'reload.reference.json')
    del model,opt,trainable,initial;gc.collect();torch.cuda.empty_cache()
    reloaded,proc,_,_=load_arm(r,arm,False,out/'best');actual,_=infer(reloaded,proc,val[:4],c,out/'reload.fresh.json')
    if expected!=actual:raise ValueError('adapter bundle reload mismatch')
    save(out/'verification.json',dict(full_train=not smoke,train_samples=len(rows),validation_full=len(val),validation_primary=score(texts)['samples'],weight_max_update=changed,frozen_parameters_unchanged=True,frozen_digest=frozen_after,bundle_reload_texts_equal=True,comparison_samples=4,gpu_peak_bytes=torch.cuda.max_memory_allocated()))


def select(r):
    root=Path(r['output']);start=verify_inputs(r);candidates=[dict(name='parent',arm='parent',checkpoint=r['parent_adapter'],metrics=start['validation']['baseline'])]
    for arm in ['language','vision']:
        selected=json.loads((root/arm/'selection.json').read_text());history=json.loads((root/arm/'history.json').read_text());record=next(x for x in history if x['epoch']==selected['epoch'])
        path=Path(selected['checkpoint'])
        if sha256(path/'adapter_model.safetensors')!=selected['adapter_sha256'] or sha256(path/'adapter_config.json')!=selected['adapter_config_sha256']:raise ValueError('checkpoint drift')
        candidates.append(dict(name=arm,arm=arm,checkpoint=str(root/arm/'best'),metrics=record['validation'],selection=selected))
    chosen=min(candidates,key=lambda x:x['metrics']['cer_micro']);save(root/'selection-lock.json',dict(chosen=chosen,candidates=candidates,criterion=r['selection'],test_used=False,start_receipt_sha256=sha256(root/'start-receipt.json')))
    print(json.dumps(dict(chosen=chosen['name'],CER=chosen['metrics']['cer_micro'])),flush=True)


def final(r):
    import gc
    import torch
    root=Path(r['output']);out=root/'final';out.mkdir(exist_ok=False);verify_inputs(r);lock=json.loads((root/'selection-lock.json').read_text());chosen=lock['chosen'];c=effective_config(r)
    if lock['start_receipt_sha256']!=sha256(root/'start-receipt.json'):raise ValueError('lock drift')
    if chosen['arm']!='parent':
        p=Path(chosen['selection']['checkpoint'])
        if sha256(p/'adapter_model.safetensors')!=chosen['selection']['adapter_sha256'] or sha256(p/'adapter_config.json')!=chosen['selection']['adapter_config_sha256']:raise ValueError('chosen checkpoint drift')
    rows=data(c,'test');model,proc,_,_=load_arm(r,'parent',False);before,bt=infer(model,proc,rows,c,out/'previous.test.predictions.json')
    if chosen['arm']=='parent':
        after=before;at=bt;payload=json.loads((out/'previous.test.predictions.json').read_text());payload['reused_identical_selected_parent']=True;save(out/'selected.test.predictions.json',payload)
    else:
        del model;gc.collect();torch.cuda.empty_cache();model,proc,_,_=load_arm(r,chosen['arm'],False,chosen['checkpoint']);after,at=infer(model,proc,rows,c,out/'selected.test.predictions.json')
    result=compare(rows,before,after);result.update(selection=chosen,previous_runtime=bt,selected_runtime=at,selected_predictions_reused=chosen['arm']=='parent',test_status=r['test_status']);save(out/'comparison.test.json',result)
    from src.evaluation.ocr_finetune_page import pages_for
    old=json.loads(Path('configs/ocr_v2_korean_ctc.json').read_text());frozen=Path('data/experiments/ocr_v2_korean/det-test/detections.json');det=json.loads(frozen.read_text())
    if det['dataset_manifest_sha256']!=sha256(Path(c['dataset'])/'manifest.json'):raise ValueError('detector manifest drift')
    detrows=[x for page in det['pages'] for x in page['regions']];verify_crops(detrows,'.')
    texts,rt=infer(model,proc,detrows,c,out/'selected.detector-crops.json');lookup={(x['page_id'],x['region_id']):t for x,t in zip(detrows,texts)};prediction=dict(mode='page',pages=copy.deepcopy(det['pages']))
    for page in prediction['pages']:
        for region in page['regions']:region['text']=lookup[region['page_id'],region['region_id']]
    metrics=evaluate(pages_for(old,'test'),prediction,.5);metrics.update(runtime=rt,detector_sha256=sha256(frozen),test_status=r['test_status'])
    save(out/'selected.pages.predictions.json',prediction);save(out/'selected.pages.metrics.json',metrics)
    print(json.dumps({k:result[k] for k in ['before','after','relative_cer_reduction','paired_page_bootstrap']},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/ocr_clean_vision.json');p.add_argument('--stage',choices=['all','init','smoke-language','smoke-vision','language','vision','select','final'],default='all');a=p.parse_args();r=json.loads(Path(a.config).read_text());root=Path(r['output'])
    if a.stage=='all':
        initialize(r)
        for stage in ['smoke-language','smoke-vision','language','vision','select','final']:
            with (root/f'{stage}.log').open('x') as log:subprocess.run([sys.executable,'-m','src.training.ocr_clean_vision','--config',a.config,'--stage',stage],stdout=log,stderr=subprocess.STDOUT,check=True)
        return
    if a.stage=='init':initialize(r);return
    r=json.loads((root/'round-config.json').read_text());(root/f'{a.stage}.source.py').write_bytes(Path(__file__).read_bytes())
    if a.stage=='select':select(r)
    elif a.stage=='final':final(r)
    else:train_arm(r,a.stage.removeprefix('smoke-'),a.stage.startswith('smoke-'))

if __name__=='__main__':main()
