"""Real, bounded PaddleOCR-VL 1.6 language-only LoRA and paired inference."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

from src.ocr_data.audit import sha256
from src.training.ocr_finetune import save,readrows,verify_crops,compare
from src.evaluation.ocr_v2 import summarize


def target_modules(names):
    selected=[n for n in names if '.language_model.' in n and n.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj'}]
    if not selected:raise ValueError('no language attention targets')
    return selected


def completion_labels(full,prefix,attention):
    active=[x for x,a in zip(full,attention) if a]
    if active[:len(prefix)]!=prefix or len(active)<=len(prefix):raise ValueError('assistant prefix mismatch or empty completion')
    seen=0;labels=[]
    for x,a in zip(full,attention):
        labels.append(x if a and seen>=len(prefix) else -100)
        if a:seen+=1
    return labels


def generation_limit(tokens,maximum,eos):
    return len(tokens)>=maximum and eos not in [int(t) for t in tokens]


def messages(prompt,text=None):
    result=[dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])]
    if text is not None:result.append(dict(role='assistant',content=[dict(type='text',text=text)]))
    return result


def encode(proc,rows,c,training):
    import torch
    from PIL import Image
    images=[];texts=[];prefixes=[]
    for r in rows:
        with Image.open(Path(c['dataset'])/r['image']) as im:images.append(im.convert('RGB'))
        prefixes.append(proc.apply_chat_template(messages(c['prompt']),tokenize=False,add_generation_prompt=True))
        texts.append(proc.apply_chat_template(messages(c['prompt'],r['text'] if training else None),tokenize=False,add_generation_prompt=not training))
    kw=dict(images_kwargs={'size':{'shortest_edge':c['min_pixels'],'longest_edge':c['max_pixels']}})
    # Processor direct kwargs accepts size; same image preprocessing for full/prefix.
    batch=proc(text=texts,images=images,padding=True,return_tensors='pt',truncation=False,size=kw['images_kwargs']['size'])
    if batch['input_ids'].shape[1]>c['max_sequence']:raise ValueError('sequence budget exceeded; labels not truncated')
    if training:
        labels=batch['input_ids'].clone()
        for i,(prefix,image) in enumerate(zip(prefixes,images)):
            p=proc(text=[prefix],images=[image],return_tensors='pt',size=kw['images_kwargs']['size'])
            masked=completion_labels(batch['input_ids'][i].tolist(),p['input_ids'][0].tolist(),batch['attention_mask'][i].tolist())
            labels[i]=torch.tensor(masked,dtype=labels.dtype)
        batch['labels']=labels
    return batch


def load(c):
    import torch
    from transformers import AutoModelForImageTextToText,AutoProcessor
    model,info=AutoModelForImageTextToText.from_pretrained(c['model'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
    if any(info[k] for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):raise ValueError(str(info))
    model=model.to(c['device']);model.config.use_cache=False
    proc=AutoProcessor.from_pretrained(c['model'],local_files_only=True)
    return model,proc,{k:list(v) for k,v in info.items()}


def infer(model,proc,rows,c,out):
    import torch
    model.eval();proc.tokenizer.padding_side='left';preds=[];failures=[];start=time.perf_counter()
    for i in range(0,len(rows),c['generation_batch_size']):
        part=rows[i:i+c['generation_batch_size']]
        try:
            batch=encode(proc,part,c,False).to(c['device'])
            with torch.inference_mode():ids=model.generate(**batch,do_sample=False,num_beams=1,max_new_tokens=c['max_new_tokens'],use_cache=True)
            generated=ids[:,batch['input_ids'].shape[1]:]
            texts=proc.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            for r,t,tokens in zip(part,texts,generated):
                preds.append(dict(page_id=r['page_id'],region_id=r['region_id'],text=t,hit_generation_limit=generation_limit(tokens,c['max_new_tokens'],proc.tokenizer.eos_token_id)))
        except Exception as e:
            for r in part:
                row=dict(page_id=r['page_id'],region_id=r['region_id'],text='',error=f'{type(e).__name__}: {e}');preds.append(row);failures.append(row)
        if i%40==0:print(json.dumps(dict(inference=out.name,done=min(i+len(part),len(rows)),total=len(rows),seconds=time.perf_counter()-start)),flush=True)
        save(out,dict(predictions=preds,failures=failures,seconds=time.perf_counter()-start,complete=len(preds)==len(rows)))
    proc.tokenizer.padding_side='right'
    if failures:raise RuntimeError(f'{len(failures)} inference failures; saved with denominator preserved')
    return [p['text'] for p in preds],dict(seconds=time.perf_counter()-start,failures=failures)


def data(c,split):
    rows=readrows(Path(c['dataset'])/f'recognition_candidates.{split}.jsonl')
    verify_crops(rows,c['dataset'])
    if any(r['type_suggestion']!='text' or r.get('hold_reasons') for r in rows):raise ValueError('invalid training scope')
    return rows


def train(c,out,smoke=False):
    import numpy as np
    import torch
    from peft import get_peft_model,LoraConfig,TaskType,PeftModel
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    model,proc,info=load(c)
    if c.get('initial_adapter'):
        if sha256(Path(c['initial_adapter'])/'adapter_model.safetensors')!=c['initial_adapter_sha256']:raise ValueError('initial adapter drift')
        model=PeftModel.from_pretrained(model,c['initial_adapter'],is_trainable=True)
    trainrows=data(c,'train');val=data(c,'validation')
    if smoke:trainrows=trainrows[:8];val=val[:4]
    quarantined=set(c.get('quarantined_validation_pages',{}))
    def val_metrics(texts):
        return summarize([(r['text'],p) for r,p in zip(val,texts) if r['page_id'] not in quarantined])
    save(out/'run-receipt.json',dict(config=c,smoke=smoke,source_sha256=sha256(__file__),strict_load=info,
        model_files_sha256={p.name:sha256(p) for p in Path(c['model']).iterdir() if p.is_file() and p.suffix in ['.json','.safetensors','.jinja']},
        dataset_manifest_sha256=sha256(Path(c['dataset'])/'manifest.json'),splits_sha256={s:sha256(Path(c['dataset'])/f'recognition_candidates.{s}.jsonl') for s in ['train','validation']},
        versions=dict(torch=torch.__version__,transformers=__import__('transformers').__version__,peft=__import__('peft').__version__),test_status=c['test_status']))
    before,rt=infer(model,proc,val,c,out/'before.validation.predictions.json')
    save(out/'before.validation.metrics.json',dict(metrics=val_metrics(before),full_validation_diagnostic=summarize([(r['text'],p) for r,p in zip(val,before)]),runtime=rt))
    targets=target_modules([n for n,_ in model.named_modules()])
    if not c.get('initial_adapter'):
        model=get_peft_model(model,LoraConfig(r=c['rank'],lora_alpha=c['alpha'],lora_dropout=c['dropout'],target_modules=targets,bias='none',task_type=TaskType.CAUSAL_LM))
    # Language-only attention adapters; no base/vision/projector/lm-head training.
    params=[p for p in model.parameters() if p.requires_grad]
    save(out/'lora-parameters.json',dict(targets=targets,trainable=sum(p.numel() for p in params),total=sum(p.numel() for p in model.parameters()),names=[n for n,p in model.named_parameters() if p.requires_grad]))
    opt=torch.optim.AdamW(params,lr=c['learning_rate'],weight_decay=c['weight_decay'])
    initial={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    history=[];step=0;best=float('inf')
    epochs=1 if smoke else c['epochs']
    for epoch in range(1,epochs+1):
        model.train();proc.tokenizer.padding_side='right';order=list(range(len(trainrows)));random.shuffle(order);losses=[];opt.zero_grad();start=time.perf_counter()
        batches=math.ceil(len(order)/c['batch_size'])
        for b,i in enumerate(range(0,len(order),c['batch_size'])):
            rows=[trainrows[k] for k in order[i:i+c['batch_size']]]
            batch=encode(proc,rows,c,True).to(c['device'])
            outputs=model(**batch,use_cache=False);value=float(outputs.loss.detach())
            if not math.isfinite(value):raise ValueError('nonfinite loss')
            # Last partial accumulation window is normalized by its actual size.
            group_size=min(c['accumulation'],batches-(b//c['accumulation'])*c['accumulation'])
            (outputs.loss/group_size).backward();losses.append(value)
            if (b+1)%c['accumulation']==0 or b+1==batches:
                norm=torch.nn.utils.clip_grad_norm_(params,1.)
                if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
                opt.step();opt.zero_grad();step+=1
                if step%10==0:print(json.dumps(dict(epoch=epoch,step=step,loss=value,elapsed=time.perf_counter()-start,gpu_peak_bytes=torch.cuda.max_memory_allocated())),flush=True)
            del outputs,batch
        model.save_pretrained(out/'latest-adapter');torch.save(dict(optimizer=opt.state_dict(),epoch=epoch,step=step,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()),out/'optimizer.pt')
        texts,rt=infer(model,proc,val,c,out/f'epoch-{epoch}.validation.predictions.json')
        metrics=val_metrics(texts);record=dict(epoch=epoch,step=step,mean_train_loss=sum(losses)/len(losses),validation=metrics,full_validation_diagnostic=summarize([(r['text'],p) for r,p in zip(val,texts)]),seconds=time.perf_counter()-start)
        history.append(record);save(out/'history.json',history);print(json.dumps(record),flush=True)
        if metrics['cer_micro']<best:
            best=metrics['cer_micro'];model.save_pretrained(out/'best-adapter');proc.save_pretrained(out/'processor')
            save(out/'selection.json',dict(criterion='minimum full validation micro CER',epoch=epoch,step=step,cer=best,test_used_for_selection=False,adapter_sha256=sha256(out/'best-adapter/adapter_model.safetensors')))
    changed=max(float((p-initial[n]).detach().abs().max()) for n,p in model.named_parameters() if p.requires_grad)
    if changed<=0:raise ValueError('no actual adapter update')
    # Reload the chosen adapter into a fresh model object and compare generation.
    model.load_adapter(out/'best-adapter',adapter_name='selected');model.set_adapter('selected')
    texts,_=infer(model,proc,val[:4],c,out/'selected.reload-reference.json')
    del model,opt,params,initial;__import__('gc').collect();torch.cuda.empty_cache()
    base,proc,_=load(c);reloaded=PeftModel.from_pretrained(base,out/'best-adapter');again,_=infer(reloaded,proc,val[:4],c,out/'selected.reloaded.json')
    if texts!=again:raise ValueError('adapter reload inference mismatch')
    save(out/'verification.json',dict(full_train=not smoke,train_samples=len(trainrows),validation_samples=len(val),adapter_max_weight_update=changed,adapter_reload_texts_equal=True,comparison_samples=4,finite_loss=True))


def final(c,out):
    import torch
    from peft import PeftModel
    trained=Path(c['output'])/'train'
    receipt=json.loads((trained/'run-receipt.json').read_text())
    if receipt['config']!=c:raise ValueError('training/evaluation config mismatch')
    if receipt['dataset_manifest_sha256']!=sha256(Path(c['dataset'])/'manifest.json'):raise ValueError('training dataset changed')
    for split,digest in receipt['splits_sha256'].items():
        if sha256(Path(c['dataset'])/f'recognition_candidates.{split}.jsonl')!=digest:raise ValueError('training split changed')
    selection=json.loads((trained/'selection.json').read_text())
    adapter_hash=sha256(trained/'best-adapter/adapter_model.safetensors')
    if selection.get('adapter_sha256',adapter_hash)!=adapter_hash:raise ValueError('selected adapter drift')
    for name,digest in receipt['model_files_sha256'].items():
        if sha256(Path(c['model'])/name)!=digest:raise ValueError('base model changed')
    rows=data(c,'test');model,proc,_=load(c)
    save(out/'final-lock.json',dict(config=c,source_sha256=sha256(__file__),selection=selection,test_status=c['test_status'],dataset_manifest_sha256=sha256(Path(c['dataset'])/'manifest.json'),adapter_sha256=adapter_hash,adapter_hash_bound_at='selection' if 'adapter_sha256' in selection else 'final pre-test lock (this run predates selection-hash guard)'))
    before,bt=infer(model,proc,rows,c,out/'before.test.predictions.json')
    model=PeftModel.from_pretrained(model,trained/'best-adapter')
    after,at=infer(model,proc,rows,c,out/'after.test.predictions.json')
    result=compare(rows,before,after);result.update(before_runtime=bt,after_runtime=at,test_status=c['test_status'],config=c)
    save(out/'comparison.test.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['cases','strata','config']},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/ocr_v2_vl_lora.json');p.add_argument('--stage',choices=['smoke','train','final'],required=True);a=p.parse_args()
    c=json.loads(Path(a.config).read_text());out=Path(c['output'])/a.stage;out.mkdir(parents=True,exist_ok=False)
    if a.stage=='final':final(c,out)
    else:train(c,out,a.stage=='smoke')

if __name__=='__main__':main()
