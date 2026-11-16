"""Two predeclared recognition-only LoRA objectives. No detector/crop/GT changes."""
import argparse,copy,gc,json,math,random,re,time,platform,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from src.evaluation.ocr_page_audit import read,write_new,sha256,ADAPTER,EXPECTED
from src.training.ocr_vl_lora import encode,load,generation_limit
from src.training.ocr_clean_vision import frozen_digest
from src.training.ocr_finetune import save,readrows,verify_crops
from src.evaluation.ocr_v2 import summarize
from src.evaluation.ocr_page_audit_infer import tensor_receipt

TRAIN=Path('data/processed/ocr_v2/train-audit-round3/clean-train.json')
BASECFG=Path('configs/ocr_v2_vl_lora.json')
QUARANTINE={'hf-0019','hf-0095'}

def language(text):
    korean=bool(re.search('[가-힣ᄀ-ᇿㄱ-ㅣ]',text));latin=bool(re.search('[A-Za-z]',text))
    return 'mixed' if korean and latin else 'korean' if korean else 'latin' if latin else 'other'

def objective(student_logits,labels,weights,teacher_logits=None,kl_weight=0.):
    import torch
    import torch.nn.functional as F
    if student_logits.ndim!=2 or labels.ndim!=1 or student_logits.shape[0]!=labels.numel() or not labels.numel():raise ValueError('invalid completion logits/labels')
    if weights.shape!=labels.shape or (weights<=0).any():raise ValueError('invalid token weights')
    logits=student_logits.float();per=F.cross_entropy(logits,labels,reduction='none');ce=(per*weights).sum()/weights.sum()
    kl=logits.sum()*0
    if teacher_logits is not None:
        if teacher_logits.shape!=student_logits.shape:raise ValueError('teacher alignment mismatch')
        teacher=F.log_softmax(teacher_logits.detach().float(),dim=-1);student=F.log_softmax(logits,dim=-1)
        kl=F.kl_div(student,teacher,reduction='batchmean',log_target=True)
    return ce+kl_weight*kl,ce,kl

def selected_tokens(logits,labels):
    mask=labels[:,1:]!=-100
    return logits[:,:-1][mask],labels[:,1:][mask]

def select_candidate(parent,candidates):
    valid=[]
    for row in candidates:
        m=row['metrics'];p=parent['metrics']
        latin=m['strata']['latin_including_mixed']['cer_micro'];oldlatin=p['strata']['latin_including_mixed']['cer_micro']
        korean=m['strata']['korean']['cer_micro'];oldkorean=p['strata']['korean']['cer_micro']
        passes=(not row.get('failures') and m['all']['cer_micro']<p['all']['cer_micro'] and m['all']['cer_no_space_micro']<=p['all']['cer_no_space_micro'] and (latin is None or oldlatin is None or latin<=oldlatin) and (korean is None or oldkorean is None or korean<=oldkorean+.005))
        if passes:valid.append(row)
    return min(valid,key=lambda x:x['metrics']['all']['cer_micro']) if valid else parent

def score(rows,texts):
    if len(rows)!=len(texts):raise ValueError('denominator mismatch')
    metrics=dict(all=summarize([(r['text'],t) for r,t in zip(rows,texts)]),strata={})
    for name in ['korean','mixed','latin','other','latin_including_mixed']:
        pairs=[(r['text'],t) for r,t in zip(rows,texts) if language(r['text'])==name or name=='latin_including_mixed' and language(r['text']) in ['mixed','latin']]
        metrics['strata'][name]=summarize(pairs)
    return metrics

def verify(root):
    lock=read(root/'input-lock.json')
    for p,h in lock['hashes'].items():
        if sha256(p)!=h:raise ValueError('frozen input drift '+p)
    if sha256(root/'plan.json')!=lock['plan_sha256']:raise ValueError('plan drift')
    for file,key in [('train-rows.json','train_rows_sha256'),('validation-rows.json','validation_rows_sha256')]:
        if sha256(root/file)!=lock[key]:raise ValueError('row drift '+file)
    return read(root/'plan.json')

def initialize(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False);c=read(BASECFG)
    train=read(TRAIN);validation=[r for r in readrows(Path(c['dataset'])/'recognition_candidates.validation.jsonl') if r['page_id'] not in QUARANTINE]
    validation.sort(key=lambda r:(r['page_id'],r['region_id']))
    if len(train)!=2341 or len(validation)!=244:raise ValueError('unexpected frozen dataset')
    if {r['page_id'] for r in train}&{r['page_id'] for r in validation}:raise ValueError('train/validation leakage')
    verify_crops(train,c['dataset']);verify_crops(validation,c['dataset'])
    if sha256(ADAPTER/'adapter_model.safetensors')!=EXPECTED:raise ValueError('parent drift')
    plan=dict(parent=str(ADAPTER),parent_sha256=EXPECTED,base_config=c,training=dict(epochs=2,learning_rate=1e-5,batch_size=2,accumulation=4,seed=42,weight_decay=.01,gradient_clip=1.,latin_token_weight=1.5,warmup_steps=10,schedule='linear warmup then cosine decay to 10% initial LR'),arms=[dict(name='latin_weighted',KL_weight=0.),dict(name='latin_weighted_parent_KL',KL_weight=.1)],selection='parent included; strictly lower validation244 singleton CER, no-space CER non-increase, Latin-including-mixed CER non-increase, Korean-only CER <= parent+0.5 percentage point; select minimum passing CER; no test selection',inference=dict(batch_size=1,seed_per_input=42,dtype='bfloat16',attention='sdpa',padding='left',prompt=c['prompt'],min_pixels=c['min_pixels'],max_pixels=c['max_pixels'],max_new_tokens=c['max_new_tokens']),scope='recognizer language attention LoRA weights only; existing clean train/crops/labels/splits unchanged; no human review claimed; no detector/crop/order/postprocessing modification',train_regions=len(train),validation_regions=len(validation),train_strata=dict(Counter(language(r['text']) for r in train)),validation_strata=dict(Counter(language(r['text']) for r in validation)),test_policy='only after validation selection; reused235 is diagnostic, not independent test; no successful candidate means no new test inference')
    write_new(root/'plan.json',plan);write_new(root/'train-rows.json',train);write_new(root/'validation-rows.json',validation)
    paths=[BASECFG,TRAIN,Path(c['dataset'])/'manifest.json',Path(c['dataset'])/'splits.json',Path(c['dataset'])/'recognition_candidates.validation.jsonl',*ADAPTER.glob('*'),*[p for p in Path(c['model']).iterdir() if p.is_file()]]
    paths += list(Path('src/cv').glob('*.py'))+list(Path('src/evaluation').glob('*.py'))+[Path(__file__)]
    paths += [Path(c['dataset'])/r['image'] for r in train+validation]
    write_new(root/'input-lock.json',dict(hashes={str(p):sha256(p) for p in paths if p.is_file()},plan_sha256=sha256(root/'plan.json'),train_rows_sha256=sha256(root/'train-rows.json'),validation_rows_sha256=sha256(root/'validation-rows.json')))
    (root/'ocr_model_only.executed.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(plan,ensure_ascii=False),flush=True)

def evaluate(model,proc,rows,c,out):
    import torch
    out=Path(out);out.mkdir(exist_ok=False);start=time.perf_counter();predictions=[];failures=[];receipts=[]
    model.eval();proc.tokenizer.padding_side='left'
    for i,r in enumerate(rows):
        torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        try:
            b=encode(proc,[r],c,False)
            if i<4:receipts.append(dict(page_id=r['page_id'],region_id=r['region_id'],tensors=tensor_receipt(b)))
            b=b.to(c['device'])
            with torch.inference_mode():ids=model.generate(**b,do_sample=False,num_beams=1,max_new_tokens=c['max_new_tokens'],use_cache=True)
            generated=ids[0,b['input_ids'].shape[1]:];text=proc.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            predictions.append(dict(page_id=r['page_id'],region_id=r['region_id'],text=text,hit_generation_limit=generation_limit(generated,c['max_new_tokens'],proc.tokenizer.eos_token_id)))
            del b,ids
        except Exception as e:
            row=dict(page_id=r['page_id'],region_id=r['region_id'],text='',error=f'{type(e).__name__}: {e}');predictions.append(row);failures.append(row)
        if (i+1)%40==0:print(json.dumps(dict(evaluate=str(out),done=i+1,total=len(rows),seconds=time.perf_counter()-start)),flush=True)
    result=dict(predictions=predictions,failures=failures,complete=True,seconds=time.perf_counter()-start)
    write_new(out/'predictions.json',result);write_new(out/'tensor-receipts.json',receipts)
    m=score(rows,[r['text'] for r in predictions]);write_new(out/'metrics.json',m)
    return m,result

def load_parent(c,trainable):
    from peft import PeftModel
    base,proc,info=load(c);model=PeftModel.from_pretrained(base,ADAPTER,is_trainable=trainable)
    return model,proc,info

def baseline(root):
    import torch
    root=Path(root);plan=verify(root);c=plan['base_config'];model,proc,info=load_parent(c,False)
    metrics,pred=evaluate(model,proc,read(root/'validation-rows.json'),c,root/'parent-validation')
    if pred['failures']:raise RuntimeError('parent inference failed')
    repeat,_=evaluate(model,proc,read(root/'validation-rows.json')[:4],c,root/'parent-repeat')
    a=read(root/'parent-validation/predictions.json')['predictions'][:4];b=read(root/'parent-repeat/predictions.json')['predictions']
    if a!=b:raise ValueError('parent repeat differs')
    write_new(root/'parent.json',dict(name='parent',checkpoint=str(ADAPTER),metrics=metrics,failures=[],strict_load=info,active_adapters=list(model.active_adapters),dtype=str(next(model.parameters()).dtype),torch=torch.__version__,transformers=__import__('transformers').__version__,peft=__import__('peft').__version__,gpu=torch.cuda.get_device_name(0),repeat_equal=True))
    verify(root)

def train_arm(root,name,smoke=False):
    import torch
    import torch.nn.functional as F
    root=Path(root);plan=verify(root);settings=plan['training'];arm=next(a for a in plan['arms'] if a['name']==name);c=plan['base_config'];out=root/(('smoke-' if smoke else '')+name);out.mkdir(exist_ok=False)
    random.seed(settings['seed']);torch.manual_seed(settings['seed']);torch.cuda.manual_seed_all(settings['seed']);model,proc,info=load_parent(c,True)
    train=read(root/'train-rows.json');val=read(root/'validation-rows.json')
    if smoke:train=train[:8];val=val[:4]
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    if not params or any('.language_model.' not in n or '.lora_' not in n for n in params):raise ValueError('non-recognition parameter training')
    initial={n:p.detach().clone() for n,p in params.items()};frozen_before=frozen_digest(model)
    teacher=None
    if arm['KL_weight']:
        teacher,teacherproc,_=load_parent(c,False);teacher.eval()
        if any(p.requires_grad for p in teacher.parameters()):raise ValueError('teacher trainable')
    token_weights={};opt=torch.optim.AdamW(list(params.values()),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    epochs=1 if smoke else settings['epochs'];batches=math.ceil(len(train)/settings['batch_size']);steps_per_epoch=math.ceil(batches/settings['accumulation']);total_steps=steps_per_epoch*epochs;step=0;history=[]
    write_new(out/'start-receipt.json',dict(arm=arm,settings=settings,train_samples=len(train),validation_samples=len(val),planned_steps=total_steps,trainable_names=list(params),trainable_parameters=sum(p.numel() for p in params.values()),frozen_before=frozen_before,strict_load=info,active_adapters=list(model.active_adapters),plan_sha256=sha256(root/'plan.json'),source_sha256=sha256(__file__),teacher='frozen same initial parent; train-only teacher-forced logits' if teacher else None))
    for epoch in range(1,epochs+1):
        model.train();proc.tokenizer.padding_side='right';order=list(range(len(train)));random.Random(settings['seed']+epoch).shuffle(order);opt.zero_grad();losses=[];kls=[];t0=time.perf_counter()
        for b,i in enumerate(range(0,len(order),settings['batch_size'])):
            rows=[train[k] for k in order[i:i+settings['batch_size']]];batch=encode(proc,rows,c,True).to(c['device']);labels=batch.pop('labels')
            mask=labels[:,1:]!=-100;target=labels[:,1:][mask]
            weights=[]
            for id in target.tolist():
                if id not in token_weights:token_weights[id]=settings['latin_token_weight'] if re.search('[A-Za-z]',proc.tokenizer.decode([id],skip_special_tokens=True,clean_up_tokenization_spaces=False)) else 1.
                weights.append(token_weights[id])
            weights=torch.tensor(weights,device=target.device)
            tl=None
            if teacher:
                with torch.no_grad():
                    output=teacher(**batch,use_cache=False);tl=output.logits[:,:-1][mask].detach();del output
            output=model(**batch,use_cache=False);sl=output.logits[:,:-1][mask];loss,ce,kl=objective(sl,target,weights,tl,arm['KL_weight'])
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            group=min(settings['accumulation'],batches-(b//settings['accumulation'])*settings['accumulation']);(loss/group).backward();losses.append(float(ce.detach()));kls.append(float(kl.detach()))
            if (b+1)%settings['accumulation']==0 or b+1==batches:
                step+=1;warmup=settings['warmup_steps'];factor=min(1.,step/warmup) if step<=warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
                for pg in opt.param_groups:pg['lr']=settings['learning_rate']*factor
                grad=torch.nn.utils.clip_grad_norm_(list(params.values()),settings['gradient_clip'])
                if not torch.isfinite(grad):raise ValueError('nonfinite gradient')
                opt.step();opt.zero_grad()
                if step%40==0:print(json.dumps(dict(arm=name,epoch=epoch,step=step,CE=losses[-1],KL=kls[-1],lr=opt.param_groups[0]['lr'],seconds=time.perf_counter()-t0)),flush=True)
            del output,batch,labels,mask,target,weights,tl,sl,loss,ce,kl
        checkpoint=out/f'epoch-{epoch}-adapter';model.save_pretrained(checkpoint)
        torch.save(dict(optimizer=opt.state_dict(),step=step,epoch=epoch,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),out/f'epoch-{epoch}-optimizer.pt')
        metrics,pred=evaluate(model,proc,val,c,out/f'epoch-{epoch}-validation')
        row=dict(name=name+f'-epoch-{epoch}',checkpoint=str(checkpoint),metrics=metrics,failures=pred['failures'],epoch=epoch,step=step,mean_CE=sum(losses)/len(losses),mean_KL=sum(kls)/len(kls),seconds=time.perf_counter()-t0)
        history.append(row);write_new(out/f'epoch-{epoch}-result.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)
    changed=max(float((p-initial[n]).detach().abs().max()) for n,p in params.items());frozen_after=frozen_digest(model)
    if changed<=0 or frozen_before!=frozen_after:raise ValueError('weight update/freeze verification failed')
    last_expected=read(out/f'epoch-{epochs}-validation/predictions.json')['predictions'][:4]
    del model,opt,params,initial,teacher;gc.collect();torch.cuda.empty_cache()
    from peft import PeftModel
    base,proc,_=load(c);model=PeftModel.from_pretrained(base,out/f'epoch-{epochs}-adapter',is_trainable=False)
    _,rep=evaluate(model,proc,val[:4],c,out/'fresh-reload')
    if rep['predictions']!=last_expected:raise ValueError('fresh reload differs')
    write_new(out/'verification.json',dict(adapter_max_weight_change=changed,frozen_parameters_unchanged=True,frozen_after=frozen_after,fresh_reload_equal=True,optimizer_steps=step,smoke=smoke,training_regions=len(train),gpu_peak_bytes=torch.cuda.max_memory_allocated(),teacher_frozen=bool(arm['KL_weight'])))
    write_new(out/'history.json',history);verify(root)

def select(root):
    root=Path(root);plan=verify(root);parent=read(root/'parent.json');candidates=[read(root/arm['name']/f'epoch-{epoch}-result.json') for arm in plan['arms'] for epoch in range(1,plan['training']['epochs']+1)]
    chosen=select_candidate(parent,candidates)
    write_new(root/'selection.json',dict(chosen=chosen,parent=parent,candidates=candidates,policy=plan['selection'],test_used=False,parent_overwritten=False))
    print(json.dumps(dict(chosen=chosen['name'],parent_CER=parent['metrics']['all']['cer_micro'],chosen_CER=chosen['metrics']['all']['cer_micro']),ensure_ascii=False))

def final(root):
    import torch
    from peft import PeftModel
    root=Path(root);plan=verify(root);selection=read(root/'selection.json');chosen=selection['chosen'];c=plan['base_config']
    if chosen['name']=='parent':
        write_new(root/'final-not-run.json',dict(reason='no validation candidate passed; retain parent; no redundant test inference',model=str(ADAPTER)));return
    rows=readrows(Path(c['dataset'])/'recognition_candidates.test.jsonl');rows.sort(key=lambda r:(r['page_id'],r['region_id']));verify_crops(rows,c['dataset'])
    model,proc,_=load_parent(c,False);pm,pr=evaluate(model,proc,rows,c,root/'diagnostic235-parent');del model;gc.collect();torch.cuda.empty_cache()
    base,proc,_=load(c);model=PeftModel.from_pretrained(base,chosen['checkpoint'],is_trainable=False);cm,cr=evaluate(model,proc,rows,c,root/'diagnostic235-chosen')
    write_new(root/'diagnostic235.json',dict(parent=pm,chosen=cm,chosen_name=chosen['name'],parent_failures=pr['failures'],chosen_failures=cr['failures'],scope='reused exploratory235; not independent generalization; no reselection from these scores'))
    verify(root)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['prepare','baseline','smoke','train','select','final'],required=True);p.add_argument('--arm',choices=['latin_weighted','latin_weighted_parent_KL']);args=p.parse_args()
    if args.stage=='prepare':initialize(args.root)
    elif args.stage=='baseline':baseline(Path(args.root))
    elif args.stage in ['smoke','train']:train_arm(Path(args.root),args.arm,args.stage=='smoke')
    elif args.stage=='select':select(Path(args.root))
    else:final(Path(args.root))
