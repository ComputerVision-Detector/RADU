"""Conditional localization: first-token divergent case, two existing contexts only."""
import argparse, hashlib, inspect, time
from pathlib import Path
from src.evaluation.ocr_batch_contract import verify,read,write_new,load,encode,ADAPTER,tensor_receipt

def probe(root, output):
    import torch
    from peft import PeftModel
    root=Path(root);lock=verify(root);plan=read(root/'plan.json');cmp=read(root/'comparisons.json')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    target=next(x['target'] for x in cmp if x['condition']=='compact244' and x['first_branch_token']==0)
    schedules=[next(m for m in plan['matrix'] if m['target']==target and m['condition']==c) for c in ['original297','compact244']]
    write_new(out/'probe-plan.json',dict(trigger='all target unpadded inputs equal, logits differ; select first compact244 first-token divergence in frozen target order',target=target,schedules=schedules,max_processed_crops=sum(len(m['members']) for m in schedules),settings='unchanged; max_new_tokens=1 for first forward localization only; not accuracy comparison'))
    model,proc,info=load(lock['config']);model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False);model.eval();proc.tokenizer.padding_side='left'
    modules=dict(model.named_modules());suffixes=['visual.vision_model.encoder.layers.0.layer_norm2','visual.vision_model.encoder.layers.0.mlp.fc1','visual.vision_model.encoder.layers.0.mlp.fc2','visual.vision_model.encoder.layers.0.layer_norm1','visual.vision_model.encoder.layers.0.self_attn.q_proj','visual.vision_model.encoder.layers.0.self_attn.k_proj','visual.vision_model.encoder.layers.0.self_attn.v_proj','visual.vision_model.encoder.layers.0.self_attn.out_proj','visual.vision_model.embeddings','visual.vision_model.encoder.layers.0','visual','mlp_AR','model.layers.0']
    names={s:next((n for n in modules if n.endswith('.'+s)),None) for s in suffixes}
    collected=[];ctx={};hooks=[]
    def hook(name):
        def capture(module,args,output):
            value=output.last_hidden_state if hasattr(output,'last_hidden_state') else output
            if isinstance(value,(list,tuple)):
                if name in ['visual','mlp_AR']:value=value[ctx['target']]
                else:value=value[0]
            if not isinstance(value,torch.Tensor):return
            if name.startswith('visual') and value.ndim==2:
                value=value[ctx['offset']:ctx['offset']+ctx['size']]
            if value.ndim==3:
                if name.startswith('visual.vision_model'):value=value[0,ctx['offset']:ctx['offset']+ctx['size']]
                elif value.shape[0]==ctx['batchsize']:value=value[ctx['target']][ctx['mask']]
            ctx['captured'][name]=value.detach().float().cpu()
        return capture
    for s,n in names.items():
        if n:hooks.append(modules[n].register_forward_hook(hook(s)))
    for schedule in schedules:
        ids=schedule['members'];rows=[plan['rows'][i] for i in ids];b=encode(proc,rows,lock['config'],False);idx=ids.index(target);sizes=[int(x.prod()) for x in b['image_grid_thw']]
        ctx.update(target=idx,offset=sum(sizes[:idx]),size=sizes[idx],batchsize=len(ids),mask=b['attention_mask'][idx].bool().to(lock['config']['device']),captured={})
        torch.manual_seed(lock['seed']);torch.cuda.manual_seed_all(lock['seed']);b=b.to(lock['config']['device'])
        with torch.inference_mode():g=model.generate(**b,do_sample=False,num_beams=1,max_new_tokens=1,use_cache=True,return_dict_in_generate=True,output_scores=True)
        ctx['captured']['first_logits']=g.scores[0][idx].detach().float().cpu();collected.append(ctx['captured']);del b,g
    for h in hooks:h.remove()
    report={}
    for name in sorted(collected[0].keys() & collected[1].keys()):
        a,b=collected[0][name],collected[1][name];same_shape=a.shape==b.shape
        report[name]=dict(shape0=list(a.shape),shape1=list(b.shape),finite0=bool(torch.isfinite(a).all()),finite1=bool(torch.isfinite(b).all()),equal=bool(torch.equal(a,b)),max_abs=float((a-b).abs().max()) if same_shape else None,rms=float((a-b).square().mean().sqrt()) if same_shape else None,hash0=hashlib.sha256(a.numpy().tobytes()).hexdigest(),hash1=hashlib.sha256(b.numpy().tobytes()).hexdigest())
    write_new(out/'probe-result.json',dict(target=target,module_names=names,module_implementation={s:dict(class_name=type(modules[n]).__name__,file=inspect.getfile(type(modules[n]))) for s,n in names.items() if n},stages=report,processed_crops=sum(len(m['members']) for m in schedules),dtype='bfloat16',attention='sdpa',model_frozen=True,settings_changed=False))
    verify(root)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);args=p.parse_args();probe(args.root,args.output)
