"""Finite, predeclared batch-context diagnosis; fixed selected language adapter."""
import argparse, copy, hashlib, json, platform, time
from pathlib import Path
from PIL import Image
from src.evaluation.ocr_page_audit import read, write_new, sha256, ADAPTER, EXPECTED
from src.training.ocr_vl_lora import encode, load
from src.evaluation.ocr_page_audit_infer import tensor_receipt

BOUND=Path('data/experiments/ocr_page_audit_boundary_v1')
DATA=Path('data/processed/ocr_v2/pilot-v2')

def key(r):return r['page_id']+'/'+r['region_id']
def rows297():
    rows=[json.loads(x) for x in (DATA/'recognition_candidates.validation.jsonl').read_text().splitlines()]
    for r in rows:r['image']=str((DATA/r['image']).resolve())
    return rows

def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    c=read('configs/ocr_v2_vl_lora.json');old=rows297();new=read(BOUND/'validation.before.rows.json')
    legacy=read('data/experiments/ocr_v2_vl_lora/train/epoch-3.validation.predictions.json')['predictions']
    recent=read(BOUND/'validation/before.json')['predictions']
    before={key(r):r['text'] for r in legacy};after={key(r):r['text'] for r in recent}
    changed=[key(r) for r in new if before[key(r)]!=after[key(r)]]
    if len(changed)!=13:raise ValueError(f'historical changed count {len(changed)} !=13')
    conv=next(r for r in old if r['image'].endswith('hf-0147-0007.png'))
    control=next(r for r in new if key(r) not in changed and key(r)!=key(conv))
    targets=list(dict.fromkeys(changed+[key(conv),key(control)]))
    lookup={key(r):r for r in old};companions=[key(r) for r in old if key(r) not in targets][:3]
    matrix=[]
    def context(rows,id):
        i=next(i for i,r in enumerate(rows) if key(r)==id);return [key(r) for r in rows[(i//4)*4:(i//4)*4+4]]
    for id in targets:
        original=context(old,id);compact=context(new,id)
        for label,members in [('original297',original),('original297_repeat',original),('compact244',compact),('single',[id]),('single_repeat',[id]),('changed_members',[id]+companions),('reordered',list(reversed(original)))]:
            matrix.append(dict(target=id,condition=label,members=members))
    paths=[Path('configs/ocr_v2_vl_lora.json'),*ADAPTER.glob('*'),*[p for p in Path(c['model']).iterdir() if p.is_file()]]
    used={i for m in matrix for i in m['members']}
    paths += [Path(lookup[i]['image']) for i in used]
    paths += [BOUND/'validation.before.rows.json',BOUND/'validation/before.json',Path('data/experiments/ocr_v2_vl_lora/train/epoch-3.validation.predictions.json'),DATA/'recognition_candidates.validation.jsonl']
    if sha256(ADAPTER/'adapter_model.safetensors')!=EXPECTED:raise ValueError('adapter drift')
    pixels={}
    for id in used:
        with Image.open(lookup[id]['image']) as im:
            im=im.convert('RGB');pixels[id]=dict(size=list(im.size),decoded_rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest(),file_sha256=sha256(lookup[id]['image']))
    write_new(root/'plan.json',dict(targets=targets,changed13=changed,conv2=key(conv),control=key(control),companions=companions,matrix=matrix,rows={i:lookup[i] for i in sorted(used)},historical={i:dict(original297=before[i],compact244=after[i]) for i in targets},selection='all historical13 + recorded CONV2 + first unchanged validation control; no output-based new selection',maximum_matrix_rows=len(matrix),maximum_processed_crops=sum(len(m['members']) for m in matrix)))
    write_new(root/'input-lock.json',dict(config=c,hashes={str(p):sha256(p) for p in paths if p.is_file()},pixels=pixels,plan_sha256=sha256(root/'plan.json'),seed=c['seed'],dtype='bfloat16',attention='sdpa',adapter=str(ADAPTER),generation=dict(do_sample=False,num_beams=1,max_new_tokens=128,use_cache=True),padding='left',cache_policy='no external output reuse; batch identity includes ordered members, tensors and config'))

def verify(root):
    lock=read(root/'input-lock.json')
    if lock.get('config')!=read('configs/ocr_v2_vl_lora.json'):raise ValueError('config drift')
    for p,h in lock['hashes'].items():
        if sha256(p)!=h:raise ValueError('input drift '+p)
    if sha256(root/'plan.json')!=lock['plan_sha256']:raise ValueError('plan drift')
    return lock

def run(root):
    import torch
    from peft import PeftModel
    root=Path(root);lock=verify(root);plan=read(root/'plan.json');out=root/'inference';out.mkdir(exist_ok=False)
    c=lock['config'];torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);start=time.perf_counter()
    model,proc,info=load(c);model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False);model.eval();proc.tokenizer.padding_side='left'
    assert not model.training and not any(p.requires_grad for p in model.parameters()) and list(model.active_adapters)==['default']
    results=[];processed=0
    for index,m in enumerate(plan['matrix']):
        rows=[plan['rows'][id] for id in m['members']];torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        batch=encode(proc,rows,c,False);receipt=tensor_receipt(batch);target=m['members'].index(m['target']);grid=batch['image_grid_thw'];sizes=[int(x.prod()) for x in grid];offset=sum(sizes[:target]);pix=batch['pixel_values'][offset:offset+sizes[target]]
        selected=tensor_receipt(dict(pixels=pix,grid=grid[target],tokens=batch['input_ids'][target][batch['attention_mask'][target].bool()]))
        batch=batch.to(c['device'])
        with torch.inference_mode():
            generated=model.generate(**batch,**lock['generation'],return_dict_in_generate=True,output_scores=True)
        ids=generated.sequences[:,batch['input_ids'].shape[1]:];tokens=ids[target].tolist();text=proc.decode(ids[target],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        # Small receipts, never persist full vocabulary logits or hidden tensors.
        score_receipts=[]
        for score in generated.scores:
            s=score[target].detach().float();top=torch.topk(s,2)
            score_receipts.append(dict(sha256=hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest(),finite=bool(torch.isfinite(s).all()),top_ids=top.indices.tolist(),top_scores=top.values.tolist(),margin=float(top.values[0]-top.values[1])))
        item=dict(**m,index=index,text=text,token_ids=tokens,EOS=proc.tokenizer.eos_token_id,hit_limit=len(tokens)>=c['max_new_tokens'] and proc.tokenizer.eos_token_id not in tokens,input_tensors=receipt,target_unpadded=selected,scores=score_receipts,active_adapters=list(model.active_adapters),eval_mode=not model.training)
        write_new(out/f'{index:03d}.json',item);results.append(item);processed+=len(rows)
        del generated,batch,ids
        print(index,m['condition'],m['target'],flush=True)
    verify(root)
    comparisons=[]
    for target in plan['targets']:
        rr={r['condition']:r for r in results if r['target']==target};baseline=rr['original297']
        for condition,r in rr.items():
            a,b=baseline['token_ids'],r['token_ids'];first=next((i for i in range(min(len(a),len(b))) if a[i]!=b[i]),min(len(a),len(b)) if len(a)!=len(b) else None)
            comparisons.append(dict(target=target,condition=condition,text_changed=r['text']!=baseline['text'],first_branch_token=first,unpadded_inputs_equal=r['target_unpadded']==baseline['target_unpadded'],first_logits_equal=r['scores'][0]['sha256']==baseline['scores'][0]['sha256']))
        if rr['original297']['token_ids']!=rr['original297_repeat']['token_ids'] or rr['single']['token_ids']!=rr['single_repeat']['token_ids']:raise ValueError('same schedule repeat unstable')
    write_new(root/'comparisons.json',comparisons)
    write_new(root/'receipt.json',dict(seconds=time.perf_counter()-start,processed_crops=processed,target_evaluations=len(results),unique_targets=len(plan['targets']),target_inputs=len(results),companion_inputs=processed-len(results),versions=dict(python=platform.python_version(),torch=torch.__version__,transformers=__import__('transformers').__version__,peft=__import__('peft').__version__),gpu=torch.cuda.get_device_name(0),strict_load=info,adapter_sha256=sha256(ADAPTER/'adapter_model.safetensors'),optimizer_steps=0,nonfinite_scores=sum(not s['finite'] for r in results for s in r['scores']),torch_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['prepare','infer'],required=True);args=p.parse_args();{'prepare':prepare,'infer':run}[args.stage](args.root)
