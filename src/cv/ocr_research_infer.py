"""Versioned research-only singleton inference; never auto-apply human draft GT."""
import argparse, copy, json
from pathlib import Path
from src.evaluation.ocr_page_audit import read,write_new,sha256,ADAPTER,EXPECTED
from src.training.ocr_vl_lora import load,encode,infer
from src.evaluation.ocr_page_audit_infer import tensor_receipt

def prepare_rows(path,dataset):
    p=Path(path);rows=read(p) if p.suffix=='.json' else [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    ids=set()
    for r in rows:
        key=(r['page_id'],r['region_id'])
        if key in ids:raise ValueError('duplicate region ID')
        ids.add(key);image=Path(r['image']);r['image']=str((image if image.is_absolute() else Path(dataset)/image).resolve())
        if sha256(r['image'])!=r['crop_sha256']:raise ValueError('crop drift')
    return sorted(rows,key=lambda r:(r['page_id'],r['region_id']))

def run(rows_path,output,arm):
    import torch
    from peft import PeftModel
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    c=read('configs/ocr_v2_vl_lora.json');c=copy.deepcopy(c);c['generation_batch_size']=1
    rows=prepare_rows(rows_path,c['dataset'])
    if sha256(ADAPTER/'adapter_model.safetensors')!=EXPECTED:raise ValueError('adapter drift')
    paths=[Path(rows_path),Path('configs/ocr_v2_vl_lora.json'),*[p for p in Path(c['model']).iterdir() if p.is_file()],*[p for p in ADAPTER.iterdir() if p.is_file()]]
    hashes={str(p):sha256(p) for p in paths};schedule=[dict(page_id=r['page_id'],region_id=r['region_id'],crop_sha256=r['crop_sha256']) for r in rows]
    write_new(out/'run-lock.json',dict(protocol='RADU-research-singleton-v1',arm=arm,config=c,hashes=hashes,schedule=schedule,padding='left',dtype='bfloat16',attention='sdpa',seed_per_input=c['seed'],batch_size=1,cache_policy='do not use historical batch4 outputs as singleton outputs',GT_policy='requires separately frozen human-approved GT; drafts never promoted automatically'))
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);model,proc,info=load(c)
    if arm=='selected':model=PeftModel.from_pretrained(model,ADAPTER,is_trainable=False)
    model.eval()
    if any(p.requires_grad for p in model.parameters()):
        # Official base loader initializes requires_grad=True; freeze for research inference.
        for p in model.parameters():p.requires_grad_(False)
    predictions=[];receipts=[]
    for i,r in enumerate(rows):
        torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);proc.tokenizer.padding_side='left'
        receipts.append(dict(page_id=r['page_id'],region_id=r['region_id'],tensors=tensor_receipt(encode(proc,[r],c,False))))
        try:
            texts,_=infer(model,proc,[r],c,out/f'input-{i:05d}.json');predictions.append(dict(page_id=r['page_id'],region_id=r['region_id'],text=texts[0]))
        except Exception as e:predictions.append(dict(page_id=r['page_id'],region_id=r['region_id'],text='',error=str(e)))
    for p,h in hashes.items():
        if sha256(p)!=h:raise ValueError('input/model drift '+p)
    write_new(out/'predictions.json',dict(predictions=predictions,complete=True,failures=[r for r in predictions if r.get('error')]))
    write_new(out/'tensor-receipts.json',receipts)
    write_new(out/'receipt.json',dict(strict_load=info,arm=arm,optimizer_steps=0,processed_crops=len(rows),active_adapters=list(model.active_adapters) if arm=='selected' else [],model_dtype=str(next(model.parameters()).dtype),torch=torch.__version__))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',required=True);p.add_argument('--output',required=True);p.add_argument('--arm',choices=['official','selected'],required=True);args=p.parse_args();run(args.rows,args.output,args.arm)
