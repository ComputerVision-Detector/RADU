"""Bind trained candidate checkpoints and verify selected adapter after fresh reload."""
import argparse,gc
from pathlib import Path
from src.training.ocr_model_only import read,write_new,sha256,verify,load,evaluate

def freeze(root):
    root=Path(root);plan=verify(root);paths=[]
    for arm in plan['arms']:
        for epoch in range(1,plan['training']['epochs']+1):paths.extend((root/arm['name']/f'epoch-{epoch}-adapter').glob('*'))
    paths=[p for p in paths if p.is_file()];write_new(root/'trained-candidate-lock.json',dict(hashes={str(p):sha256(p) for p in paths}))

def check(root):
    root=Path(root);plan=verify(root)
    for p,h in read(root/'trained-candidate-lock.json')['hashes'].items():
        if sha256(p)!=h:raise ValueError('trained candidate drift '+p)
    chosen=read(root/'selection.json')['chosen']
    if chosen['name']=='parent':write_new(root/'selected-verification.json',dict(parent_retained=True,original_lock_verified=True));return
    from peft import PeftModel
    base,proc,info=load(plan['base_config']);model=PeftModel.from_pretrained(base,chosen['checkpoint'],is_trainable=False)
    rows=read(root/'validation-rows.json')[:4];_,pred=evaluate(model,proc,rows,plan['base_config'],root/'selected-fresh-validation4')
    expected=read(Path(chosen['checkpoint']).parent/f'epoch-{chosen["epoch"]}-validation/predictions.json')['predictions'][:4]
    if pred['predictions']!=expected:raise ValueError('chosen reload output drift')
    write_new(root/'selected-verification.json',dict(parent_retained=False,checkpoint=chosen['checkpoint'],adapter_sha256=sha256(Path(chosen['checkpoint'])/'adapter_model.safetensors'),active_adapters=list(model.active_adapters),eval_mode=not model.training,all_parameters_frozen=not any(p.requires_grad for p in model.parameters()),fresh_validation4_equal=True,strict_load=info))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['freeze','check'],required=True);a=p.parse_args();{'freeze':freeze,'check':check}[a.stage](a.root)
