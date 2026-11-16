"""Bounded Korean PP-OCRv5 CTC adaptation; GT labels never come from predictions.

The full official multi-head checkpoint is loaded strictly. Only the deployed
backbone + CTC branch is optimized; the unused auxiliary NRTR branch is preserved.
No operational OCR or Qwen module is modified.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import unicodedata

from src.evaluation.ocr_v2 import summarize
from src.ocr_data.audit import sha256


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def readrows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def screen(rows, chars, maximum):
    accepted, held = [], []
    for original in rows:
        row = {k:v for k,v in original.items() if k != 'prediction'}
        text=row['text']; reasons=[]
        if not text.strip(): reasons.append('empty')
        if any(c in text for c in '\n\r\t'): reasons.append('multiline_or_tab')
        unknown=sorted(set(text)-set(chars))
        if unknown: reasons.append('unsupported_characters')
        if len(text)>maximum: reasons.append('maximum_length')
        if reasons: held.append(dict(row, reasons=reasons, unsupported=unknown))
        else: accepted.append(row)
    return accepted, held


def ctc_timesteps(text):
    return len(text)+sum(a==b for a,b in zip(text,text[1:]))


def ctc_forward_train(model,x):
    # Official PPLCNetV3 train forward hardcodes 40 steps (its default 25-char
    # recipes). Use its inference pooling geometry during CTC adaptation so long
    # untouched labels remain alignable. BN/dropout still use training mode.
    import paddle.nn.functional as F
    b=model.backbone
    for layer in [b.conv1,b.blocks2,b.blocks3,b.blocks4,b.blocks5,b.blocks6]:x=layer(x)
    x=F.avg_pool2d(x,[3,2])
    return model.head.ctc_head(model.head.ctc_encoder(x))


def compare(rows, before, after):
    if not len(rows)==len(before)==len(after): raise ValueError('prediction coverage mismatch')
    b=summarize([(r['text'],p) for r,p in zip(rows,before)])
    a=summarize([(r['text'],p) for r,p in zip(rows,after)])
    bv,av=b['cer_micro'],a['cer_micro']
    cases=[dict(page_id=r['page_id'],region_id=r['region_id'],reference=r['text'],before=p,after=q,
                before_errors=summarize([(r['text'],p)])['errors'],after_errors=summarize([(r['text'],q)])['errors'])
           for r,p,q in zip(rows,before,after)]
    strata={}
    for name,fn in [('page',lambda r:r['page_id']),
                    ('language',lambda r:'mixed' if any('가'<=c<='힣' for c in r['text']) and any(c.isascii() and c.isalpha() for c in r['text']) else 'korean' if any('가'<=c<='힣' for c in r['text']) else 'latin_numeric_other'),
                    ('length',lambda r:'1-10' if len(r['text'])<=10 else '11-30' if len(r['text'])<=30 else '31+')]:
        groups=defaultdict(list)
        for i,r in enumerate(rows):groups[fn(r)].append(i)
        strata[name]={g:dict(before=summarize([(rows[i]['text'],before[i]) for i in ix]),after=summarize([(rows[i]['text'],after[i]) for i in ix])) for g,ix in groups.items()}
    # Resample entire pages; positive delta means CER improved. Fixed seed, 2000 draws.
    pages=defaultdict(list)
    for c in cases:pages[c['page_id']].append(c)
    rng=random.Random(20261109);ids=sorted(pages);deltas=[]
    for _ in range(2000):
        sample=[c for p in rng.choices(ids,k=len(ids)) for c in pages[p]]
        units=sum(len(__import__('src.evaluation.ocr_metrics',fromlist=['normalize']).normalize(c['reference'])) for c in sample)
        if units:deltas.append(sum(c['before_errors']-c['after_errors'] for c in sample)/units)
    deltas.sort()
    return dict(before=b,after=a,cer_delta_after_minus_before=av-bv if bv is not None and av is not None else None,
                relative_cer_reduction=(bv-av)/bv if bv else None,pages=len(ids),
                paired_page_bootstrap=dict(draws=2000,seed=20261109,improvement_CI95=[deltas[int(.025*len(deltas))],deltas[min(len(deltas)-1,int(.975*len(deltas)))]] if deltas else None),
                strata=strata,cases=cases)


def bootstrap_model(config):
    import yaml
    import paddle
    sys.path.insert(0,str(Path(config['upstream']).resolve()))
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess.rec_postprocess import CTCLabelDecode
    base=yaml.safe_load(Path(config['official_config']).read_text())
    decoder=CTCLabelDecode(character_dict_path=config['dictionary'],use_space_char=True)
    count=len(decoder.character)
    base['Architecture']['Head']['out_channels_list']={'CTCLabelDecode':count,'SARLabelDecode':count+2,'NRTRLabelDecode':count+3}
    model=build_model(base['Architecture'])
    state=paddle.load(config['pretrained'])
    expected=model.state_dict()
    missing=set(expected)-set(state);extra=set(state)-set(expected)
    mismatch=[k for k in expected.keys() & state.keys() if list(expected[k].shape)!=list(state[k].shape)]
    if missing or extra or mismatch: raise ValueError(dict(missing=sorted(missing),extra=sorted(extra),mismatch=mismatch))
    model.set_state_dict(state)
    return model,decoder,dict(parameters=len(state),all_keys_and_shapes_match=True,ctc_classes=count)


def preprocess(path, width=768, height=48):
    import cv2
    import numpy as np
    im=cv2.imread(str(path))
    if im is None:raise ValueError(f'unreadable crop {path}')
    h,w=im.shape[:2];rw=min(width,math.ceil(height*w/h))
    resized=cv2.resize(im,(rw,height)).astype('float32').transpose(2,0,1)/127.5-1.
    result=np.zeros((3,height,width),dtype='float32');result[:,:,:rw]=resized
    return result


def infer(model, decoder, rows, dataset, config):
    import numpy as np
    import paddle
    model.eval();texts=[];failures=[];started=time.perf_counter()
    with paddle.no_grad():
        for i in range(0,len(rows),config['batch_size']):
            batch=rows[i:i+config['batch_size']]
            try:
                x=np.stack([preprocess(Path(dataset)/r['image'],config['width']) for r in batch])
                decoded=decoder(model(paddle.to_tensor(x)).numpy())
                texts.extend(t[0] for t in decoded)
            except Exception as e:
                # Preserve the denominator and make failures visible.
                texts.extend(['']*len(batch));failures.extend(dict(page_id=r['page_id'],region_id=r['region_id'],error=str(e)) for r in batch)
    return texts,dict(seconds=time.perf_counter()-started,failures=failures,samples=len(rows),batch_size=config['batch_size'])


def export_and_check(model,decoder,rows,dataset,config,out):
    import numpy as np
    import paddle
    model.eval();x=paddle.to_tensor(np.stack([preprocess(Path(dataset)/r['image'],config['width']) for r in rows[:2]]))
    with paddle.no_grad():expected=model(x).numpy()
    prefix=str(Path(out)/'inference')
    static=paddle.jit.to_static(model,input_spec=[paddle.static.InputSpec([None,3,48,config['width']],'float32')],full_graph=True)
    paddle.jit.save(static,prefix)
    reloaded=paddle.jit.load(prefix);reloaded.eval()
    with paddle.no_grad():actual=reloaded(x).numpy()
    error=float(np.max(np.abs(expected-actual)))
    same=[t[0] for t in decoder(expected)]==[t[0] for t in decoder(actual)]
    if not same or error>1e-3:raise ValueError(f'export mismatch {error}')
    return dict(max_absolute_error=error,decoded_equal=same,samples=len(rows[:2]),prefix=prefix)


def prepare(config,out):
    model,decoder,compat=bootstrap_model(config)
    dataset=Path(config['dataset']);held=[];counts={}
    for split in ['train','validation','test']:
        good,bad=screen(readrows(dataset/f'recognition_candidates.{split}.jsonl'),decoder.character[1:],config['max_length'])
        save(out/f'{split}.json',good);held.extend(dict(r,split=split) for r in bad);counts[split]=len(good)
        (out/f'{split}.txt').write_text(''.join(str((dataset/r['image']).resolve())+'\t'+r['text']+'\n' for r in good))
    save(out/'held.json',held)
    save(out/'provenance.json',dict(config=config,compatibility=compat,counts=counts,held=len(held),
         upstream_commit=subprocess.check_output(['git','-C',config['upstream'],'rev-parse','HEAD'],text=True).strip(),
         hashes={k:sha256(config[k]) for k in ['official_config','dictionary','pretrained']},
         dataset_manifest_sha256=sha256(dataset/'manifest.json'),source_sha256=sha256(__file__)))
    print(json.dumps(dict(counts=counts,held=len(held),compatibility=compat)),flush=True)


def train(config,out,smoke=False):
    import numpy as np
    import paddle
    paddle.seed(config['seed']);np.random.seed(config['seed']);random.seed(config['seed'])
    model,decoder,compat=bootstrap_model(config)
    dataset=config['dataset'];root=Path(config['prepared'])
    trainrows=json.loads((root/'train.json').read_text());valrows=json.loads((root/'validation.json').read_text())
    if smoke:trainrows=trainrows[:32];valrows=valrows[:8]
    cmap={c:i for i,c in enumerate(decoder.character)}
    # CTC-only fine-tuning preserves the auxiliary head without using its loss.
    for name,p in model.named_parameters():
        if name.startswith(('head.gtc_head.','head.before_gtc.')):p.stop_gradient=True
    optimizer=paddle.optimizer.Adam(learning_rate=config['learning_rate'],parameters=[p for p in model.parameters() if not p.stop_gradient],grad_clip=paddle.nn.ClipGradByGlobalNorm(5.))
    before,runtime=infer(model,decoder,valrows,dataset,config)
    baseline=summarize([(r['text'],p) for r,p in zip(valrows,before)])
    save(out/'before.validation.json',dict(texts=before,runtime=runtime,metrics=baseline))
    best=float('inf');bad_epochs=0;history=[];step=0
    initial=model.head.ctc_head.fc.weight.numpy().copy()
    epochs=2 if smoke else config['epochs']
    cached={r['image']:preprocess(Path(dataset)/r['image'],config['width']) for r in trainrows}
    for epoch in range(1,epochs+1):
        model.train();order=list(range(len(trainrows)));random.shuffle(order);losses=[];start=time.perf_counter()
        for start_index in range(0,len(order),config['batch_size']):
            rows=[trainrows[i] for i in order[start_index:start_index+config['batch_size']]]
            x=np.stack([cached[r['image']].copy() for r in rows])
            # Train-only mild contrast augmentation, no text or geometry alteration.
            if config['contrast_augmentation']:x=np.clip(x*np.random.uniform(.9,1.1),-1,1)
            labels=np.zeros((len(rows),config['max_length']),dtype='int32');lengths=[]
            for i,r in enumerate(rows):
                ids=[cmap[c] for c in r['text']];labels[i,:len(ids)]=ids;lengths.append(len(ids))
            logits=ctc_forward_train(model,paddle.to_tensor(x))
            # Repeated adjacent CTC targets need an extra time step; fail rather than drop.
            required=max(ctc_timesteps(r['text']) for r in rows)
            if required>logits.shape[1]:raise ValueError('CTC alignment too short')
            loss=paddle.nn.functional.ctc_loss(logits.transpose([1,0,2]),paddle.to_tensor(labels),
                paddle.to_tensor([logits.shape[1]]*len(rows),dtype='int64'),paddle.to_tensor(lengths,dtype='int64'),blank=0,reduction='mean')
            value=float(loss.numpy())
            if not math.isfinite(value):raise ValueError('nonfinite loss')
            loss.backward();optimizer.step();optimizer.clear_grad();losses.append(value);step+=1
            if step%20==0:print(json.dumps(dict(epoch=epoch,step=step,loss=value)),flush=True)
        after,runtime=infer(model,decoder,valrows,dataset,config)
        metrics=summarize([(r['text'],p) for r,p in zip(valrows,after)])
        record=dict(epoch=epoch,step=step,loss=float(np.mean(losses)),validation=metrics,seconds=time.perf_counter()-start)
        history.append(record);save(out/'history.json',history);print(json.dumps(record),flush=True)
        state=dict(epoch=epoch,step=step,optimizer=optimizer.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),paddle_rng=paddle.get_rng_state(),model=model.state_dict())
        paddle.save(state,str(out/'latest.pdckpt'))
        if metrics['cer_micro']<best:
            best=metrics['cer_micro'];bad_epochs=0;paddle.save(model.state_dict(),str(out/'best.pdparams'))
            save(out/'selection.json',dict(criterion='minimum validation micro CER',epoch=epoch,step=step,cer=best,test_accessed=False,baseline_cer=baseline['cer_micro']))
            save(out/'after.validation.json',dict(texts=after,runtime=runtime,metrics=metrics))
        else:bad_epochs+=1
        if bad_epochs>=config['patience']:break
    delta=float(np.max(np.abs(initial-model.head.ctc_head.fc.weight.numpy())))
    if delta==0:raise ValueError('no actual weight update')
    # Resume one step on train-only data, verify optimizer state and finite loss;
    # restore selected weights before export. This step cannot affect selection.
    state=paddle.load(str(out/'latest.pdckpt'));optimizer.set_state_dict(state['optimizer']);model.set_state_dict(state['model'])
    model.train();optimizer.clear_grad()
    logits=ctc_forward_train(model,paddle.to_tensor(x))
    resumed=paddle.nn.functional.ctc_loss(logits.transpose([1,0,2]),paddle.to_tensor(labels),paddle.to_tensor([logits.shape[1]]*len(rows),dtype='int64'),paddle.to_tensor(lengths,dtype='int64'),blank=0,reduction='mean')
    resumed.backward();optimizer.step();optimizer.clear_grad()
    if not math.isfinite(float(resumed.numpy())):raise ValueError('resume loss nonfinite')
    model.set_state_dict(paddle.load(str(out/'best.pdparams')))
    export=export_and_check(model,decoder,valrows,dataset,config,out)
    save(out/'verification.json',dict(ctc_weight_max_update=delta,resume_step_finite_loss=float(resumed.numpy()),resumed_from_step=state['step'],export=export,compatibility=compat,full_train=not smoke,train_samples=len(trainrows)))


def final(config,out):
    import paddle
    root=Path(config['prepared']);rows=json.loads((root/'test.json').read_text())
    model,decoder,_=bootstrap_model(config)
    before,bt=infer(model,decoder,rows,config['dataset'],config)
    selected=Path(config['trained'])
    model.set_state_dict(paddle.load(str(selected/'best.pdparams')))
    after,at=infer(model,decoder,rows,config['dataset'],config)
    result=compare(rows,before,after);result.update(before_runtime=bt,after_runtime=at,selected_checkpoint_sha256=sha256(selected/'best.pdparams'),config=config)
    save(out/'comparison.test.json',result)
    for name,texts in [('before',before),('after',after)]:
        save(out/f'{name}.test.json',dict(mode='recognition_only',dataset_manifest_sha256=sha256(Path(config['dataset'])/'manifest.json'),crops=[dict(page_id=r['page_id'],region_id=r['region_id'],text=t) for r,t in zip(rows,texts)]))
    print(json.dumps({k:v for k,v in result.items() if k not in ['strata','cases','config']},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--stage',choices=['prepare','smoke','train','final'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    import paddle
    paddle.set_device(config['device'])
    if a.stage=='prepare':prepare(config,out)
    elif a.stage in ['smoke','train']:train(config,out,a.stage=='smoke')
    else:final(config,out)

if __name__=='__main__':main()
