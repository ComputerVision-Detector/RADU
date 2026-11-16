"""Train-only risk audit and visual contact sheets; no OCR-derived GT edits."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import random

from PIL import Image,ImageDraw,ImageFont
from src.ocr_data.audit import sha256
from src.training.ocr_finetune import readrows,save,verify_crops


def clean_training(rows,held_pages,held_regions=None):
    kept,held=[],[];held_regions=held_regions or {}
    for original in rows:
        if original.get('split')!='train':raise ValueError('cleaning consumes train only')
        row={k:v for k,v in original.items() if k!='prediction'}
        reason=held_pages.get(row['page_id']) or held_regions.get(row['region_id'])
        if reason:held.append(dict(row,round3_hold_reason=reason))
        else:kept.append(row)
    return kept,held


def inspect_crop(path,row):
    import cv2
    import numpy as np
    im=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if im is None:raise ValueError('unreadable crop')
    h,w=im.shape;small=cv2.resize(im,(min(w,1000),max(12,round(h*min(w,1000)/w))))
    _,ink=cv2.threshold(small,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    rule=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(20,int(ink.shape[1]*.65))),np.uint8))
    ink=cv2.subtract(ink,rule);projection=(ink>0).mean(axis=1)
    smoothed=np.convolve(projection,np.ones(3)/3,mode='same');active=(smoothed>max(.025,float(smoothed.max())*.22)).astype('uint8')
    active=cv2.morphologyEx(active[:,None],cv2.MORPH_CLOSE,np.ones((max(2,round(len(active)*.08)),1),np.uint8))[:,0]
    bands=[];start=None
    for i,a in enumerate([*active,0]):
        if a and start is None:start=i
        if not a and start is not None:
            if i-start>=len(active)*.1:bands.append([int(start),i])
            start=None
    flags=[]
    if len(bands)>=2:flags.append('multiple_ink_bands_candidate')
    if w/h<5 and len(row['text'])>12:flags.append('tall_for_long_transcription')
    border=float((ink[:max(1,len(ink)//12)]>0).mean()+(ink[-max(1,len(ink)//12):]>0).mean())/2
    if border>.12:flags.append('border_ink_candidate')
    if len(row['text'])>60:flags.append('long_transcription')
    if any(0x3131<=ord(c)<=0x318e for c in row['text']):flags.append('isolated_hangul_jamo_candidate')
    score=3*('multiple_ink_bands_candidate' in flags)+2*('tall_for_long_transcription' in flags)+2*('border_ink_candidate' in flags)+('long_transcription' in flags)+('isolated_hangul_jamo_candidate' in flags)
    return dict(width=w,height=h,ink_bands=bands,border_ink_fraction=border,flags=flags,risk_score=score)


def panels(dataset,rows,out):
    bypage=defaultdict(list)
    for r in rows:bypage[r['page_id']].append(r)
    # Two highest-risk representatives from every train page, plus global risk
    # rows until 260. This is coverage sampling, not a claim of full GT review.
    selected={r['region_id']:r for page in sorted(bypage) for r in sorted(bypage[page],key=lambda x:(-x['risk_score'],x['region_id']))[:2]}
    for r in sorted(rows,key=lambda x:(-x['risk_score'],x['region_id'])):
        if len(selected)>=260:break
        selected[r['region_id']]=r
    selected=sorted(selected.values(),key=lambda x:(x['page_id'],-x['risk_score'],x['region_id']))
    font=ImageFont.truetype('/usr/share/fonts/truetype/nanum/NanumGothic.ttf',19)
    index=[]
    for start in range(0,len(selected),16):
        part=selected[start:start+16];canvas=Image.new('RGB',(1800,160*len(part)),'white');d=ImageDraw.Draw(canvas)
        for i,r in enumerate(part):
            with Image.open(Path(dataset)/r['image']) as original:
                im=original.convert('RGB');im.thumbnail((1750,94));canvas.paste(im,(20,160*i+28))
            d.text((20,160*i),f"{start+i:03d} {r['page_id']} {r['region_id']} risk={r['risk_score']} {','.join(r['flags'])}",font=font,fill='blue')
            label=r['text'];d.text((20,160*i+126),'GT: '+label[:130],font=font,fill='black')
            index.append(dict(index=start+i,panel=f'panel-{start//16:02d}.jpg',**r))
        canvas.save(out/f'panel-{start//16:02d}.jpg',quality=94)
    save(out/'review-index.json',index)
    return dict(selected_crops=len(selected),represented_pages=len(bypage),panels=(len(selected)+15)//16,policy='top2 risk per page + highest global risk up to260; model predictions not used')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',default='data/processed/ocr_v2/pilot-v2');p.add_argument('--output',default='data/processed/ocr_v2/train-audit-round3');p.add_argument('--decisions');a=p.parse_args()
    root=Path(a.output)
    if a.decisions:
        decisions=json.loads(Path(a.decisions).read_text());rows=readrows(Path(a.dataset)/'recognition_candidates.train.jsonl');kept,held=clean_training(rows,decisions['held_pages'],decisions.get('held_regions'))
        save(root/'clean-train.json',kept);save(root/'held-train.json',held)
        save(root/'clean-manifest.json',dict(source_dataset_manifest_sha256=sha256(Path(a.dataset)/'manifest.json'),source_train_sha256=sha256(Path(a.dataset)/'recognition_candidates.train.jsonl'),decisions_sha256=sha256(a.decisions),kept=len(kept),held=len(held),kept_pages=len({r['page_id'] for r in kept}),held_pages=len({r['page_id'] for r in held}),labels_changed=False,selection_uses_model_predictions=False,clean_train_sha256=sha256(root/'clean-train.json'),status='conservatively screened training candidates; not fully human-verified'))
        print(json.dumps(dict(kept=len(kept),held=len(held))));return
    root.mkdir(parents=True,exist_ok=False);rows=readrows(Path(a.dataset)/'recognition_candidates.train.jsonl');clean_training(rows,{})
    verify_crops(rows,a.dataset);audited=[]
    for r in rows:audited.append(dict(r,**inspect_crop(Path(a.dataset)/r['image'],r)))
    save(root/'audit.json',audited);coverage=panels(a.dataset,audited,root)
    save(root/'audit-summary.json',dict(train_crops=len(rows),pages=len({r['page_id'] for r in rows}),flags=dict(Counter(f for r in audited for f in r['flags'])),coverage=coverage,source_train_sha256=sha256(Path(a.dataset)/'recognition_candidates.train.jsonl'),note='heuristic flags are review suggestions only; no automatic semantic label correction'))
    print(json.dumps(coverage))

if __name__=='__main__':main()
