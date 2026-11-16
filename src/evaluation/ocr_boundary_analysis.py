"""Reproduce saved crop error tables and image failure evidence; no inference."""
from PIL import Image,ImageDraw
import numpy as np
from src.evaluation.ocr_boundary import ROOT,read,write_new,summarize


def main():
    cases={};meta=read(ROOT/'crop-review-index.json')
    for split in ['validation','diagnostic']:
        rows=read(ROOT/f'{split}.before.rows.json');before=read(ROOT/split/'before.json')['predictions'];after=read(ROOT/split/'after.json')['predictions'];lookup={r['index']:r for r in meta if r['split']==split};result=[]
        for i,(r,a,b) in enumerate(zip(rows,before,after)):
            x=summarize([(r['text'],a['text'])])['errors'];y=summarize([(r['text'],b['text'])])['errors'];m=lookup[i]
            result.append(dict(index=i,page_id=r['page_id'],region_id=r['region_id'],reference=r['text'],before=a['text'],after=b['text'],before_errors=x,after_errors=y,delta=y-x,**{k:m[k] for k in ['original','trimmed','changed','local_trim_bbox','original_size']}))
        cases[split]=result
    write_new(ROOT/'crop-error-cases.json',cases)
    chosen=sorted(cases['validation'],key=lambda x:x['delta'],reverse=True)[:4]+sorted(cases['validation'],key=lambda x:x['delta'])[:4];write_new(ROOT/'visual-error-examples.json',chosen)
    for start in [0,4]:
        im=Image.new('RGB',(2200,4*240),'#eee');d=ImageDraw.Draw(im)
        for j,r in enumerate(chosen[start:start+4]):
            d.text((4,j*240+3),f"{start+j} {r['page_id']} {r['region_id']} errors {r['before_errors']} -> {r['after_errors']}",fill='black')
            for x,key in [(0,'original'),(1100,'trimmed')]:
                crop=Image.open(r[key]);crop.thumbnail((1090,200));im.paste(crop,(x,j*240+30))
        path=ROOT/'review'/f'errors-{start//4}.jpg'
        if path.exists():raise FileExistsError(path)
        im.save(path)
    proof=[]
    for r in cases['validation']:
        if r['region_id'] not in ['hf-0157-line-0026','hf-0157-line-0027']:continue
        im=Image.open(r['original']);a=np.asarray(im.convert('L'));x1,y1,x2,y2=r['local_trim_bbox'];mask=a<160
        proof.append(dict(page_id=r['page_id'],region_id=r['region_id'],original_size=list(im.size),trim_bbox=r['local_trim_bbox'],retained_width_fraction=(x2-x1)/im.width,gray_median_original=float(np.median(a)),gray_median_removed_prefix=float(np.median(a[:,:x1])) if x1 else None,removed_prefix_dark_fraction=float(mask[:,:x1].mean()) if x1 else None,before_errors=r['before_errors'],after_errors=r['after_errors'],conclusion='fixed threshold/long-run removal fails on dark background and cuts visible sentence; unselected method'))
    write_new(ROOT/'dark-background-failure.json',proof)

if __name__=='__main__':main()
