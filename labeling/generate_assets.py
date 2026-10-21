"""Generate web reading images and an inventory; never modify frozen sources."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/benchmark/qwen_ablation_pilot'
OUT=ROOT/'labeling/assets'

def make(sample):
    source=BASE/sample['image'];target=OUT/(sample['id']+'.webp')
    with Image.open(source) as original:
        image=ImageOps.exif_transpose(original).convert('RGB');image.thumbnail((3200,3200),Image.Resampling.LANCZOS)
        image.save(target,quality=85,method=3)
        return {'id':sample['id'],'file':target.name,'width':image.width,'height':image.height,'source_sha256':sample['sha256'],
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size,'encoding':'WebP quality 85, max side 3200; web reading copy, not benchmark input'}

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    samples=json.loads((BASE/'additional_images.manifest.json').read_text())['samples']
    with ThreadPoolExecutor(max_workers=4) as pool:
        result=[]
        for item in pool.map(make,samples):
            result.append(item)
            if len(result)%20==0:print(f'{len(result)}/{len(samples)} web images generated',flush=True)
    (ROOT/'labeling/data/assets.manifest.json').write_text(json.dumps({'images':result},ensure_ascii=False,indent=2)+'\n')
    print(f'{len(result)} web images, {sum(i["bytes"] for i in result)/1024**2:.1f} MiB',flush=True)
