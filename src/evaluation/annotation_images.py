"""Restore annotation PNGs from the frozen HF revision, without OCR/GPU imports."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore(manifest, directory=None, identities=None, cache_only=False):
    manifest=Path(manifest)
    data=json.loads(manifest.read_text(encoding='utf-8'))
    base=Path(directory) if directory else manifest.parent
    samples=data['samples']
    if identities:
        requested=set(identities)
        missing=requested-{s['id'] for s in samples}
        if missing: raise ValueError(f'Unknown image IDs: {sorted(missing)}')
        samples=[s for s in samples if s['id'] in requested]
    pending=[]
    for sample in samples:
        # Inventory paths are deliberately restricted to the numbered image directory.
        if sample['image'] != f"images/{sample['id']}.png" or '/' in sample['id'] or '\\' in sample['id']:
            raise ValueError(f"Invalid inventory path: {sample['image']}")
        target=base/sample['image']
        if target.exists():
            if digest(target)!=sample['sha256']:
                raise ValueError(f'Existing image differs from frozen manifest: {target}; not overwritten')
        else: pending.append(sample)
    if not pending: return {'restored':0,'verified':len(samples)}
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as parquet
    from PIL import Image, ImageOps
    for shard in sorted({s['shard'] for s in pending}):
        path=hf_hub_download(data['dataset'],shard,repo_type='dataset',revision=data['revision'],local_files_only=cache_only)
        reader=parquet.ParquetFile(path)
        group_starts=[]; offset=0
        for group in range(reader.num_row_groups):
            count=reader.metadata.row_group(group).num_rows
            group_starts.append((group,offset,count));offset+=count
        cached_group=None; table=None
        for sample in sorted((s for s in pending if s['shard']==shard),key=lambda s:s['row']):
            row=sample['row']
            group,start,_=next((g,st,n) for g,st,n in group_starts if st<=row<st+n)
            if group!=cached_group:
                table=reader.read_row_group(group,columns=['image']);cached_group=group
            feature=table.slice(row-start,1).to_pylist()[0]['image']
            if not isinstance(feature,dict) or not feature.get('bytes'):
                raise ValueError('Expected embedded HF image bytes in frozen parquet')
            with Image.open(io.BytesIO(feature['bytes'])) as source:
                image=ImageOps.exif_transpose(source).convert('RGB')
            if image.size != (sample['width'],sample['height']):
                raise ValueError(f"Image dimensions differ: {sample['id']}")
            encoded=io.BytesIO();image.save(encoded,format='PNG');raw=encoded.getvalue()
            if hashlib.sha256(raw).hexdigest()!=sample['sha256']:
                raise ValueError(f"PNG hash differs: {sample['id']}; install requirements-annotation.txt to match the frozen encoder. Nothing overwritten.")
            target=base/sample['image'];target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as out: out.write(raw)
            print(f"Restored {sample['id']}",flush=True)
    return {'restored':len(pending),'verified':len(samples)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('data/benchmark/qwen_ablation_pilot/additional_images.manifest.json'))
    parser.add_argument('--directory',type=Path,help='Dataset root; images are placed in its images/ subfolder')
    parser.add_argument('--ids',nargs='+',help='Optional subset of image IDs')
    parser.add_argument('--cache-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(restore(args.manifest,args.directory,args.ids,args.cache_only),ensure_ascii=False))

if __name__=='__main__': main()
