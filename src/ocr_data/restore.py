"""Restore hash-identical original PNGs using existing parquet restoration first."""
import argparse
import io
import json
from pathlib import Path
from PIL import Image,ImageOps
from src.evaluation.annotation_images import restore as restore_parquet
from src.ocr_data.audit import sha256,verify_image


def restore(snapshot, output, cache_only=False):
    from huggingface_hub import hf_hub_download
    snapshot=Path(snapshot);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    from src.ocr_data.build import verify_snapshot
    verify_snapshot(snapshot)
    catalog=json.loads((snapshot/'dataset.catalog.json').read_text())
    # The existing frozen parquet inventory covers the original 122 IDs. New
    # catalog entries have loose-file provenance with their own frozen revision.
    old=snapshot/'legacy_images.manifest.json'
    if not old.exists():
        # Backward-compatible snapshots bind the legacy inventory through code_commit.
        import subprocess
        code_commit=json.loads((snapshot/'manifest.json').read_text())['code_commit']
        import re
        if not re.fullmatch('[0-9a-f]{40}',code_commit):raise ValueError('invalid frozen code commit')
        raw=subprocess.check_output(['git','show',code_commit+':data/benchmark/qwen_ablation_pilot/additional_images.manifest.json'])
        old=output/'legacy_images.manifest.json'
        if old.exists():
            if old.read_bytes()!=raw:raise ValueError('legacy inventory changed')
        else:
            with old.open('xb') as stream:stream.write(raw)
    legacy=json.loads(old.read_text())
    expected={s['id']:s for s in catalog['samples']}
    compatible=[s['id'] for s in legacy['samples'] if s['id'] in expected and s['sha256']==expected[s['id']]['sha256']]
    restore_parquet(old,output,compatible,cache_only)
    receipts=[]
    for sample in catalog['samples']:
        id_=sample['id'];target=output/f'images/{id_}.png'
        if not target.exists():
            if not sample.get('hf_file'):raise ValueError(f'missing loose-image provenance: {id_}')
            revision=sample.get('hf_revision',catalog['revision'])
            source=Path(hf_hub_download(catalog['dataset'],sample['hf_file'],repo_type='dataset',revision=revision,local_files_only=cache_only))
            if sample.get('hf_source_sha256') and sha256(source)!=sample['hf_source_sha256']:raise ValueError(f'HF source hash mismatch: {id_}')
            with Image.open(source) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
            encoded=io.BytesIO();image.save(encoded,format='PNG');raw=encoded.getvalue()
            import hashlib
            if image.size!=(sample['width'],sample['height']) or hashlib.sha256(raw).hexdigest()!=sample['sha256']:
                raise ValueError(f'restored PNG hash/size mismatch: {id_}; match requirements-annotation.txt; no crop created')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as out:out.write(raw)
        verify_image(target,sample)
        receipts.append(dict(id=id_,sha256=sha256(target),width=sample['width'],height=sample['height'],
                             revision=legacy['revision'] if id_ in compatible else sample.get('hf_revision',catalog['revision']),
                             origin=dict(shard=sample.get('shard'),row=sample.get('row')) if id_ in compatible else dict(hf_file=sample['hf_file'])))
    receipt=output/'restore_receipts.json'
    if receipt.exists():
        if json.loads(receipt.read_text())!=receipts: raise ValueError('restore receipts changed')
    else:receipt.write_text(json.dumps(receipts,indent=2)+'\n')
    return dict(verified=len(receipts),image_root=str(output))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True);p.add_argument('--cache-only',action='store_true')
    a=p.parse_args();print(json.dumps(restore(a.snapshot,a.output,a.cache_only)))
if __name__=='__main__':main()
