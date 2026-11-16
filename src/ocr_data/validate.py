"""Verify dataset bytes, frozen GT, exact crop pixels and split isolation."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from PIL import Image
from src.ocr_data.audit import sha256,verify_image
from src.ocr_data.build import load,verify_snapshot
from src.ocr_data.split import validate_splits


def validate(dataset):
    root=Path(dataset);manifest=load(root/'manifest.json')
    for name,expected in manifest['artifacts_sha256'].items():
        if Path(name).name!=name or sha256(root/name)!=expected:raise ValueError(f'artifact hash mismatch: {name}')
    source=Path(manifest['snapshot_path']);current=verify_snapshot(source)
    if current!=manifest['snapshot_manifest']:raise ValueError('snapshot binding mismatch')
    labels=load(source/'live-v2.json')['items']
    pages=[json.loads(x) for x in (root/'pages.jsonl').read_text().splitlines()]
    crops=[json.loads(x) for x in (root/'crops.jsonl').read_text().splitlines()]
    splits=load(root/'splits.json');validate_splits(pages,splits['assignments'])
    if set(labels)!={p['id'] for p in pages}:raise ValueError('page ground truth coverage drift')
    grouped=defaultdict(list);seen=set()
    for row in crops:
        key=row['page_id'],row['region_id']
        if key in seen:raise ValueError('duplicate crop ID')
        seen.add(key);grouped[row['page_id']].append(row)
    if set(grouped)-{p['id'] for p in pages}:raise ValueError('unknown crop page')
    for p in pages:
        id_=p['id'];ann=labels[id_]['bbox_annotation']
        if p['split']!=splits['assignments'][id_]:raise ValueError('page split mismatch')
        expected=[(r['id'],r['bbox'],r['text'],r['order']) for r in ann['boxes']]
        actual=[(r['id'],r['bbox'],r['text'],r['order']) for r in p['regions']]
        if expected!=actual or p['no_text']!=ann['no_text']:raise ValueError('page ground truth drift')
        original=labels[id_]['image']
        if (p['width'],p['height'],p['source_sha256'])!=(original['source_width'],original['source_height'],original['source_sha256']):raise ValueError('page original provenance drift')
        verify_image(p['image'],dict(width=p['width'],height=p['height'],sha256=p['source_sha256']))
        rows=grouped[id_]
        wanted={r['id'] for r in p['regions']} if p['eligible'] else set()
        if {r['region_id'] for r in rows}!=wanted:raise ValueError('crop coverage/eligibility mismatch')
        with Image.open(p['image']) as original_image:
            rgb=original_image.convert('RGB')
            for row in rows:
                if row['split']!=p['split']:raise ValueError('crop split mismatch')
                b=next(x for x in ann['boxes'] if x['id']==row['region_id'])
                if row['text']!=b['text'] or row['bbox']!=b['bbox'] or row['source_sha256']!=p['source_sha256']:raise ValueError('ground truth drift')
                xy=[math.floor(b['bbox'][0]),math.floor(b['bbox'][1]),math.ceil(b['bbox'][2]),math.ceil(b['bbox'][3])]
                if row['crop_bbox']!=xy:raise ValueError('crop coordinates mismatch')
                path=Path(row['image'])
                if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe crop path')
                if sha256(root/path)!=row['crop_sha256']:raise ValueError('crop hash mismatch')
                with Image.open(root/path) as im:
                    expected_image=rgb.crop(xy)
                    if im.size!=expected_image.size:raise ValueError('crop dimensions mismatch')
                    if im.convert('RGB').tobytes()!=expected_image.tobytes():raise ValueError('crop pixels mismatch')
    return dict(pages_verified=len(pages),crops_verified=len(crops),split_leakage=False,prediction_as_ground_truth=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True)
    print(json.dumps(validate(p.parse_args().dataset)))
if __name__=='__main__':main()
