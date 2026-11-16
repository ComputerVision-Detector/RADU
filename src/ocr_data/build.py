"""Audit frozen V2 labels, group pages, then allocate hash-verified crops."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps
from src.ocr_data.audit import audit_page,sha256,verify_image
from src.ocr_data.split import assign_splits


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,data):
    with Path(path).open('x',encoding='utf-8') as out:json.dump(data,out,ensure_ascii=False,indent=2);out.write('\n')
def jsonl(path,rows):
    with Path(path).open('x',encoding='utf-8') as out:
        for row in rows:out.write(json.dumps(row,ensure_ascii=False)+'\n')


def verify_snapshot(snapshot):
    manifest=load(snapshot/'manifest.json')
    for name,expected in manifest['files_sha256'].items():
        if Path(name).name!=name or sha256(snapshot/name)!=expected:raise ValueError(f'snapshot hash mismatch: {name}')
    return manifest


def fingerprint(image):
    small=ImageOps.grayscale(image).resize((9,8),Image.Resampling.LANCZOS)
    pixels=list(small.get_flattened_data())
    return sum((pixels[y*9+x]>pixels[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))


def build(snapshot, originals, output, metadata=None, seed=42):
    snapshot=Path(snapshot);originals=Path(originals).resolve();output=Path(output)
    manifest=verify_snapshot(snapshot)
    data=load(snapshot/'live-v2.json');catalog=load(snapshot/'dataset.catalog.json');assets=load(snapshot/'assets.manifest.json')
    if data.get('dataset_version')!='v2' or not isinstance(data.get('items'),dict):raise ValueError('expected internal V2 items/bbox_annotation, not samples export')
    samples={s['id']:s for s in catalog['samples']};assetmap={a['id']:a for a in assets['images']}
    if len(samples)!=len(catalog['samples']) or len(assetmap)!=len(assets['images']):raise ValueError('duplicate catalog/asset ID')
    if set(samples)!=set(data['items']) or set(assetmap)!=set(samples):raise ValueError('catalog/labels/assets coverage mismatch')
    meta={}
    if metadata:
        meta=load(metadata)
        if set(meta)-set(samples):raise ValueError('unknown metadata page IDs')
    output.mkdir(parents=True,exist_ok=False);(output/'crops').mkdir();(output/'overlays').mkdir();(output/'review_images').mkdir()
    pages=[];integrity=[];fingerprints={};pixelhashes={};review=[]
    for id_,item in sorted(data['items'].items()):
        if item.get('id')!=id_ or not __import__('re').fullmatch('[A-Za-z0-9_-]+',id_):raise ValueError('unsafe/mismatched page ID')
        sample=samples[id_];p=audit_page(item,sample,assetmap[id_]);path=originals/f'images/{id_}.png'
        try:
            verify_image(path,sample)
            with Image.open(path) as im:
                rgb=im.convert('RGB');fingerprints[id_]=fingerprint(rgb);pixelhashes[id_]=hashlib.sha256(rgb.tobytes()).hexdigest()
        except (OSError,ValueError) as e:
            p['hold_reasons'].append('original_integrity_failure');p['eligible']=False;integrity.append(dict(id=id_,error=str(e)))
        p.update(image=str(path),groups=[f"source_sha256:{sample['sha256']}"])
        for field in ['writer_id','note_id','design_id','source_family_id']:
            value=meta.get(id_,{}).get(field)
            if value is not None and (not isinstance(value,str) or not value.strip()):raise ValueError(f'invalid metadata: {id_}/{field}')
            p[field]=value
            if value:p['groups'].append(f'{field}:{value}')
        pages.append(p)
    similar=[];exact=[]
    for i,p in enumerate(pages):
        for q in pages[i+1:]:
            a,b=p['id'],q['id']
            if a not in fingerprints or b not in fingerprints:continue
            distance=(fingerprints[a]^fingerprints[b]).bit_count()
            if pixelhashes[a]==pixelhashes[b]:
                exact.append([a,b]);group='decoded_pixels:'+pixelhashes[a];p['groups'].append(group);q['groups'].append(group)
            elif distance<=4:
                similar.append(dict(pages=[a,b],dhash_distance=distance,status='candidate_not_confirmed'))
                # Conservative co-location, even though similarity alone cannot establish identity.
                group=f'similarity_candidate:{a}:{b}';p['groups'].append(group);q['groups'].append(group)
    split=assign_splits([dict(id=p['id'],groups=sorted(set(p['groups']))) for p in pages],seed)
    split['labels_sha256']=manifest['files_sha256']['live-v2.json']
    split['metadata_sha256']=sha256(metadata) if metadata else None
    split['similarity_policy']='64-bit horizontal dHash distance <=4; co-locate all candidate pairs transitively; requires human confirmation'
    crops=[];lists={s:[] for s in ['train','validation','test']};held=Counter();flags=Counter();types=Counter()
    for p in pages:
        id_=p['id'];p['split']=split['assignments'][id_];p['writer_independence_verified']=False
        p['groups']=sorted(set(p['groups']))
        for r in p['regions']:flags.update(r['flags']);types.update([r['type_suggestion']])
        if p['hold_reasons'] or any(r['hold_reasons'] for r in p['regions']):
            review.append(dict(page_id=id_,reasons=p['hold_reasons'],regions=[dict(id=r['id'],reasons=r['hold_reasons']) for r in p['regions'] if r['hold_reasons']],notes=p['notes']))
        for r in p['regions']:
            if r['type_suggestion']!='text' or r['flags']:
                review.append(dict(page_id=id_,region_id=r['id'],type_suggestion=r['type_suggestion'],type_confirmed=None,flags=r['flags'],action='confirm semantic type and transcription conventions; do not auto-split fractions/matrices'))
        if not p['eligible']:
            held.update(p['hold_reasons']);held.update(reason for r in p['regions'] for reason in r['hold_reasons'])
        if any(x['id']==id_ for x in integrity):continue
        with Image.open(p['image']) as raw:
            image=raw.convert('RGB')
            # Overlays include held pages so humans can examine defects.
            if id_ in ('hf-0011','hf-0104') or (p['eligible'] and len(list((output/'overlays').glob('*.jpg')))<10):
                overlay=image.copy();draw=ImageDraw.Draw(overlay)
                for r in p['regions']:
                    from src.ocr_data.audit import valid_bbox
                    if not valid_bbox(r['bbox'],p['width'],p['height']):continue
                    draw.rectangle(r['bbox'],outline='red' if r['hold_reasons'] or p['hold_reasons'] else 'blue',width=5)
                    draw.text((r['bbox'][0],r['bbox'][1]),str(r['order']),fill='red',stroke_width=1)
                overlay.thumbnail((1600,1600));overlay.save(output/'overlays'/f'{id_}.jpg')
                if id_ in ('hf-0011','hf-0104'):
                    preview=image.copy();preview.thumbnail((1600,1600));preview.save(output/'review_images'/f'{id_}.jpg')
            if not p['eligible']:continue
            for index,r in enumerate(p['regions']):
                # Keep text unchanged; all four original-pixel coordinates remain in metadata.
                bounds=[math.floor(r['bbox'][0]),math.floor(r['bbox'][1]),math.ceil(r['bbox'][2]),math.ceil(r['bbox'][3])]
                name=f'{id_}-{index+1:04d}.png';target=output/'crops'/name
                image.crop(bounds).save(target)
                row=dict(page_id=id_,region_id=r['id'],image=f'crops/{name}',crop_sha256=sha256(target),
                         source_sha256=p['source_sha256'],bbox=r['bbox'],crop_bbox=bounds,text=r['text'],order=r['order'],
                         type_suggestion=r['type_suggestion'],type_confirmed=None,split=p['split'],flags=r['flags'],
                         hold_reasons=[],writer_independence_verified=False)
                crops.append(row)
                if r['type_suggestion']=='text' and not set(r['flags']) & {'multiline','tab','long_string','control_or_unassigned'}:
                    lists[p['split']].append(row)
    jsonl(output/'pages.jsonl',pages);jsonl(output/'crops.jsonl',crops);jsonl(output/'review_queue.jsonl',review)
    for s,rows in lists.items():jsonl(output/f'recognition_candidates.{s}.jsonl',rows)
    # No invented author IDs: annotator is a distinct field, never a group.
    template={p['id']:dict(writer_id=None,note_id=None,design_id=None,source_family_id=None,
                          evidence=None,type_overrides={},review_resolution=None) for p in pages}
    write(output/'group_metadata.template.json',template);write(output/'splits.json',split)
    write(output/'integrity_failures.json',integrity);write(output/'similarity_candidates.json',dict(exact_decoded_images=exact,similar=similar))
    stats=dict(pages=len(pages),statuses=dict(Counter(p['status'] for p in pages)),boxes=sum(len(p['regions']) for p in pages),
               reviewed_true=sum(b.get('reviewed') is True for p in data['items'].values() for b in p['bbox_annotation']['boxes']),
               no_text_pages=sum(p['no_text'] for p in pages),flags=dict(flags),type_suggestions=dict(types),
               eligible_pages=sum(p['eligible'] for p in pages),held_pages=sum(not p['eligible'] for p in pages),
               held_boxes=sum(len(p['regions']) for p in pages if not p['eligible']),generated_crops=len(crops),
               natural_text_recognition_candidates={k:len(v) for k,v in lists.items()},hold_reasons=dict(held),
               original_integrity_failures=len(integrity),exact_duplicate_pairs=len(exact),similar_candidate_pairs=len(similar),
               metadata_coverage={k:sum(bool(p[k]) for p in pages) for k in ['writer_id','note_id','design_id','source_family_id']},
               split_counts=split['counts'],writer_independence_verified=False,
               status='INCOMPLETE_ORIGINALS' if integrity else 'provisional_pilot_ready_requires_semantic_and_group_review')
    write(output/'quality_report.json',stats)
    write(output/'manifest.json',dict(schema=1,snapshot_manifest=manifest,snapshot_path=str(snapshot.resolve()),seed=seed,
                                    augmentation='none; future augmentations train only',model_paths_unchanged=True,
                                    artifacts_sha256={f.name:sha256(f) for f in sorted(output.iterdir()) if f.is_file()},
                                    original_receipts_sha256=sha256(originals/'restore_receipts.json') if (originals/'restore_receipts.json').exists() else None))
    return stats


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',required=True);p.add_argument('--originals',required=True);p.add_argument('--output',required=True);p.add_argument('--metadata');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();stats=build(a.snapshot,a.originals,a.output,a.metadata,a.seed);print(json.dumps(stats,ensure_ascii=False,indent=2))
    if stats['original_integrity_failures']:raise SystemExit(2)
if __name__=='__main__':main()
