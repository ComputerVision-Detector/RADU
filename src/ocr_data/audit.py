"""Conservative label audit. Types are suggestions, never corrected ground truth."""
from __future__ import annotations
import hashlib
import math
import re
import unicodedata
from pathlib import Path
from PIL import Image
from src.evaluation.evaluate import _iou

LATEX = re.compile(r'\\[A-Za-z]+|\\[\[\]() ]|\$')
ALNUM = re.compile(r'[가-힣ᄀ-ᇿㄱ-ㅣA-Za-z0-9]')
KNOWN_HOLDS = {
    'hf-0000': 'visible_unboxed_handwriting: header and left sticky notes outside GT regions; page coverage requires review',
    'hf-0104': 'no_text_conflicts_with_visible_handwriting: human reannotation required',
    'hf-0011': 'unresolved_merge_note: source boxes 5+6, 8+9, 10+11; do not merge automatically',
}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def verify_image(path, sample):
    if sha256(path) != sample['sha256']: raise ValueError(f"original_hash_mismatch: {path}")
    with Image.open(path) as image:
        if image.size != (sample['width'],sample['height']): raise ValueError(f"original_size_mismatch: {path}")
        image.verify()


def valid_bbox(box, width, height):
    return (isinstance(box,list) and len(box)==4 and
            all(type(x) in (int,float) and math.isfinite(x) for x in box) and
            0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def text_flags(text):
    flags=[]
    if '\n' in text or '\r' in text: flags.append('multiline')
    if '\t' in text: flags.append('tab')
    if LATEX.search(text): flags.append('latex_notation')
    if text.strip() and not ALNUM.search(text): flags.append('symbol_only')
    if len(text)>256: flags.append('long_string')
    if any(unicodedata.category(c).startswith('C') and c not in '\n\r\t' for c in text): flags.append('control_or_unassigned')
    return flags


def audit_page(item, sample, asset=None):
    id_=item['id']; image=item['image']; ann=item['bbox_annotation']
    holds=[]
    if id_ in KNOWN_HOLDS: holds.append(KNOWN_HOLDS[id_])
    if ann.get('status')!='verified': holds.append('page_not_verified')
    if ann.get('notes','').strip(): holds.append('annotation_note_requires_review')
    if (image.get('source_width'),image.get('source_height'),image.get('source_sha256')) != (sample['width'],sample['height'],sample['sha256']): holds.append('label_image_provenance_mismatch')
    if asset and (asset['source_sha256']!=sample['sha256'] or image.get('asset_sha256')!=asset['sha256']): holds.append('asset_provenance_mismatch')
    boxes=ann.get('boxes',[])
    if bool(ann.get('no_text')) == bool(boxes): holds.append('no_text_box_contradiction')
    regions=[]; ids=set(); orders=set()
    for b in boxes:
        reasons=[]; text=b.get('text')
        if not isinstance(text,str): reasons.append('missing_text'); text=''
        if not text.strip(): reasons.append('empty_transcription')
        if b.get('reviewed') is not True: reasons.append('not_reviewed')
        if b.get('uncertain') is not False: reasons.append('uncertain_or_missing')
        if not valid_bbox(b.get('bbox'),sample['width'],sample['height']): reasons.append('invalid_bbox')
        bid=b.get('id'); order=b.get('order')
        if not isinstance(bid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',bid): reasons.append('invalid_id')
        if bid in ids: holds.append('duplicate_region_id')
        if type(order) is not int or order<1 or order in orders: holds.append('invalid_or_duplicate_reading_order')
        ids.add(bid);orders.add(order)
        flags=text_flags(text)
        if 'control_or_unassigned' in flags: reasons.append('unsupported_control_requires_review')
        kind='formula' if 'latex_notation' in flags else 'symbol' if 'symbol_only' in flags else 'text' if ALNUM.search(text) else 'unclassified'
        if 'multiline' in flags and kind!='formula': reasons.append('multiline_text_requires_region_review')
        regions.append(dict(id=bid,bbox=b.get('bbox'),text=text,order=order,flags=flags,
                            type_suggestion=kind,type_confirmed=None,classification='rule_suggestion_only',hold_reasons=reasons))
    overlap=[]
    for i,a in enumerate(regions):
        for b in regions[i+1:]:
            if valid_bbox(a['bbox'],sample['width'],sample['height']) and valid_bbox(b['bbox'],sample['width'],sample['height']):
                iou=_iou(a['bbox'],b['bbox'])
                if iou>=.8:
                    overlap.append(dict(ids=[a['id'],b['id']],iou=iou))
                    if iou>=.95:
                        a['hold_reasons'].append('duplicate_region');b['hold_reasons'].append('duplicate_region')
    return dict(id=id_,width=sample['width'],height=sample['height'],source_sha256=sample['sha256'],
                no_text=bool(ann.get('no_text')),status=ann.get('status'),annotator=ann.get('annotator'),notes=ann.get('notes'),
                hold_reasons=holds,regions=regions,overlap_candidates=overlap,
                eligible=not holds and not any(r['hold_reasons'] for r in regions))
