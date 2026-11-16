"""Independent distance/assembly and gallery checks for a completed cached replay."""
import argparse
import re
import unicodedata
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote
from src.evaluation.ocr_page_audit import read, write_new, PRED


def canonical(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def distance(reference, hypothesis):
    previous = list(range(len(hypothesis)+1))
    for i, x in enumerate(reference, 1):
        current = [i]
        for j, y in enumerate(hypothesis, 1):
            current.append(min(previous[j]+1, current[-1]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def verify(root):
    root = Path(root)
    result = read(root/'summary/results.json')
    prediction = {p['id']:p for p in read(PRED)['pages']}
    errors = units = ignored = mixed = 0
    independent = []
    for page in read(root/'audited-gt-v2.json')['pages']:
        valid = sorted([g for g in page['regions'] if g['include']], key=lambda g:g['order'])
        held = [g for g in page['regions'] if not g['include']]
        texts = []
        for p in sorted(prediction[page['id']]['regions'], key=lambda p:p['order']):
            box = p['bbox']; area = (box[2]-box[0])*(box[3]-box[1])
            ignore = any(overlap(box, g['bbox']) >= area*.5 for g in held)
            hit = any(overlap(box, g['bbox']) > 0 for g in valid)
            if ignore and not hit:
                ignored += 1
                continue
            mixed += int(ignore and hit)
            texts.append(p['text'])
        reference = canonical('\n'.join(g['text'] for g in valid))
        hypothesis = canonical('\n'.join(texts))
        e = distance(reference, hypothesis)
        errors += e; units += len(reference)
        independent.append(dict(page_id=page['id'], errors=e, reference_units=len(reference)))
    assert errors == result['P0']['errors'] and units == result['P0']['reference_units']
    assert ignored == 93 and mixed == 26
    class Links(HTMLParser):
        def __init__(self):
            super().__init__(); self.links=[]; self.cards=0
        def handle_starttag(self, tag, attrs):
            attrs=dict(attrs)
            if tag == 'article': self.cards += 1
            self.links.extend(attrs[k] for k in ('href','src') if k in attrs)
    parser=Links();parser.feed((root/'review.html').read_text())
    assert parser.cards == 201
    for url in parser.links:
        assert (root/unquote(url.split('#')[0])).is_file(), url
    write_new(root/'independent-validation.json', dict(
        P0_errors=errors, P0_reference_units=units, P0_CER=errors/units,
        ignored=ignored, mixed_kept=mixed, pages=independent,
        gallery_cards=parser.cards, gallery_links_checked=len(parser.links),
        method='separate dynamic-programming distance, NFC/whitespace and ignore/assembly implementation; does not independently certify visual labels'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    verify(parser.parse_args().root)
