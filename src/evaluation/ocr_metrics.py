"""Reference-based OCR metrics and explicitly non-accuracy change statistics."""
from __future__ import annotations
from collections import Counter
import re
import unicodedata
from rapidfuzz.distance import Levenshtein

NORMALIZATION = 'Unicode NFC; collapse whitespace to one space; strip; preserve case and punctuation'


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def error_counts(reference, hypothesis):
    operations = Counter(op.tag for op in Levenshtein.editops(reference, hypothesis))
    return {'substitutions': operations['replace'], 'deletions': operations['delete'],
            'insertions': operations['insert'], 'errors': sum(operations.values()), 'reference_units': len(reference)}


def score(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        raise ValueError('Reference must contain at least one non-whitespace character')
    chars = error_counts(reference, hypothesis)
    compact = error_counts(re.sub(r'\s+', '', reference), re.sub(r'\s+', '', hypothesis))
    words = error_counts(reference.split(), hypothesis.split())
    return {'cer': chars['errors']/chars['reference_units'],
            'cer_no_space': compact['errors']/compact['reference_units'],
            'wer': words['errors']/words['reference_units'], 'exact_match': reference == hypothesis,
            'characters': chars, 'characters_no_space': compact, 'words': words}


def aggregate(scores):
    if not scores:
        return None
    result = {'samples': len(scores), 'exact_match_rate': sum(s['exact_match'] for s in scores)/len(scores)}
    for name, field in [('cer','characters'), ('cer_no_space','characters_no_space'), ('wer','words')]:
        numerator = sum(s[field]['errors'] for s in scores)
        denominator = sum(s[field]['reference_units'] for s in scores)
        result[name+'_micro'] = numerator/denominator
        result[name+'_macro'] = sum(s[name] for s in scores)/len(scores)
        result[name+'_errors'] = numerator
        result[name+'_reference_units'] = denominator
    return result


def apply_review(ocr, review):
    """Match the production editor: keep OCR order/anchors, last valid edit per ID wins."""
    lines = ocr.get('lines', [])
    known = {line['id'] for line in lines}
    edits = {}
    unknown, invalid, duplicates = [], 0, []
    for item in review.get('vector_edits', []):
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not isinstance(item.get('text'), str):
            invalid += 1
            continue
        if item['id'] not in known:
            unknown.append(item['id'])
            continue
        if item['id'] in edits:
            duplicates.append(item['id'])
        edits[item['id']] = item['text']
    baseline = '\n'.join(str(line.get('text', '')) for line in lines)
    corrected = '\n'.join(edits.get(line['id'], str(line.get('text', ''))) for line in lines)
    changed = sum(edits.get(line['id'], line.get('text', '')) != line.get('text', '') for line in lines)
    before, after = normalize(baseline), normalize(corrected)
    changes = error_counts(before, after)
    return baseline, corrected, {'baseline_lines': len(lines), 'changed_lines': changed,
        'changed_line_rate': changed/len(lines) if lines else 0,
        'text_edit_distance': changes['errors'],
        'text_change_ratio': changes['errors']/len(before) if before else None,
        'unknown_edit_ids': unknown, 'invalid_edits': invalid, 'duplicate_edit_ids': duplicates,
        'uncertainty_count': len(review.get('uncertainties', []))}
