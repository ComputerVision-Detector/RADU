"""Create whole-word recognition crops; preserve all old character crops/labels."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DICT = ROOT / 'src/cv/ocr/PaddleOCR/ppocr/utils/dict/korean_dict.txt'


def word_sample(annotation, characters, max_length=25):
    text = annotation.get('text', {})
    if text.get('type') != 'word' or not text.get('word'):
        raise ValueError('not_word')
    entries = text['word']
    label = ''.join(item['value'] for item in entries)
    if not label.strip() or any(c in label for c in '\t\r\n'):
        raise ValueError('invalid_label')
    if len(label) > max_length:
        raise ValueError('too_long')
    if any(c not in characters for c in label):
        raise ValueError('out_of_dictionary')
    boxes = [item['charbox'] for item in entries]
    if any(len(b) != 4 or b[2] <= b[0] or b[3] <= b[1] for b in boxes):
        raise ValueError('invalid_box')
    # Preserve the annotated reading order, not an order inferred from pixels.
    return label, (min(b[0] for b in boxes), min(b[1] for b in boxes),
                   max(b[2] for b in boxes), max(b[3] for b in boxes))


def convert_one(args):
    path, label_root, image_root, out_root, characters, max_length = args
    try:
        annotation = json.loads(path.read_text(encoding='utf-8'))
        label, box = word_sample(annotation, characters, max_length)
    except ValueError as exc:
        return str(exc), None
    relative = path.relative_to(label_root)
    source = image_root / relative.parent / annotation.get('image', {}).get('file_name', relative.with_suffix('.jpg').name)
    image = cv2.imread(str(source))
    if image is None:
        return 'unreadable_image', None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        return 'box_out_of_bounds', None
    pad = max(2, round((y2 - y1) * 0.05))
    crop = image[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
    destination = out_root / relative.with_suffix('.jpg')
    if not cv2.imwrite(str(destination), crop, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f'Failed to write {destination}')
    return 'ok', (str(destination.resolve()), label, crop.shape[1] / crop.shape[0])


def convert_split(data_root, split, characters, workers=8, max_length=25):
    label_root = data_root / 'labels' / split
    image_root = data_root / 'images' / split
    out_root = data_root / 'word_crops' / split
    paths = sorted(label_root.rglob('*.json'))
    if not paths:
        raise FileNotFoundError(f'No annotations under {label_root}')
    for parent in {p.parent.relative_to(label_root) for p in paths}:
        (out_root / parent).mkdir(parents=True, exist_ok=True)
    output = data_root / f'rec_gt_word_{split}.txt'
    temporary = output.with_suffix('.txt.tmp')
    counts, lengths = Counter(), Counter()
    wide = 0
    with temporary.open('w', encoding='utf-8') as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(paths), 2048):
            jobs = ((p, label_root, image_root, out_root, characters, max_length)
                    for p in paths[start:start+2048])
            for status, result in pool.map(convert_one, jobs):
                counts[status] += 1
                if result:
                    path, label, ratio = result
                    handle.write(f'{path}\t{label}\n')
                    lengths[len(label)] += 1
                    wide += ratio > 320 / 48
            if start % 40960 == 0:
                print(f"{split}: {min(start+2048, len(paths))}/{len(paths)}, saved={counts['ok']}", flush=True)
    if not counts['ok'] or not any(length > 1 for length in lengths):
        raise RuntimeError(f'No multi-character samples: {counts}')
    temporary.replace(output)
    return {'counts': dict(counts), 'label_lengths': dict(sorted(lengths.items())),
            'aspect_ratio_over_320_48': wide, 'labels': str(output),
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/ai_hub_OCR')
    parser.add_argument('--dictionary', type=Path, default=DEFAULT_DICT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=25)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    characters = set(args.dictionary.read_text(encoding='utf-8').splitlines()) | {' '}
    report = {split: convert_split(args.data_root, split, characters, args.workers, args.max_length)
              for split in ('train', 'val')}
    report['dictionary_sha256'] = hashlib.sha256(args.dictionary.read_bytes()).hexdigest()
    (args.data_root / 'word_conversion_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
