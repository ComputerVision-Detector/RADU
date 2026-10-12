"""Evaluate recognition on held-out crops, retaining every prediction and error."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import cv2
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/cv/ocr'))
from recognition_runtime import RecognitionRuntime


def metrics(rows):
    return {'samples': len(rows),
            'exact_accuracy': sum(r['target'] == r['prediction'] for r in rows) / len(rows),
            'accuracy_ignore_spaces': sum(r['target'].replace(' ', '') == r['prediction'].replace(' ', '') for r in rows) / len(rows),
            'cer': sum(Levenshtein.distance(r['target'], r['prediction']) for r in rows) / sum(len(r['target']) for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'src/cv/ocr/custom_rec_korean.yml')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'data/models/ocr_word_finetuned/best_accuracy')
    parser.add_argument('--labels', type=Path, default=ROOT / 'data/ocr_real_eval/rec_gt.txt')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/ocr_real_eval/after_training.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    runtime = RecognitionRuntime(args.config, args.checkpoint, args.device)
    rows = []
    for line in args.labels.read_text(encoding='utf-8').splitlines():
        name, target = line.split('\t', 1)
        image = Path(name)
        if not image.is_absolute():
            image = ROOT / image
        prediction, score = runtime.predict(cv2.imread(str(image)))
        rows.append({'image': str(image), 'target': target, 'prediction': prediction, 'confidence': score})
    if not rows:
        raise ValueError('Empty evaluation set')
    by_page = defaultdict(list)
    for row in rows:
        by_page[Path(row['image']).stem.rsplit('_', 1)[0]].append(row)
    result = {'checkpoint': str(runtime.checkpoint), 'checkpoint_sha256': hashlib.sha256(runtime.checkpoint.read_bytes()).hexdigest(),
              'labels_sha256': hashlib.sha256(args.labels.read_bytes()).hexdigest(),
              'metrics': metrics(rows), 'by_page': {p: metrics(r) for p, r in by_page.items()}, 'predictions': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['metrics'], indent=2))
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
