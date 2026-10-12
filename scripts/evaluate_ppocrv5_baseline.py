"""Compare official Korean PP-OCRv5 recognition on the same manual holdout."""
import hashlib
import json
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from paddleocr import TextRecognition

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path.home() / '.paddlex/official_models/korean_PP-OCRv5_mobile_rec'
LABELS = ROOT / 'data/ocr_real_eval/rec_gt.txt'


def main():
    rows = []
    model = TextRecognition(model_name='korean_PP-OCRv5_mobile_rec', model_dir=str(MODEL), device='gpu:0')
    for line in LABELS.read_text(encoding='utf-8').splitlines():
        name, target = line.split('\t', 1)
        output = model.predict(input=name)
        item = output[0]
        result = item.json if hasattr(item, 'json') else item
        if callable(result):
            result = result()
        result = result.get('res', result) if isinstance(result, dict) else result
        prediction = result.get('rec_text', result.get('text', ''))
        score = result.get('rec_score', result.get('score', 0.0))
        rows.append({'image': name, 'target': target, 'prediction': prediction, 'confidence': float(score)})
    model.close()
    result = {'model': 'korean_PP-OCRv5_mobile_rec', 'model_dir': str(MODEL),
              'metrics': {'samples': len(rows),
                          'exact_accuracy': sum(r['target'] == r['prediction'] for r in rows) / len(rows),
                          'accuracy_ignore_spaces': sum(r['target'].replace(' ', '') == r['prediction'].replace(' ', '') for r in rows) / len(rows),
                          'cer': sum(Levenshtein.distance(r['target'], r['prediction']) for r in rows) / sum(len(r['target']) for r in rows)},
              'predictions': rows, 'labels_sha256': hashlib.sha256(LABELS.read_bytes()).hexdigest()}
    output = ROOT / 'data/ocr_real_eval/ppocrv5_baseline.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result['metrics'], indent=2))
    for row in rows:
        print(f"{row['target']} => {row['prediction']} ({row['confidence']:.3f})")


if __name__ == '__main__':
    main()
