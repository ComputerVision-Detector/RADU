"""Run page OCR with an explicitly selected exported Korean recognition model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import yaml

ROOT = Path(__file__).resolve().parents[3]


def validate_export(model_dir, config_path):
    infer_path = model_dir / 'inference.yml'
    if not infer_path.is_file() or not (model_dir / 'inference.pdiparams').is_file() or not any(
            (model_dir / name).is_file() for name in ('inference.json', 'inference.pdmodel')):
        raise FileNotFoundError(f'Exported custom model is missing: {model_dir}. Run scripts/export_ocr_model.py after training.')
    inference = yaml.safe_load(infer_path.read_text(encoding='utf-8'))
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    expected = Path(config['Global']['character_dict_path']).read_text(encoding='utf-8').splitlines()
    if inference['PostProcess']['character_dict'] != expected:
        raise ValueError('Training and inference dictionaries differ (including character order).')
    name = inference.get('Global', {}).get('model_name')
    if name != config['Global']['model_name']:
        raise ValueError(f'Export model_name differs from training configuration: {name}')
    provenance_path = model_dir / 'provenance.json'
    if not provenance_path.is_file():
        raise FileNotFoundError(f'Model provenance is missing: {provenance_path}')
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    checkpoint = Path(provenance['checkpoint'])
    if not checkpoint.is_file() or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != provenance['checkpoint_sha256']:
        raise ValueError(f'Export checkpoint does not match its provenance: {checkpoint}')
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != provenance['config_sha256']:
        raise ValueError('Export was produced using a different training config.')
    dictionary = Path(config['Global']['character_dict_path'])
    if hashlib.sha256(dictionary.read_bytes()).hexdigest() != provenance['dictionary_sha256']:
        raise ValueError('Export was produced using a different character dictionary.')
    return name, checkpoint.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', type=Path, default=ROOT / 'IMG_2756.JPG')
    parser.add_argument('--rec-model-dir', type=Path, default=ROOT / 'data/models/ocr_word_inference')
    parser.add_argument('--config', type=Path, default=ROOT / 'src/cv/ocr/custom_rec_korean.yml')
    parser.add_argument('--det-model-dir', type=Path,
                        default=Path.home() / '.paddlex/official_models/PP-OCRv5_server_det')
    parser.add_argument('--device', default='gpu:0')
    parser.add_argument('--min-score', type=float, default=0.0)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/ocr_predictions/custom_result.jpg')
    args = parser.parse_args()
    name, checkpoint = validate_export(args.rec_model_dir, args.config)
    if not args.det_model_dir.is_dir():
        raise FileNotFoundError(f'Local detector not found: {args.det_model_dir}')
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        text_detection_model_name='PP-OCRv5_server_det',
        text_detection_model_dir=str(args.det_model_dir.resolve()),
        text_recognition_model_name=name,
        text_recognition_model_dir=str(args.rec_model_dir.resolve()),
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, device=args.device, enable_mkldnn=False,
    )
    image = ImageOps.exif_transpose(Image.open(args.image)).convert('RGB')
    print(f'Recognition model: {name}')
    print(f'Trained checkpoint: {checkpoint}')
    # PaddleX expects BGR NumPy input. Keep coordinates on this same oriented image.
    results = ocr.predict(np.array(image)[:, :, ::-1].copy(), text_rec_score_thresh=args.min_score)
    draw = ImageDraw.Draw(image)
    font_path = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    font = ImageFont.truetype(str(font_path), 28) if font_path.exists() else ImageFont.load_default()
    rows = []
    for result in results:
        for text, score, polygon in zip(result['rec_texts'], result['rec_scores'], result['rec_polys']):
            points = np.asarray(polygon).astype(int).tolist()
            draw.line([tuple(p) for p in points] + [tuple(points[0])], fill='lime', width=3)
            draw.text((points[0][0], max(0, points[0][1]-32)), f'{text} ({score:.2f})', fill='red', font=font)
            rows.append({'text': text, 'confidence': float(score), 'polygon': points})
            print(f'{text}\t{score:.4f}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    args.output.with_suffix('.json').write_text(json.dumps({
        'image': str(args.image.resolve()), 'recognition_model': str(args.rec_model_dir.resolve()),
        'checkpoint': str(checkpoint),
        'results': rows}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
