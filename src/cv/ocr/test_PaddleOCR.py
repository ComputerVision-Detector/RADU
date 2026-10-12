"""Run baseline inference with official PP-OCRv5 models, before fine-tuning."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = Path.home() / '.paddlex/official_models'
DETECTOR = MODEL_ROOT / 'PP-OCRv5_server_det'
RECOGNIZER = MODEL_ROOT / 'korean_PP-OCRv5_mobile_rec'
IMAGE_DIRS = (ROOT, Path(__file__).resolve().parent)
DEFAULT_IMAGES = [next((directory / name for directory in IMAGE_DIRS
                        if (directory / name).is_file()), ROOT / name)
                  for name in ('KakaoTalk_20261011_022928686.jpg',
                               'IMG_2756.JPG', 'IMG_2728.JPG')]


def run_one(ocr, image_path, output_dir, font):
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    # PaddleX's input is BGR; use the same EXIF-oriented pixels for drawing.
    result = ocr.predict(np.array(image)[:, :, ::-1].copy())
    draw = ImageDraw.Draw(image)
    rows = []
    for item in result:
        for text, score, polygon in zip(item['rec_texts'], item['rec_scores'], item['rec_polys']):
            points = np.asarray(polygon).astype(int).tolist()
            draw.line([tuple(p) for p in points] + [tuple(points[0])], fill='lime', width=3)
            draw.text((points[0][0], max(0, points[0][1]-32)), f'{text} ({score:.2f})', fill='red', font=font)
            rows.append({'text': text, 'confidence': float(score), 'polygon': points})
            print(f'{text}\t{score:.4f}')
    output = output_dir / f'{image_path.stem}_ppocrv5.jpg'
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output)
    report = {'image': str(image_path.resolve()),
              'detector': str(DETECTOR.resolve()),
              'recognizer': str(RECOGNIZER.resolve()),
              'fine_tuned': False, 'results': rows}
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved: {output}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', nargs='?', type=Path, help='Image path; omitted with --all-samples uses the three project photos.')
    parser.add_argument('--all-samples', action='store_true', help='Run KakaoTalk_20261011_022928686.jpg, IMG_2756.JPG and IMG_2728.JPG.')
    parser.add_argument('--device', default='gpu:0')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/ocr_predictions/ppocrv5_before_finetune')
    args = parser.parse_args()
    if args.image and args.all_samples:
        parser.error('Choose either one image or --all-samples.')
    images = DEFAULT_IMAGES if args.all_samples or not args.image else [args.image]
    missing = [path for path in (DETECTOR, RECOGNIZER) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f'Official PP-OCRv5 models are not cached: {missing}')
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
    from paddleocr import PaddleOCR
    print('Detector: PP-OCRv5_server_det (official, no fine-tuning)')
    print('Recognizer: korean_PP-OCRv5_mobile_rec (official, no fine-tuning)')
    print(f'Device: {args.device}')
    ocr = PaddleOCR(
        text_detection_model_name='PP-OCRv5_server_det',
        text_detection_model_dir=str(DETECTOR.resolve()),
        text_recognition_model_name='korean_PP-OCRv5_mobile_rec',
        text_recognition_model_dir=str(RECOGNIZER.resolve()),
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, device=args.device, enable_mkldnn=False,
    )
    font_path = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    font = ImageFont.truetype(str(font_path), 28) if font_path.exists() else ImageFont.load_default()
    for image in images:
        print(f'\n=== {image.name} ===')
        run_one(ocr, image.resolve(), args.output_dir.resolve(), font)


if __name__ == '__main__':
    main()
