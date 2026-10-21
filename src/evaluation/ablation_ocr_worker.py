"""Isolated OCR baseline worker; does not load or invoke Qwen."""
import argparse
import json
import time
from pathlib import Path
import cv2
from src.cv.ocr_engine import recognize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', default='gpu:0')
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f'Could not decode {args.image}')
    started = time.perf_counter()
    text, details = recognize(image, 'paddleocrvl16', device=args.device)
    result = {'text': text, 'lines': [{**line, 'id': f'ocr-text-{index+1:04d}'}
              for index, line in enumerate(details.get('lines', []))],
              'image': {'width': image.shape[1], 'height': image.shape[0]},
              'metadata': {**details.get('metadata', {}), 'adapter_wall_seconds': time.perf_counter()-started}}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
