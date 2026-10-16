# RADU local model assets

These model directories are local runtime assets. `.gitignore` excludes their
weights from source control; keep the upstream metadata and license files with
each model directory when moving or backing up this project.

| Directory | Purpose | Approximate size |
| --- | --- | ---: |
| `detection/PP-OCRv6_medium_det/` | PP-OCRv6 text detection | 60 MB |
| `recognition/PP-OCRv6_medium_rec/` | PP-OCRv6 text recognition | 74 MB |
| `detection/PP-DocLayoutV3/` | PaddleOCR-VL 1.6 layout analysis | 126 MB |
| `vision_language/PaddleOCR-VL-1.6/` | PaddleOCR-VL 1.6 recognition model and processor | 1.8 GB |

`src/cv/ocr_engine.py` loads these paths directly. It does not use another
project's model cache. PaddleOCR/PaddlePaddle Python packages remain environment
dependencies and are listed in the root `requirements.txt`.
