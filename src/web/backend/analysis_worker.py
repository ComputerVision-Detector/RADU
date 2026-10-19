"""Isolated per-request analysis process for the RADU web API."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.web.backend import app as api


def main() -> int:
    if len(sys.argv) != 8:
        print("Expected image, result, engine, device, session ID, source name, and session directory", file=sys.stderr)
        return 2
    image_path, result_path, engine, device, session_id, source_name, session_dir = sys.argv[1:]
    try:
        api.OUT = Path(session_dir)
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        document = api._analyze(bgr, engine, device, session_id, Path(image_path), source_name)
        Path(result_path).write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
