import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.cv.detector import detect_components
from src.web.backend import app


class OcrFirstAnalysisTests(unittest.TestCase):
    def test_detector_can_omit_opencv_text_candidates_without_disabling_shapes(self):
        image = np.full((100, 240, 3), 255, dtype=np.uint8)
        cv2.putText(image, "FLOW", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)

        with patch("src.cv.detector.cv2.HoughLinesP", return_value=None):
            with_text = detect_components(image, include_text=True)
            without_text = detect_components(image, include_text=False)

        self.assertTrue(any(element["type"] == "Text" for element in with_text))
        self.assertFalse(any(element["type"] == "Text" for element in without_text))
        self.assertTrue(any(element["type"] == "Box" for element in without_text))

    def test_selected_ocr_is_the_only_source_of_text_elements(self):
        document = {
            "image": {"width": 20, "height": 10},
            "elements": [{"id": "box-1", "type": "Box", "bbox": [1, 1, 10, 8]}],
            "relations": [],
            "warnings": [],
        }
        details = {
            "lines": [{"text": "시작", "bbox": [[2, 2], [8, 2], [8, 6], [2, 6]]}],
            "metadata": {"model": "PaddleOCR-VL-1.6", "device": "gpu:0"},
        }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app, "OUT", Path(directory)), patch.object(
                app, "preprocess_image", side_effect=lambda image: image
            ), patch.object(app, "analyze_image", return_value=document) as analyze_image, patch.object(
                app, "recognize", return_value=("시작", details)
            ), patch.object(app, "analyze_relations", side_effect=lambda value: value):
                result = app._analyze(
                    np.zeros((10, 20, 3), dtype=np.uint8),
                    "paddleocrvl16",
                    "gpu:0",
                    "session-1",
                    Path(directory) / "source.png",
                    "source.png",
                )

        analyze_image.assert_called_once()
        self.assertFalse(analyze_image.call_args.kwargs["include_text"])
        self.assertEqual([element["type"] for element in result["elements"]], ["Box", "Text"])
        self.assertEqual(result["elements"][1]["text"], "시작")
        self.assertEqual(result["ocr"]["metadata"]["model"], "PaddleOCR-VL-1.6")
        self.assertEqual(result["ocr"]["metadata"]["device"], "gpu:0")

    def test_cv_text_proposals_remain_enabled_when_ocr_is_disabled(self):
        document = {
            "image": {"width": 20, "height": 10},
            "elements": [],
            "relations": [],
            "warnings": [],
        }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app, "OUT", Path(directory)), patch.object(
                app, "preprocess_image", side_effect=lambda image: image
            ), patch.object(app, "analyze_image", return_value=document) as analyze_image, patch.object(
                app, "analyze_relations", side_effect=lambda value: value
            ):
                app._analyze(
                    np.zeros((10, 20, 3), dtype=np.uint8),
                    "none",
                    "cpu",
                    "session-2",
                    Path(directory) / "source.png",
                    "source.png",
                )

        self.assertTrue(analyze_image.call_args.kwargs["include_text"])


if __name__ == "__main__":
    unittest.main()
