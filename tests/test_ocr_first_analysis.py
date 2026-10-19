import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.cv.detector import detect_components
from src.web.backend import app


class OcrFirstAnalysisTests(unittest.TestCase):
    def test_analysis_worker_output_is_forwarded_to_server_console(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "input.png"
            result_path = Path(directory) / "result.json"
            image_path.write_bytes(b"image")
            result_path.write_text(json.dumps({"elements": []}), encoding="utf-8")

            class CompletedWorker:
                returncode = 0

                @staticmethod
                def poll():
                    return 0

            def start_worker(command, **kwargs):
                kwargs["stdout"].write(b"[VL] active_device=gpu:0\n")
                return CompletedWorker()

            output = io.StringIO()
            with patch("src.web.backend.app.subprocess.Popen", side_effect=start_worker):
                with redirect_stdout(output):
                    result = asyncio.run(app._run_analysis_worker(
                        image_path,
                        result_path,
                        "paddleocrvl16",
                        "gpu:0",
                        "session-console",
                        "input.png",
                    ))

        self.assertEqual(result["elements"], [])
        self.assertIn("[VL] active_device=gpu:0", output.getvalue())

    def test_detector_can_omit_opencv_text_candidates_without_disabling_shapes(self):
        image = np.full((100, 240, 3), 255, dtype=np.uint8)
        cv2.putText(image, "FLOW", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)

        with patch("src.cv.detector.cv2.HoughLinesP", return_value=None):
            with_text = detect_components(image, include_text=True)
            without_text = detect_components(image, include_text=False)

        self.assertTrue(any(element["type"] == "Text" for element in with_text))
        self.assertFalse(any(element["type"] == "Text" for element in without_text))
        self.assertTrue(any(element["type"] == "Box" for element in without_text))

    def test_non_vl_ocr_remains_combined_with_detector_results(self):
        document = {
            "image": {"width": 20, "height": 10},
            "elements": [{"id": "box-1", "type": "Box", "bbox": [1, 1, 10, 8]}],
            "relations": [],
            "warnings": [],
        }
        details = {
            "lines": [{"text": "시작", "bbox": [[2, 2], [8, 2], [8, 6], [2, 6]]}],
            "metadata": {"model": "PP-OCRv6", "device": "gpu:0"},
        }

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app, "OUT", Path(directory)), patch.object(
                app, "preprocess_image", side_effect=lambda image: image
            ), patch.object(app, "analyze_image", return_value=document) as analyze_image, patch.object(
                app, "recognize", return_value=("시작", details)
            ), patch.object(app, "analyze_relations", side_effect=lambda value: value):
                result = app._analyze(
                    np.zeros((10, 20, 3), dtype=np.uint8),
                    "paddleocrv6",
                    "gpu:0",
                    "session-1",
                    Path(directory) / "source.png",
                    "source.png",
                )

        analyze_image.assert_called_once()
        self.assertFalse(analyze_image.call_args.kwargs["include_text"])
        self.assertEqual([element["type"] for element in result["elements"]], ["Box", "Text"])
        self.assertEqual(result["elements"][1]["text"], "시작")
        self.assertEqual(result["ocr"]["metadata"]["model"], "PP-OCRv6")
        self.assertEqual(result["ocr"]["metadata"]["device"], "gpu:0")

    def test_paddleocr_vl_only_overlays_ocr_and_skips_radu_processing(self):
        details = {
            "lines": [
                {"text": "시작", "bbox": [[2, 2], [8, 2], [8, 6], [2, 6]]},
                {"text": "잘못된 좌표", "bbox": []},
            ],
            "metadata": {"model": "PaddleOCR-VL-1.6", "device": "gpu:1"},
        }
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app, "OUT", Path(directory)), patch.object(
                app, "preprocess_image", side_effect=lambda image: image
            ), patch.object(app, "analyze_image") as analyze_image, patch.object(
                app, "recognize", return_value=("시작\n잘못된 좌표", details)
            ), patch.object(app, "analyze_relations") as analyze_relations:
                with redirect_stdout(output):
                    result = app._analyze(
                        np.zeros((10, 20, 3), dtype=np.uint8),
                        "paddleocrvl16",
                        "gpu:1",
                        "session-vl",
                        Path(directory) / "source.png",
                        "source.png",
                    )

        analyze_image.assert_not_called()
        analyze_relations.assert_not_called()
        self.assertEqual([element["type"] for element in result["elements"]], ["Text"])
        self.assertEqual(result["elements"][0]["text"], "시작")
        self.assertEqual(result["radu"]["status"], "not_run")
        self.assertIn("radu_not_run", {item["code"] for item in result["warnings"]})
        logs = output.getvalue()
        self.assertIn("ocr_success", logs)
        self.assertIn("engine=paddleocrvl16", logs)
        self.assertIn("device=gpu:1", logs)
        self.assertIn("lines=2 valid_boxes=1", logs)

    def test_paddleocr_vl_failure_is_reported_and_still_skips_radu(self):
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(app, "OUT", Path(directory)), patch.object(
                app, "preprocess_image", side_effect=lambda image: image
            ), patch.object(app, "analyze_image") as analyze_image, patch.object(
                app, "recognize", side_effect=RuntimeError("GPU initialization failed")
            ), patch.object(app, "analyze_relations") as analyze_relations:
                with redirect_stdout(output):
                    result = app._analyze(
                        np.zeros((10, 20, 3), dtype=np.uint8),
                        "paddleocrvl16",
                        "gpu:1",
                        "session-vl-error",
                        Path(directory) / "source.png",
                        "source.png",
                    )

        analyze_image.assert_not_called()
        analyze_relations.assert_not_called()
        self.assertNotIn("ocr", result)
        self.assertIn("ocr_unavailable", {item["code"] for item in result["warnings"]})
        self.assertIn("requested_device=gpu:1", output.getvalue())
        self.assertIn("GPU initialization failed", output.getvalue())

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
