from pathlib import Path
import unittest

import numpy as np

from src.web.backend import labeled_dataset as ld

HAS_DATA = (ld.DATASET / "pages.jsonl").is_file() and (ld.IMAGES / "hf-0137.png").is_file()
PAGE = {"id": "hf-9999", "split": "test", "annotator": "a", "status": "verified", "eligible": True, "hold_reasons": [], "no_text": False,
        "regions": [{"id": "r1", "bbox": [10, 10, 110, 40], "text": "가나다라", "order": 1, "type_suggestion": "text", "hold_reasons": []},
                    {"id": "r2", "bbox": [10, 60, 110, 90], "text": "x=1", "order": 2, "type_suggestion": "formula", "hold_reasons": []},
                    {"id": "r3", "bbox": [10, 110, 110, 140], "text": "놓침", "order": 3, "type_suggestion": "text", "hold_reasons": []},
                    {"id": "r4", "bbox": [10, 160, 110, 190], "text": "보류", "order": 4, "type_suggestion": "text", "hold_reasons": ["review"]}]}


class ScoringTests(unittest.TestCase):
    def test_matched_missed_extra_and_held_regions_are_counted_separately(self):
        lines = [{"bbox": [[10, 10], [110, 10], [110, 40], [10, 40]], "text": "가나다마"},
                 {"bbox": [12, 61, 108, 89], "text": "x=1"},
                 {"bbox": [20, 165, 100, 185], "text": "무시"},
                 {"bbox": [150, 10, 190, 40], "text": "추가"}]
        result = ld.evaluate_page(PAGE, lines, np.eye(3).tolist(), 200, 200)
        detection = result["detection"]
        self.assertEqual((detection["matched"], detection["references"], detection["predictions"], detection["ignored_predictions"], detection["held_references"]), (2, 3, 3, 1, 1))
        self.assertAlmostEqual(result["matched_regions"]["cer_micro"], 1 / 7)
        self.assertAlmostEqual(result["all_regions"]["cer_micro"], 3 / 9)  # the missed line counts as deleted
        self.assertEqual({k: v["samples"] for k, v in result["by_type"].items()}, {"formula": 1, "text": 1})
        self.assertEqual([p["text"] for p in result["extra_predictions"]], ["추가"])
        self.assertEqual([r["prediction"] for r in result["regions"]], ["가나다마", "x=1", None, None])

    def test_label_boxes_follow_the_page_transform(self):
        shift = [[1, 0, 5], [0, 1, -5], [0, 0, 1]]
        self.assertEqual(ld.transform_box([10, 10, 20, 20], shift, 100, 100), [15, 5, 25, 15])
        lines = [{"bbox": [15, 5, 115, 35], "text": "가나다라"}]
        self.assertEqual(ld.evaluate_page(PAGE, lines, shift, 200, 200)["regions"][0]["errors"], 0)


@unittest.skipUnless(HAS_DATA, "label snapshot and restored originals are not on this machine")
class DatasetTests(unittest.TestCase):
    def test_listing_matches_the_frozen_labels(self):
        listing = ld.list_pages()
        self.assertEqual((listing["dataset"], len(listing["pages"])), ("hakuuung/HybridOCR_CustomNotes", 163))
        self.assertEqual({page["split"] for page in listing["pages"]}, {"train", "validation", "test"})
        with self.assertRaises(KeyError):
            ld.get_page("../hf-0000")

    def test_recorded_transform_reproduces_preprocessing_and_restores_cv2(self):
        import cv2
        from src.cv import preprocess
        image = cv2.imread(str(ld.image_path("hf-0137")))
        matrix, size, steps = ld.preprocess_geometry(image)
        self.assertIs(preprocess.cv2, cv2)
        expected = preprocess.correct_rotation(preprocess.correct_perspective(image))
        self.assertEqual(size, (expected.shape[1], expected.shape[0]))
        mine = cv2.warpPerspective(image, matrix, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) if steps else image
        self.assertLess(float(np.abs(expected.astype(int) - mine.astype(int)).mean()), 0.05)

    def test_api_lists_pages_and_rejects_unknown_ones(self):
        from fastapi.testclient import TestClient
        from src.web.backend.app import app
        client = TestClient(app)
        self.assertEqual(len(client.get("/api/labeled-dataset").json()["pages"]), 163)
        self.assertEqual(client.get("/api/labeled-dataset/hf-9999/image").status_code, 404)
        self.assertEqual(client.post("/api/sessions/" + "0" * 32 + "/evaluate-labels?page=hf-0137").status_code, 404)


if __name__ == "__main__":
    unittest.main()
