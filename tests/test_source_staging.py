import json
from pathlib import Path
import tempfile
import unittest

from src.training.stage_sources import stage_flowlearn, load_rst_annotations, stage_ai2d


class SourceStagingTests(unittest.TestCase):
    def test_scientific_split_overlap_is_reported_not_silently_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "flowlearn/source/SciFlowchart"
            (source / "images").mkdir(parents=True)
            (source / "images/shared.png").write_bytes(b"fixture; existence check only")
            row = {"image_file":"shared.png", "file_name":"paper.json"}
            for split in ("train", "test"):
                (source / (split + ".json")).write_text(json.dumps([row]), encoding="utf-8")
            out = root / "output"
            report = stage_flowlearn(root, out)
            self.assertEqual(report["samples"], 2)
            self.assertEqual(report["scientific_unique_images"], 1)
            self.assertEqual(report["scientific_cross_split_images"], ["shared.png"])
            catalog = [json.loads(line) for line in (out / "flowlearn/catalog.jsonl").read_text().splitlines()]
            self.assertNotEqual(catalog[0]["id"], catalog[1]["id"])
            self.assertTrue(all(not item["training_ready"] and item["ocr"] is None for item in catalog))
            self.assertNotIn("\\", catalog[0]["image"])


class RstStagingTests(unittest.TestCase):
    def fixtures(self, root):
        source = root / "ai2d_rst/source/ai2d"
        rst = root / "ai2d_rst/ai2d-rst-v1-1/json/ai2d-rst"
        (source / "images").mkdir(parents=True)
        (source / "annotations").mkdir()
        rst.mkdir(parents=True)
        for image_id in ("0", "1"):
            (source / "images" / (image_id + ".png")).write_bytes(b"existence-only fixture")
            (source / "annotations" / (image_id + ".png.json")).write_text(json.dumps({"blobs":{"B0":{},"B1":{}}}))
            row = {"id":image_id, "grouping":[{"id":"B0","data":{"kind":"blobs"}}, {"id":"B1","data":{"kind":"blobs"}}],
                   "connectivity":None if image_id == "1" else {"nodes":[["B0",{}],["B1",{}]],"edges":[["B0","B1",{"kind":"bidirectional"}]]}, "rst":[]}
            (rst / (image_id + ".png.json")).write_text(json.dumps(row))
        (rst.parent / "categories_ai2d-rst.json").write_text(json.dumps({"0.png":"cycle","1.png":"illustration"}))
        return source, rst

    def test_null_connectivity_and_bidirectionality_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            source, rst = self.fixtures(Path(directory))
            records, stats = load_rst_annotations(source, rst)
            self.assertEqual(stats["rst_matched_images"], 2)
            self.assertEqual(stats["rst_connectivity_null"], 1)
            self.assertIsNone(records["1.png"]["record"]["connectivity"])
            self.assertEqual(stats["rst_edge_kinds"], {"bidirectional":1})

    def test_missing_source_image_fails_join(self):
        with tempfile.TemporaryDirectory() as directory:
            source, rst = self.fixtures(Path(directory))
            (source / "images/1.png").unlink()
            with self.assertRaisesRegex(ValueError, "image/segmentation missing"):
                load_rst_annotations(source, rst)

    def test_dangling_connectivity_fails_join(self):
        with tempfile.TemporaryDirectory() as directory:
            source, rst = self.fixtures(Path(directory))
            p = rst / "0.png.json"
            row = json.loads(p.read_text())
            row["connectivity"]["edges"][0][1] = "missing"
            p.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "dangling RST connectivity"):
                load_rst_annotations(source, rst)

    def test_joined_subset_catalog_keeps_null_as_unsupervised(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow required for staging image verification")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, rst = self.fixtures(root)
            for image_id in ("0", "1"):
                Image.new("RGB", (10,10)).save(source / "images" / (image_id + ".png"))
            out = root / "output"
            stats = stage_ai2d(root, out)
            rows = [json.loads(line) for line in (out / "ai2d_rst/rst_catalog.jsonl").read_text().splitlines()]
            self.assertEqual(stats["rst_annotations"], 2)
            self.assertEqual(rows[1]["rst_connectivity_status"], "null_not_supervised")
            self.assertIsNone(rows[1]["source_structure"]["connectivity"])
            self.assertFalse(rows[1]["training_ready"])


if __name__ == "__main__":
    unittest.main()
