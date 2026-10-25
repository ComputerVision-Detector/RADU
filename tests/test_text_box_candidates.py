"""Proposal coordinates, provenance, resume and immutable output without a model."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.evaluation import text_box_candidates as proposals


class ProposalTests(unittest.TestCase):
    def test_coordinates_clip_round_outward_and_reject_degenerate_boxes(self):
        self.assertEqual(proposals._xyxy([[-1, 1.1], [4.1, 1], [4, 5]], 10, 10, 20, 30), [0, 3, 9, 15])
        self.assertEqual(proposals._xyxy([1, 2, 5, 6], 10, 10, 20, 30), [2, 6, 10, 18])
        self.assertIsNone(proposals._xyxy([0, 0, float('nan'), 2], 10, 10, 20, 30))
        self.assertIsNone(proposals._xyxy([1, 1, 1, 2], 10, 10, 20, 30))

    def test_generation_fingerprints_resume_and_never_changes_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'hf-0000.webp'
            cv2.imwrite(str(image), np.full((20, 30, 3), 255, dtype=np.uint8))
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            items = {'hf-0000': {'image': {'file': 'assets/hf-0000.webp', 'source_width': 60, 'source_height': 40}}}
            assets = {'hf-0000': {'sha256': digest, 'source_sha256': 'a'*64, 'width': 30, 'height': 20}}
            output = root / 'proposals.json'
            details = {'model': 'PaddleOCR-VL-1.6', 'device': 'gpu:0', 'lines': [
                {'text': 'OCR 원문', 'bbox': [1, 2, 10, 8]},
                {'text': 'invalid', 'bbox': []}]}
            with patch.object(proposals, '_load_catalog', return_value=(items, assets)), \
                    patch.object(proposals, '_vl_batches', return_value=iter([details])):
                result = proposals.generate(root, 'paddleocrvl16', output, 'gpu:0')
                record = result['images']['hf-0000']
                self.assertEqual(record['lines'][0]['bbox'], [2, 4, 20, 16])
                self.assertEqual(record['lines'][0]['prediction'], 'OCR 원문')
                self.assertNotIn('text', record['lines'][0])
                self.assertEqual(record['input_sha256'], record['asset_sha256'])
                self.assertEqual(record['skipped_lines'], 1)
                before = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    proposals.generate(root, 'paddleocrvl16', output)
                self.assertEqual(output.read_bytes(), before)
                output.unlink()
                # A complete checkpoint needs no inference and yields the same records.
                resumed = proposals.generate(root, 'paddleocrvl16', output, resume=True)
                self.assertEqual(resumed['images'], result['images'])
                output.unlink()
                image.write_bytes(b'changed input')
                with self.assertRaisesRegex(ValueError, 'fingerprint'):
                    proposals.generate(root, 'paddleocrvl16', output, resume=True)


if __name__ == '__main__':
    unittest.main()
