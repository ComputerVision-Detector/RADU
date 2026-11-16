import tempfile
import unittest
from pathlib import Path
from src.evaluation.ocr_page_audit_replay import check_hashes
from src.ocr_data.audit import sha256


class ReplayTests(unittest.TestCase):
    def test_cached_prediction_drift_and_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prediction.json'
            path.write_text('{"text":"가"}')
            lock = {'prediction.json':sha256(path)}
            check_hashes(directory, lock)
            path.write_text('{"text":"나"}')
            with self.assertRaisesRegex(ValueError, 'drift'):
                check_hashes(directory, lock)
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'drift'):
                check_hashes(directory, lock)
