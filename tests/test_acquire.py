import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.training.acquire import checksums, extract, fetch, materialize, safe_path
import json


class AcquisitionTests(unittest.TestCase):
    def test_traversal_and_windows_paths_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for name in ("../outside", "/absolute", "C:/outside", "a\\b", "a/../../b"):
                with self.assertRaises(ValueError, msg=name):
                    safe_path(root, name)
            self.assertEqual(safe_path(root, "fc_a/source/a.png"), Path(root).resolve() / "fc_a/source/a.png")

    def test_git_blob_checksum_and_existing_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "image.bin"
            path.write_bytes(b"test")
            actual = checksums(path)
            self.assertEqual(actual["git_blob_sha1"], hashlib.sha1(b"blob 4\0test").hexdigest())
            item = {"path":"image.bin", "url":"https://invalid.example/test", "sha256":"0" * 64}
            with self.assertRaisesRegex(ValueError, "does not match"):
                fetch(item, root)

    def test_unverified_existing_file_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "file").write_bytes(b"anything")
            with self.assertRaisesRegex(ValueError, "no upstream/prior checksum"):
                fetch({"path":"file", "url":"https://invalid.example/file"}, root)

    def test_zip_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "source.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("images/a.txt", "source data")
            out = Path(root) / "out"
            extract(archive, out)
            self.assertEqual((out / "images/a.txt").read_text(), "source data")
            with self.assertRaisesRegex(ValueError, "destination exists"):
                extract(archive, out)

    def test_malicious_zip_cannot_escape(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "source.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escaped.txt", "bad")
            with self.assertRaisesRegex(ValueError, "unsafe"):
                extract(archive, Path(root) / "out")
            self.assertFalse((Path(root) / "escaped.txt").exists())

    def test_lfs_pointer_is_not_materialized_as_data(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkout = root / "checkout"
            checkout.mkdir()
            pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:test\nsize 1\n"
            (checkout / "image.jpeg").write_bytes(pointer)
            plan = root / "plan.jsonl"
            plan.write_text(json.dumps({"dataset":"flowlearn", "path":"flowlearn/source/image.jpeg", "url":"https://invalid.example/image", "size":len(pointer)}) + "\n")
            failures = materialize(plan, checkout, root, "flowlearn/source")
            self.assertEqual(len(failures), 1)
            self.assertFalse((root / "flowlearn/source/image.jpeg").exists())

    def test_materialize_checks_source_hash_and_copies(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkout = root / "checkout"
            checkout.mkdir()
            (checkout / "asset").write_bytes(b"actual data")
            plan = root / "plan.jsonl"
            plan.write_text(json.dumps({"dataset":"flowlearn", "path":"flowlearn/source/asset", "url":"https://invalid.example/asset", "sha256":hashlib.sha256(b"actual data").hexdigest()}) + "\n")
            self.assertEqual(materialize(plan, checkout, root, "flowlearn/source"), [])
            self.assertEqual((root / "flowlearn/source/asset").read_bytes(), b"actual data")

    def test_sha_pinned_lfs_cache_can_supply_real_data(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            checkout = root / "checkout"
            checkout.mkdir()
            (checkout / "asset").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
            payload = b"actual cached binary"
            digest = hashlib.sha256(payload).hexdigest()
            objects = root / "objects"
            obj = objects / digest[:2] / digest[2:4] / digest
            obj.parent.mkdir(parents=True)
            obj.write_bytes(payload)
            plan = root / "plan.jsonl"
            plan.write_text(json.dumps({"dataset":"flowlearn", "path":"flowlearn/source/asset", "url":"https://invalid.example/asset", "sha256":digest}) + "\n")
            self.assertEqual(materialize(plan, checkout, root, "flowlearn/source", objects), [])
            self.assertEqual((root / "flowlearn/source/asset").read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
