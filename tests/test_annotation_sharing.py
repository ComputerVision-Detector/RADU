"""CPU-only collaboration contracts; no OCR or network needed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from src.evaluation.annotation_files import check
from src.evaluation.annotation_images import restore

class AnnotationSharingTests(unittest.TestCase):
    def test_strict_inventory_validation_rejects_unexpected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); labels=root/'labels';labels.mkdir();manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'samples':[{'id':'hf-0000'}]}))
            sample={'id':'hf-0000','text':None,'status':'unlabeled','annotator':'','notes':''}
            (labels/'hf-0000.json').write_text(json.dumps(sample))
            self.assertEqual(len(check(labels,manifest,strict=True)),1)
            (labels/'typo.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Unknown label files'):check(labels,manifest,strict=True)
            (labels/'typo.json').unlink();sample.update(status='verified')
            (labels/'hf-0000.json').write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError,'Empty verified text'):check(labels,manifest,strict=True)

    def test_restorer_verifies_existing_files_and_never_replaces_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'images').mkdir();image=root/'images/hf-0000.png';image.write_bytes(b'frozen')
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'samples':[{'id':'hf-0000','image':'images/hf-0000.png','sha256':hashlib.sha256(b'frozen').hexdigest()}]}))
            self.assertEqual(restore(manifest),{'restored':0,'verified':1})
            with self.assertRaisesRegex(ValueError,'Unknown image IDs'):restore(manifest,identities=['hf-0001'])
            image.write_bytes(b'user-edited')
            with self.assertRaisesRegex(ValueError,'not overwritten'):restore(manifest)
            self.assertEqual(image.read_bytes(),b'user-edited')

if __name__=='__main__':unittest.main()
