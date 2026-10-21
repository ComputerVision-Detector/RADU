"""HybridOCR contract tests; model execution is replaced with deterministic evidence."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from src.hybrid import app as module
from src.hybrid.pipeline import validate_review, parse_review
from src.hybrid.pdf import render_pdf

class HybridTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch.object(module, 'STORE', Path(self.tmp.name))
        self.patch.start()
        self.client = TestClient(module.app)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_review_rejects_malformed_model_output(self):
        for payload in ({}, {'markdown':'hello','changes':'bad','uncertainties':[]}):
            with self.assertRaises(ValueError):
                validate_review(payload)

    def test_model_literal_backslash_preserves_content(self):
        raw = r'{"markdown":"수식 \( x \)","changes":[],"uncertainties":[]}'
        result = parse_review(raw)
        self.assertEqual(result['markdown'], r'수식 \( x \)')
        self.assertIn('format_repairs', result)
        with self.assertRaises(ValueError):
            parse_review('not a JSON document')

    def test_upload_save_and_pdf_uses_human_edits(self):
        def fake_process(path, dataset=None):
            (path/'ocr.json').write_text(json.dumps({'text':'OCR 원문'}))
            (path/'review.json').write_text(json.dumps({'markdown':'교정','changes':[], 'uncertainties':['검수 필요']}))
            (path/'final.json').write_text(json.dumps({'markdown':'교정','human_reviewed':False}))
            module.write_state(path,'ready')
        image=io.BytesIO()
        Image.new('RGB',(40,30),'white').save(image,format='PNG')
        with patch.object(module, 'process', side_effect=fake_process):
            response=self.client.post('/api/jobs',content=image.getvalue())
            import time
            for _ in range(100):
                current=self.client.get('/api/jobs/'+response.json()['id']).json()
                if current['status']=='ready':
                    break
                time.sleep(.01)
        self.assertEqual(response.status_code,200)
        job=response.json()['id']
        self.assertEqual(self.client.get(f'/api/jobs/{job}/pdf').status_code,409)
        final='사용자 수정\n| 항목 | English |\n| --- | --- |\n| 값 | value |\nA → B'
        self.assertEqual(self.client.put(f'/api/jobs/{job}',json={'markdown':final}).status_code,200)
        self.assertEqual(self.client.get(f'/api/jobs/{job}').json()['final']['markdown'],final)
        pdf=self.client.get(f'/api/jobs/{job}/pdf')
        self.assertEqual(pdf.status_code,200)
        self.assertTrue(pdf.content.startswith(b'%PDF-'))
        self.assertEqual(self.client.get(f'/api/jobs/{job}/image').status_code,200)

    def test_invalid_inputs_and_job_failures(self):
        self.assertEqual(self.client.post('/api/jobs',content=b'bad').status_code,400)
        self.assertEqual(self.client.get('/api/jobs/bad').status_code,400)
        self.assertEqual(self.client.get('/api/jobs/'+'a'*32).status_code,404)
        path=Path(self.tmp.name)/('b'*32)
        path.mkdir()
        module.write_state(path,'queued')
        with patch.object(module.subprocess,'Popen',side_effect=RuntimeError('mock GPU failure')):
            module.process(path)
        self.assertEqual(json.loads((path/'state.json').read_text())['status'],'failed')

    def test_parquet_sample_preserves_selected_row_and_revision(self):
        import pyarrow as arrow
        import pyarrow.parquet as parquet
        from types import SimpleNamespace
        from src.hybrid.dataset import load_sample
        payload = io.BytesIO()
        Image.new('RGB', (12, 8), 'white').save(payload, format='PNG')
        path = Path(self.tmp.name) / 'source.parquet'
        parquet.write_table(arrow.Table.from_pylist([
            {'image': {'bytes': payload.getvalue(), 'path': None}},
            {'image': {'bytes': payload.getvalue(), 'path': None}},
        ]), path)
        info = SimpleNamespace(sha='revision123', siblings=[SimpleNamespace(rfilename='data/train-00000-of-00001.parquet')])
        with patch('huggingface_hub.HfApi') as api, patch('huggingface_hub.hf_hub_download', return_value=str(path)) as download:
            api.return_value.dataset_info.return_value = info
            image, meta = load_sample('owner/data', None, 42)
            self.assertEqual(image.size, (12, 8))
            self.assertEqual(meta['revision'], 'revision123')
            self.assertIn(meta['row'], (0, 1))
            self.assertEqual(download.call_args.kwargs['revision'], 'revision123')

    def test_multipage_korean_pdf(self):
        self.assertTrue(render_pdf(('한국어 English → 다음\n'*180)).startswith(b'%PDF-'))

if __name__=='__main__':
    unittest.main()
