"""Artifact integrity, inert text encoding, and character diff fidelity."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from src.evaluation.ablation_html import differences, generate, build_payload


class HtmlComparisonTests(unittest.TestCase):
    def test_diff_segments_reconstruct_model_text_exactly(self):
        for before,after in [('이전 글자','새 글자'),('same','same'),('delete',''),('', '추가'),('<script>','&<>')]:
            left,right=differences(before,after)
            self.assertEqual(''.join(part['text'] for part in left),before)
            self.assertEqual(''.join(part['text'] for part in right),after)
        left,right=differences('same','same')
        self.assertFalse(any(part['changed'] for part in left+right))

    def test_report_is_static_preserves_coordinates_and_escapes_scripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'images').mkdir();(root/'results'/'one').mkdir(parents=True)
            image=root/'images'/'one.png';Image.new('RGB',(30,20),'white').save(image)
            manifest={'dataset':'test','revision':'fixed','seed':42,'population_rows':1,
                'samples':[{'id':'one','image':'images/one.png','sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                            'width':30,'height':20,'shard':'data/train.parquet','row':0}]}
            manifest_path=root/'manifest.json';manifest_path.write_text(json.dumps(manifest))
            bad='</script><script>alert("x")</script>'
            result={'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 'samples':[
                {'id':'one','paddle':{'status':'ok'},'qwen':{'status':'ok'},'baseline_text':bad,'hybrid_text':'교정'}]}
            results=root/'results'/'results.json';results.write_text(json.dumps(result))
            (root/'results/one/ocr.json').write_text(json.dumps({'lines':[{'id':'a','text':bad,'bbox':[[1,2],[9,2],[9,8],[1,8]]}]}))
            (root/'results/one/review.json').write_text(json.dumps({'vector_edits':[{'id':'a','text':'교정'}]}))
            payload=build_payload(manifest_path,results,root/'index.html')
            self.assertEqual(payload['samples'][0]['rows'][0]['bbox'],[1,2,9,8])
            self.assertEqual(payload['samples'][0]['image'],'images/one.png')
            output=generate(manifest_path,results,root/'index.html')
            html=output.read_text()
            self.assertNotIn(bad,html)
            self.assertIn('\\u003c/script\\u003e',html)
            self.assertNotIn('__REPORT_DATA__',html)
            self.assertNotIn('fetch(',html)
            references={'manifest_sha256':result['manifest_sha256'],'samples':[{'id':'one','text':'사람 정답','status':'verified','notes':None}]}
            reference_path=root/'references.json';reference_path.write_text(json.dumps(references))
            reference_hash=hashlib.sha256(reference_path.read_bytes()).hexdigest()
            (root/'results/summary.json').write_text(json.dumps({'accuracy_available':True,'reference_provenance':{'sha256':reference_hash}}))
            (root/'results/sample_scores.json').write_text(json.dumps({'samples':[{'id':'one','paddle':{'cer':.5},'hybrid':{'cer':.2}}]}))
            labelled=build_payload(manifest_path,results,root/'index.html',reference_path)
            self.assertEqual(labelled['samples'][0]['reference']['text'],'사람 정답')
            self.assertEqual(labelled['samples'][0]['scores']['hybrid']['cer'],.2)
            reference_path.write_text(json.dumps({**references,'notes':'changed'}))
            with self.assertRaises(ValueError):
                build_payload(manifest_path,results,root/'index.html',reference_path)
            result['manifest_sha256']='wrong';results.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                build_payload(manifest_path,results,root/'index.html')

if __name__=='__main__':
    unittest.main()
