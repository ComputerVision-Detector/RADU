"""Coordinate-grounded review, geometry persistence, and vector export contracts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree
from PIL import Image
from fastapi.testclient import TestClient
from src.hybrid import app as api
from src.hybrid.document import build_document


class VectorReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Path(self.tmp.name)
        self.patch = patch.object(api, 'STORE', self.store)
        self.patch.start()
        self.id = 'c' * 32
        self.path = self.store / self.id
        self.path.mkdir()
        Image.new('RGB', (400, 300), 'white').save(self.path / 'source.png')
        self.ocr = {'text':'원문\n셀', 'lines':[
            {'id':'ocr-text-0001','text':'원문','bbox':[[20,20],[150,20],[150,60],[20,60]]},
            {'id':'ocr-text-0002','text':'셀','bbox':[25,105,90,125]},
        ]}
        self.review = {'markdown':'', 'vector_edits':[{'id':'ocr-text-0001','text':'교정 결과','uncertain':False}],
                       'changes':[], 'uncertainties':[], 'regions':[
            {'type':'Table','cells':[{'bbox':[20,100,180,160],'text':'표 셀'}]},
            {'type':'Arrow','bbox':[210,100,300,180],'start':[210,100],'end':[300,180]},
            {'type':'Box','bbox':[-20,0,20,20],'text':'invalid'},
        ]}
        for name, data in [('ocr',self.ocr),('review',self.review),('state',{'id':self.id,'status':'ready'})]:
            (self.path / f'{name}.json').write_text(json.dumps(data,ensure_ascii=False))
        self.client = TestClient(api.app)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_ocr_anchor_and_qwen_cell_ownership(self):
        doc = build_document(self.path/'source.png',self.ocr,self.review,self.id)
        self.assertEqual(doc['elements'][0]['bbox'],[20,20,150,60])
        self.assertEqual(doc['elements'][0]['text'],'교정 결과')
        self.assertEqual(doc['elements'][0]['original_text'],'원문')
        self.assertFalse(any(e['id']=='ocr-text-0002' for e in doc['elements']))
        self.assertTrue(any(e.get('semantic_type')=='table-cell' and e['text']=='표 셀' for e in doc['elements']))
        self.assertTrue(any(e['type']=='Arrow' for e in doc['elements']))
        self.assertTrue(any(w['code']=='invalid_region' for w in doc['warnings']))

    def test_anchored_layout_candidates_keep_geometry_and_empty_edits_are_current_format(self):
        ocr = {**self.ocr, 'layout': [
            {'id':'layout-0001', 'type':'Box', 'shape':'decision', 'bbox':[200,20,300,90], 'source':'opencv-proposal'},
            {'id':'layout-0002', 'type':'Arrow', 'bbox':[210,100,300,180], 'start':[210,100], 'end':[300,180], 'source':'opencv-arrowhead-proposal'},
        ]}
        review = {'markdown':'', 'vector_edits':[], 'regions':[], 'changes':[], 'uncertainties':[]}
        doc = build_document(self.path/'source.png',ocr,review,self.id)
        box = next(e for e in doc['elements'] if e['id']=='layout-0001')
        self.assertEqual(box['bbox'],[200,20,300,90])
        self.assertEqual(box['shape'],'decision')
        self.assertFalse(any(w['code']=='legacy_review' for w in doc['warnings']))
        self.assertTrue(any(w['code']=='layout_proposals' for w in doc['warnings']))
        self.assertTrue(any(e['text']=='원문' for e in doc['elements']))

    def test_visual_save_exports_actual_edited_geometry_and_checks_revision(self):
        base=f'/api/jobs/{self.id}/vector'
        doc=self.client.get(base).json()
        self.assertEqual(len(doc['elements']),3)
        doc['elements'][0]['bbox']=[50,40,200,80]
        doc['elements'][0]['text']='직접 수정'
        result=self.client.put(base+'?reviewed=true',json=doc)
        self.assertEqual(result.status_code,200,result.text)
        self.assertEqual(self.client.put(base,json=doc).status_code,409)
        stored=self.client.get(base).json()
        self.assertEqual(stored['elements'][0]['text'],'직접 수정')
        self.assertEqual(stored['elements'][0]['bbox'],[50,40,200,80])
        self.assertTrue(stored['human_reviewed'])
        response=self.client.post(base+'/export/svg',json=stored)
        self.assertEqual(response.status_code,200,response.text)
        root=ElementTree.fromstring(response.content)
        self.assertEqual(root.attrib['viewBox'],'0 0 400 300')
        self.assertIn('직접 수정',response.text)
        self.assertNotIn('<image',response.text)
        response=self.client.post(base+'/export/pdf',json=stored)
        self.assertEqual(response.status_code,200)
        self.assertTrue(response.content.startswith(b'%PDF-'))
        self.assertEqual(self.client.get(f'/api/jobs/{self.id}/pdf').status_code,200)
        stored['elements'][0]['bbox']=[70,40,200,80]
        self.assertEqual(self.client.put(base,json=stored).status_code,200)
        self.assertFalse(self.client.get(base).json()['human_reviewed'])
        self.assertEqual(self.client.get(f'/api/jobs/{self.id}/pdf').status_code,409)

    def test_visual_editor_and_invalid_coordinates(self):
        response=self.client.get('/review/'+self.id)
        self.assertEqual(response.status_code,200)
        self.assertIn('window.HYBRID_REVIEW',response.text)
        self.assertIn('hybrid-table',response.text)
        self.assertNotIn('id="editor"',self.client.get('/').text)
        doc=self.client.get(f'/api/jobs/{self.id}/vector').json()
        doc['elements'][0]['bbox']=[0,0,999,999]
        self.assertEqual(self.client.put(f'/api/jobs/{self.id}/vector',json=doc).status_code,422)

if __name__=='__main__':
    unittest.main()
