import hashlib
import io
import json
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
class Selects(HTMLParser):
    def __init__(self):super().__init__();self.select=None;self.option=None;self.values={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='select':self.select=a.get('id');self.values[self.select]=[]
        if tag=='option' and self.select:self.option=dict(a,text='');self.values[self.select].append(self.option)
    def handle_data(self,data):
        if self.option is not None:self.option['text']+=data
    def handle_endtag(self,tag):
        if tag=='option':self.option=None
        if tag=='select':self.select=None
def page():
    p=Selects();p.feed((ROOT/'src/web/frontend/index.html').read_text());return p.values
class FinalModesTests(unittest.TestCase):
    def test_auto_is_default_and_korean_mode_is_named(self):
        opts=page()['ocr-engine'];selected=[o for o in opts if 'selected' in o] or [opts[0]]
        self.assertEqual([o['value'] for o in selected],['paddleocrvl16routed'])
        self.assertIn('자동 모드',next(o['text'] for o in opts if o['value']=='paddleocrvl16routed'))
        self.assertIn('한국어 전용',next(o['text'] for o in opts if o['value']=='paddleocrvl16lora'))
    def test_legacy_engines_preserved_and_diagnostic_candidates_not_exposed(self):
        values={o['value'] for o in page()['ocr-engine']}
        self.assertEqual(values,{'paddleocrvl16routed','paddleocrvl16lora','none','paddleocrv6','paddleocrvl16','clova'})
    def test_two_modes_dispatch_to_existing_fixed_paths(self):
        import numpy as np
        from src.cv.ocr_engine import recognize
        frame=np.zeros((2,2,3),dtype=np.uint8)
        with patch('src.cv.ocr_dual_route.recognize_routed',return_value=('auto',{})) as auto,patch('src.cv.ocr_lora.recognize_lora',return_value=('ko',{})) as ko:
            self.assertEqual(recognize(frame,'paddleocrvl16routed','gpu:0')[0],'auto');self.assertEqual(recognize(frame,'paddleocrvl16lora','gpu:0')[0],'ko')
            auto.assert_called_once_with(frame,'gpu:0');ko.assert_called_once_with(frame,'gpu:0')

class ReleaseTests(unittest.TestCase):
    def fixture(self,folder):
        root=Path(folder);experiment=root/'experiment';experiment.mkdir();(root/'configs').mkdir();(root/'ui').write_text('old ui');(experiment/'binding').write_text('frozen weights/config')
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        deploy=dict(files={'binding':sha(experiment/'binding')},installed_files={str(root/'ui'):sha(root/'ui')},engine='paddleocrvl16routed',parent_sha256='a'*64,candidate_sha256='b'*64)
        path=experiment/'deployment.json';path.write_text(json.dumps(deploy));approved=sha(path)
        (root/'ui').write_text('new labels only')
        release=dict(schema=1,engine=deploy['engine'],base_deployment_sha256=approved,parent_sha256=deploy['parent_sha256'],candidate_sha256=deploy['candidate_sha256'],installed_files={'ui':sha(root/'ui')})
        lock=root/'configs/ocr_cv_final.lock.json';lock.write_text(json.dumps(release));return root,experiment,path,approved,lock
    def test_new_ui_binding_preserves_original_deployment_and_rejects_drift(self):
        from src.cv import ocr_dual_route_worker as w
        self.assertTrue(hasattr(w,'verify_deployment'),'versioned final UI binding verifier missing')
        with tempfile.TemporaryDirectory() as d:
            root,exp,path,h,lock=self.fixture(d);before=path.read_bytes()
            with patch.object(w,'FROZEN_DEPLOYMENT_SHA256',h):
                result=w.verify_deployment(root,exp);self.assertEqual(result['parent_sha256'],'a'*64)
                self.assertEqual(path.read_bytes(),before)
                (root/'ui').write_text('unapproved UI')
                with self.assertRaises(ValueError):w.verify_deployment(root,exp)
    def test_release_cannot_replace_frozen_expert_identity(self):
        from src.cv import ocr_dual_route_worker as w
        self.assertTrue(hasattr(w,'verify_deployment'),'versioned verifier missing')
        with tempfile.TemporaryDirectory() as d:
            root,exp,path,h,lock=self.fixture(d);x=json.loads(lock.read_text());x['candidate_sha256']='c'*64;lock.write_text(json.dumps(x))
            with patch.object(w,'FROZEN_DEPLOYMENT_SHA256',h):
                with self.assertRaises(ValueError):w.verify_deployment(root,exp)
    def test_frozen_binding_drift_still_fails_before_model_load(self):
        from src.cv import ocr_dual_route_worker as w
        self.assertTrue(hasattr(w,'verify_deployment'),'versioned verifier missing')
        with tempfile.TemporaryDirectory() as d:
            root,exp,path,h,lock=self.fixture(d);(exp/'binding').write_text('changed model')
            with patch.object(w,'FROZEN_DEPLOYMENT_SHA256',h):
                with self.assertRaises(ValueError):w.verify_deployment(root,exp)
if __name__=='__main__':unittest.main()
