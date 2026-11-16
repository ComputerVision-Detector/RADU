import copy
import unittest
from src.evaluation import ocr_scope_recovery as s

class ScopeRecoveryTests(unittest.TestCase):
    def test_recovery_exact_and_changed_geometry_rejected(self):
        old={'pages':[{'id':'p','regions':[{'region_id':'det-0000','bbox':[0,0,10,10],'text':'가','crop_sha256':'hash'}]}]}
        raw=copy.deepcopy(old);raw['pages'][0]['regions'].append({'region_id':'det-0001','bbox':[20,0,30,10],'text':'','crop_sha256':'extra'})
        shared,extra=s.recovery_map(old,raw)
        self.assertEqual(shared,['p/det-0000']);self.assertEqual(extra,['p/det-0001'])
        raw['pages'][0]['regions'][0]['bbox'][2]=11
        with self.assertRaisesRegex(ValueError,'geometry'):s.recovery_map(old,raw)
    def test_duplicate_and_missing_recovery_ids_rejected(self):
        p={'pages':[{'id':'p','regions':[{'region_id':'x','bbox':[0,0,10,10],'crop_sha256':'x'}]}]}
        with self.assertRaises(ValueError):s.recovery_map(p,{'pages':[]})
        q=copy.deepcopy(p);q['pages'][0]['regions']*=2
        with self.assertRaises(ValueError):s.recovery_map(p,q)
    def test_merge_preserves_every_cached_string(self):
        old={'pages':[{'id':'p','regions':[{'region_id':'x','bbox':[0,0,10,10],'text':'OLD','order':0,'crop_sha256':'x'}]}]}
        raw=copy.deepcopy(old);raw['pages'][0]['regions'][0]['text']=''
        raw['pages'][0]['regions'].append({'region_id':'y','bbox':[0,20,10,30],'text':'','order':1,'crop_sha256':'y'})
        full=s.merge_bank(old,raw,{'p/y':'NEW'})
        self.assertEqual([r['text'] for r in full['pages'][0]['regions']],['OLD','NEW'])
        with self.assertRaises(ValueError):s.merge_bank(old,raw,{'p/x':'CHANGED','p/y':'NEW'})
    def test_gt_revision_never_modifies_input_and_requires_evidence(self):
        gt={'pages':[{'id':'p','regions':[{'annotation_id':'a','text':'bad','include':False,'ignore_reason':'uncertain'}]}]}
        with self.assertRaises(ValueError):s.revise_gt(gt,[{'image_id':'p','annotation_id':'a','text':'good','include':True}])
        revised,changes=s.revise_gt(gt,[{'image_id':'p','annotation_id':'a','text':'good','include':True,'evidence':'original image literal review'}])
        self.assertEqual(gt['pages'][0]['regions'][0]['text'],'bad');self.assertEqual(revised['pages'][0]['regions'][0]['text'],'good');self.assertEqual(changes[0]['before']['text'],'bad')
    def test_scope_comparison_rejects_changed_shared_texts(self):
        gt=[{'id':'p','regions':[{'annotation_id':'g','bbox':[0,0,10,10],'text':'가','order':0,'include':True}]}]
        before={'pages':[{'id':'p','regions':[{'region_id':'x','bbox':[0,0,10,10],'text':'가','order':0,'crop_sha256':'h'}]}]}
        after=copy.deepcopy(before);after['pages'][0]['regions'][0]['text']='나'
        with self.assertRaisesRegex(ValueError,'shared recognition drift'):s.paired_scope_compare(gt,before,after)
        after=copy.deepcopy(before);after['pages'][0]['regions'].append({'region_id':'y','bbox':[20,0,30,10],'text':'오탐','order':1,'crop_sha256':'y'})
        d=s.paired_scope_compare(gt,before,after)
        self.assertEqual(d['before']['page']['reference_units'],d['after']['page']['reference_units'])
        self.assertGreater(d['after']['page']['insertions'],0)
    def test_review_freeze_drift_rejected_before_inference(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name in ['prior-gt-v2.json','audited-gt-v3.json','review37-decisions.json']:(p/name).write_text('{}')
            s.write_new(p/'review-freeze.json',{'GTv2':s.sha256(p/'prior-gt-v2.json'),'GTv3':s.sha256(p/'audited-gt-v3.json'),'decisions':s.sha256(p/'review37-decisions.json')})
            s.verify_review_freeze(p)
            (p/'audited-gt-v3.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'review freeze drift'):s.verify_review_freeze(p)
