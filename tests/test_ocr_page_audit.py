import unittest
from src.evaluation import ocr_page_audit as a

class PageAuditTests(unittest.TestCase):
    def test_ignore_never_drops_mixed_valid_prediction(self):
        self.assertEqual(a.ignore_state([0,0,100,10],[[0,0,20,10]],[[20,0,100,10]]),'mixed_keep')
        self.assertEqual(a.ignore_state([30,0,100,10],[[0,0,20,10]],[[20,0,100,10]]),'ignore')
    def test_review_exactly_covers_frozen_fp_ids(self):
        with self.assertRaises(ValueError):a.validate_reviews(['a','b'],[{'prediction_id':'a'},{'prediction_id':'a'}])
        with self.assertRaises(ValueError):a.validate_reviews(['a','b'],[{'prediction_id':'a'}])
        a.validate_reviews(['a'],[{'prediction_id':'a','primary':'uncertain'}])
    def test_order_only_preserves_strings_and_set(self):
        rows=[{'prediction_id':'a','text':'B','gt_order':2},{'prediction_id':'b','text':'A','gt_order':1}]
        d=a.order_diagnostic(rows,['A','B']);self.assertEqual(d['inversions'],1);self.assertEqual(d['possible_pairs'],1)
        self.assertEqual(d['O1']['cer_micro'],0);self.assertGreater(d['O0']['cer_micro'],0)
    def test_clear_matching_excludes_split_merge(self):
        p=[[0,0,50,10],[50,0,100,10]];g=[[0,0,100,10]]
        self.assertEqual(a.clear_pairs(p,g),[])
    def test_paired_crop_denominators_and_nfc(self):
        d=a.crop_comparison([{'page_id':'p','region_id':'x','text':'가'}],['가'],[''])
        self.assertEqual(d['C0']['errors'],0);self.assertEqual(d['C1']['deletions'],1)
        with self.assertRaises(ValueError):a.crop_comparison([{'page_id':'p','region_id':'x','text':'가'}],[],[''])
    def test_no_overwrite(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';a.write_new(p,{'a':1})
            with self.assertRaises(FileExistsError):a.write_new(p,{'a':2})
    def test_scope_keeps_legacy_immutable_and_tracks_changes(self):
        import copy
        pages=[{'id':'p','regions':[{'id':'g','bbox':[0,0,10,10],'text':'한글','order':1,'type_suggestion':'text','hold_reasons':[]}]}]
        original=copy.deepcopy(pages)
        new,changes=a.make_scope(pages,{'p':{0:'uncertain handwriting'}},[])
        self.assertEqual(pages,original);self.assertFalse(new[0]['regions'][0]['include'])
        self.assertEqual(new[0]['regions'][0]['source_gt_ids'],['g']);self.assertEqual(len(changes),1)
    def test_page_empty_missing_failure_preserves_denominator(self):
        pages=[{'id':'p','regions':[{'annotation_id':'g','bbox':[0,0,10,10],'text':'가나','order':1,'include':True}]}]
        result=a.score_scope(pages,{'pages':[]})
        self.assertEqual(result['page']['deletions'],2);self.assertEqual(result['detection']['fn'],1)
        result=a.score_scope(pages,{'pages':[{'id':'p','error':'failed','regions':[]}]})
        self.assertEqual(result['failures'],['p']);self.assertEqual(result['page']['reference_units'],2)
    def test_order_revision_changes_order_only(self):
        p=[{'id':'p','regions':[{'annotation_id':'left','bbox':[0,1,5,5],'text':'L','order':1},{'annotation_id':'right','bbox':[10,0,15,5],'text':'R','order':0}]}]
        q=a.revise_order(p,{'p':[['left','right']]})
        self.assertEqual([r['text'] for r in q[0]['regions']],['L','R'])
        self.assertLess(q[0]['regions'][0]['order'],q[0]['regions'][1]['order'])
        self.assertEqual(p[0]['regions'][0]['order'],1)
    def test_file_drift_is_rejected(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);weight=p/'adapter';weight.write_bytes(b'first')
            a.write_new(p/'input-lock.json',{'hashes':{str(weight):a.sha256(weight)},'config':a.read('configs/ocr_v2_vl_lora.json')})
            with patch.object(a,'ROOT',p):
                a.verify_lock();weight.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'input drift'):a.verify_lock()
    def test_classification_and_scope_artifacts_are_complete(self):
        fps=a.read(a.ROOT/'frozen-fp201.json');review=a.read(a.ROOT/'fp201-audit.json')
        self.assertEqual(len(fps),201);a.validate_reviews([x['prediction_id'] for x in fps],review)
        self.assertEqual(sum(__import__('collections').Counter(x['primary'] for x in review).values()),201)
        legacy=a.read(a.ROOT/'legacy-gt.json');self.assertEqual(sum(not g['hold_reasons'] for p in legacy for g in p['regions']),199)
        v1=a.read(a.ROOT/'audited-gt-v1.json')['pages'];v2=a.read(a.ROOT/'audited-gt-v2.json')['pages']
        for p,q in zip(v1,v2):
            for r,s in zip(p['regions'],q['regions']):
                self.assertEqual({k:v for k,v in r.items() if k!='order'},{k:v for k,v in s.items() if k!='order'})
    def test_one_to_one_is_distinct_from_split_diagnostic(self):
        # Two partial detections cover one GT; only one may be a .5 IoU TP.
        p=[[0,0,60,10],[40,0,100,10]];g=[[0,0,100,10]]
        self.assertEqual(len(a.match_boxes(p,g,.5)),1);self.assertEqual(a.clear_pairs(p,g),[])
    def test_new_scope_mixed_prediction_kept_and_empty_output_counted(self):
        pages=[{'id':'p','regions':[{'annotation_id':'valid','bbox':[0,0,20,10],'text':'가','order':0,'include':True},{'annotation_id':'held','bbox':[20,0,100,10],'text':'','order':1,'include':False}]}]
        pred={'pages':[{'id':'p','regions':[{'region_id':'x','bbox':[0,0,100,10],'text':'','order':0}]}]}
        r=a.score_scope(pages,pred)
        self.assertEqual(r['cases'][0]['mixed_keep'],['x']);self.assertEqual(r['page']['deletions'],1)
        self.assertEqual(r['cases'][0]['prediction_regions'],1)
