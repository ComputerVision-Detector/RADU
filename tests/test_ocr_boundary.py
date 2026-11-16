import copy
import unittest
from PIL import Image,ImageDraw
from src.evaluation import ocr_boundary as b

class BoundaryTests(unittest.TestCase):
    def test_crop_retains_punctuation_and_input_pixels(self):
        im=Image.new('RGB',(300,120),'white');d=ImageDraw.Draw(im);d.rectangle([50,40,100,80],fill='black');d.rectangle([115,79,117,81],fill='black');old=im.tobytes()
        box=b.ink_bounds(im,b.POLICY)
        self.assertLessEqual(box[0],50);self.assertGreaterEqual(box[2],118);self.assertEqual(im.tobytes(),old)
        self.assertTrue(0<=box[0]<box[2]<=im.width and 0<=box[1]<box[3]<=im.height)
    def test_long_ruling_line_removed_only_from_bbox_estimation(self):
        im=Image.new('RGB',(500,150),'white');d=ImageDraw.Draw(im);d.line([0,130,499,130],fill='black',width=1);d.rectangle([200,50,250,100],fill='black')
        box=b.ink_bounds(im,b.POLICY);self.assertLess(box[3],130);self.assertEqual(im.getpixel((0,130)),(0,0,0))
    def test_empty_or_faint_falls_back_without_black_magic(self):
        self.assertEqual(b.ink_bounds(Image.new('RGB',(100,40),'white'),b.POLICY),[0,0,100,40])
        self.assertEqual(b.ink_bounds(Image.new('RGB',(100,40),(180,180,180)),b.POLICY),[0,0,100,40])
    def test_geometry_revision_only_changes_bbox_and_metadata(self):
        gt={'pages':[{'id':'p','regions':[{'annotation_id':'a','bbox':[0,0,50,50],'text':'가','order':1,'include':True}]}]}
        old=copy.deepcopy(gt)
        with self.assertRaises(ValueError):b.apply_geometry(gt,[{'page_id':'p','annotation_id':'a','bbox':[1,1,49,49]}])
        new,trace=b.apply_geometry(gt,[{'page_id':'p','annotation_id':'a','bbox':[1,1,49,49],'evidence':'original pixels reviewed','decision':'accept'}])
        self.assertEqual(gt,old);self.assertEqual(new['pages'][0]['regions'][0]['text'],'가');self.assertEqual(len(trace),1)
    def test_paired_comparison_requires_matching_ids(self):
        rows=[{'page_id':'p','region_id':'g','text':'가'}]
        with self.assertRaises(ValueError):b.paired_metrics(rows,[{'page_id':'p','region_id':'bad','text':'가'}],[{'page_id':'p','region_id':'g','text':'가'}])
        r=b.paired_metrics(rows,[{'page_id':'p','region_id':'g','text':'가'}],[{'page_id':'p','region_id':'g','text':''}])
        self.assertEqual(r['before']['errors'],0);self.assertEqual(r['after']['deletions'],1)
    def test_validation_selection_is_strict_and_ties_keep_baseline(self):
        self.assertEqual(b.select_method({'cer_micro':0.1},{'cer_micro':0.1}),'baseline')
        self.assertEqual(b.select_method({'cer_micro':0.1},{'cer_micro':0.11}),'baseline')
        self.assertEqual(b.select_method({'cer_micro':0.1},{'cer_micro':0.09}),'image_trim')
    def test_geometry_outside_source_rejected(self):
        gt={'pages':[{'id':'p','regions':[{'annotation_id':'g','bbox':[0,0,10,10],'text':'A','order':1,'include':True}]}]}
        with self.assertRaisesRegex(ValueError,'outside'):b.apply_geometry(gt,[{'page_id':'p','annotation_id':'g','bbox':[-1,0,10,10],'decision':'accept','evidence':'invalid'}])
    def test_validation_method_rejects_row_drift(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);row=p/'validation.before.rows.json';row.write_text('[]')
            b.write_new(p/'validation-method-lock.json',{'policy':b.POLICY,'policy_sha256':b.sha256('configs/ocr_boundary.json'),'validation_rows_sha256':{'before':b.sha256(row)}})
            with patch.object(b,'ROOT',p):
                b.verify_method();row.write_text('[{}]')
                with self.assertRaisesRegex(ValueError,'validation row drift'):b.verify_method()
