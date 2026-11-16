import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from src.ocr_data.audit import audit_page, verify_image
from src.ocr_data.split import assign_splits, validate_splits
from src.evaluation.ocr_v2 import evaluate, match_boxes


def page(id='p', text='가나'):
    return dict(id=id, image=dict(source_width=20, source_height=20, source_sha256='x'),
                bbox_annotation=dict(status='verified', no_text=False, notes='', boxes=[
                    dict(id='b', bbox=[0, 0, 10, 10], text=text, prediction='WRONG', order=1, reviewed=True, uncertain=False)]))


class DataTests(unittest.TestCase):
    def test_prediction_is_never_ground_truth(self):
        row=audit_page(page(), dict(width=20,height=20,sha256='x'))
        self.assertEqual(row['regions'][0]['text'], '가나')
        self.assertNotIn('prediction', row['regions'][0])

    def test_uncertain_empty_unreviewed_and_no_text(self):
        for key,value in [('uncertain',True),('text',''),('reviewed',False)]:
            p=page();p['bbox_annotation']['boxes'][0][key]=value
            self.assertTrue(audit_page(p,dict(width=20,height=20,sha256='x'))['regions'][0]['hold_reasons'])
        p=page('hf-0104');p['bbox_annotation'].update(no_text=True,boxes=[])
        self.assertTrue(audit_page(p,dict(width=20,height=20,sha256='x'))['hold_reasons'])
        p=page();p['bbox_annotation'].update(no_text=True,boxes=[])
        self.assertFalse(audit_page(p,dict(width=20,height=20,sha256='x'))['hold_reasons'])

    def test_geometry_and_provenance(self):
        p=page();p['bbox_annotation']['boxes'][0]['bbox']=[12,0,2,4]
        self.assertIn('invalid_bbox',audit_page(p,dict(width=20,height=20,sha256='x'))['regions'][0]['hold_reasons'])
        self.assertTrue(audit_page(page(),dict(width=21,height=20,sha256='y'))['hold_reasons'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.png';Image.new('RGB',(20,20)).save(path)
            with self.assertRaises(ValueError):verify_image(path,dict(width=20,height=20,sha256='bad'))
            with self.assertRaises(ValueError):verify_image(path,dict(width=21,height=20,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def test_multiline_formula_preserved(self):
        p=page(text='\\frac{가}{나}\n행\t끝')
        r=audit_page(p,dict(width=20,height=20,sha256='x'))['regions'][0]
        self.assertEqual(r['text'],p['bbox_annotation']['boxes'][0]['text'])
        self.assertEqual(r['type_suggestion'],'formula')
        self.assertIn('multiline',r['flags'])

    def test_transitive_group_split(self):
        rows=[dict(id='a',groups=['writer:1']),dict(id='b',groups=['writer:1','note:1']),dict(id='c',groups=['note:1'])]
        split=assign_splits(rows,42)
        self.assertEqual(len(set(split['assignments'].values())),1)
        validate_splits(rows,split['assignments'])
        with self.assertRaises(ValueError):validate_splits(rows,dict(a='train',b='validation',c='test'))
        with self.assertRaises(ValueError):validate_splits(rows+rows[:1],split['assignments'])


class MetricTests(unittest.TestCase):
    def gt(self,text='가나',empty=False):
        return [dict(id='p',split='validation',eligible=True,no_text=empty,regions=[] if empty else [dict(id='b',bbox=[0,0,10,10],text=text,order=1,type_suggestion='text',hold_reasons=[])])]

    def test_one_to_one_miss_and_false_positive(self):
        m=match_boxes([[0,0,10,10],[0,0,10,10]],[[0,0,10,10],[30,30,40,40]],.5)
        self.assertEqual(len(m),1)
        r=evaluate(self.gt(),dict(mode='page',pages=[dict(id='p',regions=[dict(bbox=[0,0,10,10],text='가나'),dict(bbox=[20,20,30,30],text='x')])]))
        self.assertEqual(r['detection']['tp'],1);self.assertEqual(r['detection']['fp'],1)
        self.assertGreater(r['end_to_end']['cer_micro'],0)

    def test_nfc_and_single_edits(self):
        for pred,counts in [('가나',(0,0,0)),('가다',(1,0,0)),('가',(0,1,0)),('가나다',(0,0,1))]:
            r=evaluate(self.gt(),dict(mode='recognition_only',crops=[dict(page_id='p',region_id='b',text=pred)]))
            c=r['recognition']['all']
            self.assertEqual(tuple(c[k] for k in ['substitutions','deletions','insertions']),counts)

    def test_empty_failure_and_missing(self):
        r=evaluate(self.gt(empty=True),dict(mode='page',pages=[dict(id='p',regions=[dict(bbox=[0,0,2,2],text='x')])]))
        self.assertEqual(r['no_text_false_positive_pages'],1)
        for pages in [[],[dict(id='p',error='failure')],[dict(id='p',regions=[])]]:
            r=evaluate(self.gt(),dict(mode='page',pages=pages))
            self.assertEqual(r['detection']['fn'],1)
            self.assertEqual(r['end_to_end']['cer_micro'],1)

    def test_order_and_exclusion(self):
        gt=self.gt();gt[0]['regions'] += [dict(id='c',bbox=[10,0,20,10],text='다라',order=2,type_suggestion='text',hold_reasons=[]),dict(id='held',bbox=[30,0,40,10],text='xxx',order=3,hold_reasons=['uncertain'])]
        pred=dict(mode='page',pages=[dict(id='p',regions=[dict(bbox=[10,0,20,10],text='다라'),dict(bbox=[0,0,10,10],text='가나'),dict(bbox=[30,0,40,10],text='WRONG')])])
        r=evaluate(gt,pred)
        self.assertEqual(r['detection']['fp'],0)
        self.assertGreater(r['end_to_end']['cer_micro'],0)
        self.assertEqual(r['ignored_predictions'],1)

    def test_reject_wrong_recognition_mode(self):
        with self.assertRaises(ValueError):evaluate(self.gt(),dict(mode='crop_detection',crops=[]))


class BuildTests(unittest.TestCase):
    def test_full_conversion_and_snapshot_tamper(self):
        import json
        from src.ocr_data.build import build
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);snap=root/'snap';snap.mkdir();original=root/'original';(original/'images').mkdir(parents=True)
            image=original/'images/p.png';Image.new('RGB',(20,20),'white').save(image)
            digest=hashlib.sha256(image.read_bytes()).hexdigest()
            p=page();p['image']['source_sha256']=digest
            payload={'live-v2.json':dict(dataset_version='v2',items={'p':p}),
                     'dataset.catalog.json':dict(samples=[dict(id='p',width=20,height=20,sha256=digest)]),
                     'assets.manifest.json':dict(images=[dict(id='p',source_sha256=digest,sha256='asset')])}
            p['image']['asset_sha256']='asset'
            for name,value in payload.items():(snap/name).write_text(json.dumps(value))
            (snap/'manifest.json').write_text(json.dumps(dict(files_sha256={n:hashlib.sha256((snap/n).read_bytes()).hexdigest() for n in payload})))
            stats=build(snap,original,root/'build')
            self.assertEqual(stats['generated_crops'],1)
            row=json.loads((root/'build/crops.jsonl').read_text())
            self.assertEqual(row['text'],'가나');self.assertEqual(row['split'],'train')
            with Image.open(root/'build'/row['image']) as crop:self.assertEqual(crop.size,(10,10))
            (snap/'live-v2.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'snapshot hash mismatch'):build(snap,original,root/'bad')

    def test_exact_duplicate_and_invalid_ids_orders(self):
        p=page();p['bbox_annotation']['boxes'].append(copy.deepcopy(p['bbox_annotation']['boxes'][0]))
        row=audit_page(p,dict(width=20,height=20,sha256='x'))
        self.assertFalse(row['eligible']);self.assertIn('duplicate_region_id',row['hold_reasons'])
        self.assertIn('duplicate_region',row['regions'][0]['hold_reasons'])

    def test_cardinality_matching_beats_greedy(self):
        # p0 connects to both GT boxes, p1 only to g0; greedy IoU would lose a TP.
        preds=[[2,0,12,10],[0,0,7,10]];refs=[[0,0,10,10],[6,0,16,10]]
        self.assertEqual(len(match_boxes(preds,refs,.4)),2)

    def test_export_reports_unsupported_without_substitution(self):
        import json
        from src.ocr_data.export_recognition import export
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();(data/'manifest.json').write_text('{}')
            row=dict(page_id='p',region_id='b',image='crop.png',text='가😀')
            for s in ['train','validation','test']:(data/f'recognition_candidates.{s}.jsonl').write_text(json.dumps(row,ensure_ascii=False)+'\n')
            config=root/'inference.yml';config.write_text('PostProcess:\n  character_dict: [가]\n')
            result=export(data,config,10,root/'export')
            self.assertEqual(result['rejected'],3)
            self.assertEqual((root/'export/train.txt').read_text(),'')
            self.assertEqual(json.loads((root/'export/rejected.jsonl').read_text().splitlines()[0])['text'],'가😀')

class ProvenanceTests(unittest.TestCase):
    def fixture(self,root):
        import json
        from src.ocr_data.build import build
        snap=root/'snap';snap.mkdir();orig=root/'original';(orig/'images').mkdir(parents=True)
        image=orig/'images/p.png';Image.new('RGB',(20,20),'white').save(image);digest=hashlib.sha256(image.read_bytes()).hexdigest()
        p=page();p['image'].update(source_sha256=digest,asset_sha256='asset')
        payload={'live-v2.json':dict(dataset_version='v2',items={'p':p}), 'dataset.catalog.json':dict(samples=[dict(id='p',width=20,height=20,sha256=digest)]),'assets.manifest.json':dict(images=[dict(id='p',source_sha256=digest,sha256='asset')])}
        for name,value in payload.items():(snap/name).write_text(json.dumps(value))
        (snap/'manifest.json').write_text(json.dumps(dict(files_sha256={n:hashlib.sha256((snap/n).read_bytes()).hexdigest() for n in payload})))
        build(snap,orig,root/'built');return root/'built',snap

    def test_snapshot_binding(self):
        import json
        from src.ocr_data.validate import validate
        with tempfile.TemporaryDirectory() as d:
            root,snap=self.fixture(Path(d));self.assertEqual(validate(root)['crops_verified'],1)
            p=snap/'manifest.json';m=json.loads(p.read_text());m['replaced']=True;p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'snapshot binding'):validate(root)

    def test_crop_pixels_against_source(self):
        import json
        from src.ocr_data.validate import validate
        with tempfile.TemporaryDirectory() as d:
            root,snap=self.fixture(Path(d));p=root/'crops.jsonl';r=json.loads(p.read_text());Image.new('RGB',(10,10),'black').save(root/r['image'])
            r['crop_sha256']=hashlib.sha256((root/r['image']).read_bytes()).hexdigest();p.write_text(json.dumps(r)+'\n')
            m=json.loads((root/'manifest.json').read_text());m['artifacts_sha256']['crops.jsonl']=hashlib.sha256(p.read_bytes()).hexdigest();(root/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'crop pixels'):validate(root)

    def test_page_ground_truth_drift(self):
        import json
        from src.ocr_data.validate import validate
        with tempfile.TemporaryDirectory() as d:
            root,snap=self.fixture(Path(d));p=root/'pages.jsonl';r=json.loads(p.read_text());r['regions'][0]['text']='WRONG';p.write_text(json.dumps(r)+'\n')
            m=json.loads((root/'manifest.json').read_text());m['artifacts_sha256']['pages.jsonl']=hashlib.sha256(p.read_bytes()).hexdigest();(root/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'page ground truth'):validate(root)

    def test_prediction_dataset_binding(self):
        from src.evaluation.ocr_v2 import evaluate_dataset
        with tempfile.TemporaryDirectory() as d:
            root,snap=self.fixture(Path(d))
            with self.assertRaisesRegex(ValueError,'prediction dataset fingerprint'):evaluate_dataset(root,dict(mode='page',dataset_manifest_sha256='wrong',pages=[]),'train')

    def test_page_inference_integrity(self):
        import json
        from src.ocr_data.infer_smoke import run
        with tempfile.TemporaryDirectory() as d:
            root,snap=self.fixture(Path(d));p=json.loads((root/'pages.jsonl').read_text());Image.new('RGB',(20,20),'black').save(p['image'])
            r=run(root,'page',Path(d)/'smoke.json','cpu',1)
            self.assertIn('original_hash_mismatch',r['pages'][0]['error'])


class VLExportTests(unittest.TestCase):
    def test_vl_labels_preserved_and_token_overflow_reported(self):
        import json
        from tokenizers import Tokenizer,models,pre_tokenizers
        from src.ocr_data.export_recognition import export_vl
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';data.mkdir();(data/'manifest.json').write_text('{}')
            tok=Tokenizer(models.WordLevel({'[UNK]':0,'가':1,'나':2},unk_token='[UNK]'));tok.pre_tokenizer=pre_tokenizers.Whitespace();tok.save(str(root/'tokenizer.json'))
            rows=[dict(page_id='p',region_id='good',image='crop.png',text='가 나'),dict(page_id='p',region_id='long',image='crop.png',text='가 나 가')]
            for s in ['train','validation','test']:(data/f'recognition_candidates.{s}.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n')
            r=export_vl(data,root/'tokenizer.json',2,root/'vl')
            self.assertEqual(r['counts']['train'],1);self.assertEqual(r['rejected'],3)
            saved=json.loads((root/'vl/train.jsonl').read_text());self.assertEqual(saved['text'],'가 나');self.assertEqual(saved['query'],'OCR:')

if __name__=='__main__':unittest.main()
