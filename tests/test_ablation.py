"""Known edit distances, ground-truth gating, and paired report semantics."""
import json
from pathlib import Path
import tempfile
import unittest

from src.evaluation.ocr_metrics import score, aggregate, apply_review
from src.evaluation.ablation import save_json, digest, write_report, load_references


class AblationTests(unittest.TestCase):
    def test_korean_normalization_and_exact_distances(self):
        self.assertEqual(score('한글 English', '한글  English')['cer'], 0)
        result=score('abc def','axc def')
        self.assertAlmostEqual(result['cer'],1/7)
        self.assertAlmostEqual(result['wer'],1/2)
        self.assertEqual(result['characters']['substitutions'],1)
        self.assertEqual(score('a','aaa')['cer'],2)
        self.assertEqual(score('a b','ab')['cer_no_space'],0)
        with self.assertRaises(ValueError):
            score(' \n ', 'hello')

    def test_micro_is_error_weighted_not_macro_average(self):
        metrics=aggregate([score('a','b'),score('abcdefghi','abcdefghi')])
        self.assertAlmostEqual(metrics['cer_micro'],.1)
        self.assertAlmostEqual(metrics['cer_macro'],.5)

    def test_qwen_id_mapping_preserves_order_and_unedited_lines(self):
        ocr={'lines':[{'id':'one','text':'원문'},{'id':'two','text':'keep'}]}
        baseline,hybrid,changes=apply_review(ocr,{'vector_edits':[
            {'id':'one','text':'교정'}, {'id':'unknown','text':'hallucination'}, {'id':'one','text':'최종'},
            {'id':'two','text':123}], 'uncertainties':['확인 필요']})
        self.assertEqual(baseline,'원문\nkeep')
        self.assertEqual(hybrid,'최종\nkeep')
        self.assertEqual(changes['changed_lines'],1)
        self.assertEqual(changes['unknown_edit_ids'],['unknown'])
        self.assertEqual(changes['duplicate_edit_ids'],['one'])
        self.assertEqual(changes['invalid_edits'],1)

    def test_report_has_no_accuracy_without_verified_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest={'dataset':'test','revision':'fixed','split':'train','seed':42,'population_rows':2,
                      'samples':[{'id':'s1'},{'id':'s2'}]}
            save_json(root/'manifest.json',manifest)
            report={'samples':[
                {'id':'s1','paddle':{'status':'ok','wall_seconds':2},'qwen':{'status':'ok','wall_seconds':3},
                 'hybrid_total_seconds':5.1,'baseline_text':'abc','hybrid_text':'axc',
                 'changes':{'baseline_lines':1,'changed_lines':1}},
                {'id':'s2','paddle':{'status':'ok','wall_seconds':2},'qwen':{'status':'failed','wall_seconds':3},
                 'hybrid_total_seconds':5,'baseline_text':'same'}]}
            save_json(root/'results.json',report)
            summary=write_report(root/'manifest.json',root/'results.json')
            self.assertFalse(summary['accuracy_available'])
            self.assertIsNone(summary['reference_paired_baseline'])
            self.assertEqual(summary['qwen_failures'],1)
            self.assertEqual(summary['paired_success'],1)
            refs={'manifest_sha256':digest(root/'manifest.json'),'samples':[
                  {'id':'s1','text':'abc','status':'verified'}, {'id':'s2','text':'same','status':'draft'}]}
            save_json(root/'refs.json',refs)
            summary=write_report(root/'manifest.json',root/'results.json',root/'refs.json')
            self.assertTrue(summary['accuracy_available'])
            self.assertEqual(summary['reference_paired_baseline']['cer_micro'],0)
            self.assertAlmostEqual(summary['reference_paired_hybrid']['cer_micro'],1/3)
            self.assertEqual(summary['reference_counts']['draft'],1)
            self.assertEqual(summary['cer_sample_outcomes'],{'improved':0,'worsened':1,'unchanged':0})
            details=json.loads((root/'sample_scores.json').read_text())
            self.assertEqual(details['samples'][0]['hybrid']['characters']['substitutions'],1)
            self.assertEqual(summary['reference_provenance']['sha256'],digest(root/'refs.json'))
            refs['manifest_sha256']='wrong'
            save_json(root/'refs.json',refs)
            with self.assertRaises(ValueError):
                load_references(root/'refs.json',digest(root/'manifest.json'),{'s1','s2'})

if __name__=='__main__':
    unittest.main()
