import copy,json,tempfile,unittest
from pathlib import Path
import torch
from src.training.ocr_model_only import objective,selected_tokens,select_candidate,language,score,verify,write_new,sha256

class ModelOnlyTests(unittest.TestCase):
    def test_weighted_CE_and_teacher_detached(self):
        logits=torch.tensor([[2.,0.],[0.,0.]],requires_grad=True);labels=torch.tensor([0,1]);weights=torch.tensor([1.,1.5]);teacher=logits.detach().clone().requires_grad_(True)
        loss,ce,kl=objective(logits,labels,weights,teacher,.1)
        expected=(torch.nn.functional.cross_entropy(logits,labels,reduction='none')*weights).sum()/weights.sum()
        self.assertAlmostEqual(float(ce),float(expected),places=6);self.assertAlmostEqual(float(kl),0.,places=6)
        loss.backward();self.assertIsNotNone(logits.grad);self.assertIsNone(teacher.grad)
    def test_prompt_padding_mask_and_causal_shift(self):
        labels=torch.tensor([[-100,-100,3,2,-100]]);logits=torch.arange(20.).reshape(1,5,4)
        selected,target=selected_tokens(logits,labels)
        self.assertEqual(target.tolist(),[3,2]);self.assertEqual(selected.tolist(),logits[0,1:3].tolist())
    def test_teacher_shape_mismatch_fails(self):
        with self.assertRaises(ValueError):objective(torch.zeros(2,3),torch.tensor([0,1]),torch.ones(2),torch.zeros(1,3),.1)
    def row(self,name,cer=.1,compact=.1,latin=.1,korean=.1):
        return dict(name=name,failures=[],metrics=dict(all=dict(cer_micro=cer,cer_no_space_micro=compact),strata=dict(latin_including_mixed=dict(cer_micro=latin),korean=dict(cer_micro=korean))))
    def test_parent_retained_for_tie_failure_or_regression(self):
        p=self.row('parent');self.assertEqual(select_candidate(p,[self.row('tie')])['name'],'parent')
        self.assertEqual(select_candidate(p,[self.row('space-only',.09,.11)])['name'],'parent')
        self.assertEqual(select_candidate(p,[self.row('latin-regression',.09,.09,.11)])['name'],'parent')
        self.assertEqual(select_candidate(p,[self.row('korean-regression',.09,.09,.09,.106)])['name'],'parent')
        bad=self.row('failed',.01,.01,.01,.01);bad['failures']=['missing'];self.assertEqual(select_candidate(p,[bad])['name'],'parent')
    def test_selection_has_no_test_dependency_and_uses_fixed_metric(self):
        p=self.row('parent');a=self.row('a',.09,.09,.09,.10);b=self.row('b',.08,.09,.09,.10)
        self.assertEqual(select_candidate(p,[a,b])['name'],'b');self.assertNotIn('test',b)
    def test_language_strata_and_empty_output_denominator(self):
        self.assertEqual([language(x) for x in ['한글','English','한글 English','123']],['korean','latin','mixed','other'])
        d=score([dict(text='가A')],['']);self.assertEqual(d['all']['deletions'],2);self.assertEqual(d['all']['reference_units'],2)
        with self.assertRaises(ValueError):score([dict(text='가')],[])
    def test_rows_config_and_original_hash_drift_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for n in ['plan.json','train-rows.json','validation-rows.json','weight']:write_new(p/n,{'a':1})
            lock=dict(hashes={str(p/'weight'):sha256(p/'weight')},plan_sha256=sha256(p/'plan.json'),train_rows_sha256=sha256(p/'train-rows.json'),validation_rows_sha256=sha256(p/'validation-rows.json'))
            write_new(p/'input-lock.json',lock);verify(p)
            (p/'train-rows.json').write_text('[]')
            with self.assertRaises(ValueError):verify(p)
            with self.assertRaises(FileExistsError):write_new(p/'plan.json',{})
