import unittest
import tempfile
from pathlib import Path
from src.training.ocr_lora_round2 import primary_validation,arm_config,verify_selected_adapter,add_halfstep_adapter


class RoundTwoContract(unittest.TestCase):
    def test_halfstep_is_exact_delta_average(self):
        try:
            import torch
            from peft import get_peft_model,LoraConfig
        except ImportError:self.skipTest('run this integration test in the dedicated LoRA environment')
        base=torch.nn.Sequential(torch.nn.Linear(3,3,bias=False))
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=2,target_modules=['0']))
        model.add_adapter('fitted',LoraConfig(r=2,lora_alpha=2,target_modules=['0']))
        layer=model.base_model.model[0]
        with torch.no_grad():
            layer.lora_A['default'].weight.fill_(.2);layer.lora_B['default'].weight.fill_(.3)
            layer.lora_A['fitted'].weight.fill_(.4);layer.lora_B['fitted'].weight.fill_(.5)
        expected=.5*layer.get_delta_weight('default')+.5*layer.get_delta_weight('fitted')
        add_halfstep_adapter(model)
        self.assertTrue(torch.allclose(layer.get_delta_weight('halfstep'),expected,atol=1e-6))
    def test_checkpoint_drift_cannot_be_bound_to_old_metric(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'selected adapter'):
                verify_selected_adapter(d,{'adapter_sha256':'old'})
    def test_quarantine_never_alters_transcription(self):
        rows=[dict(page_id='ok',text='원문'),dict(page_id='hold',text='검수필요')]
        kept,held=primary_validation(rows,{'hold':'human review'})
        self.assertEqual(kept,[rows[0]]);self.assertEqual(held,[rows[1]])
        self.assertEqual(rows[1]['text'],'검수필요')

    def test_training_arms_differ_only_in_resolution(self):
        base=dict(seed=1,epochs=3,learning_rate=.00005)
        r=dict(seed=42,epochs=3,learning_rate=.00002,parent_adapter='a',output='new',arms={'continue':dict(min_pixels=100,max_pixels=200),'highres':dict(min_pixels=200,max_pixels=400)})
        a=arm_config(base,r,'continue');b=arm_config(base,r,'highres')
        self.assertEqual(a['initial_adapter'],b['initial_adapter'])
        self.assertEqual(a['learning_rate'],.00002)
        self.assertEqual({k:v for k,v in a.items() if k not in ['min_pixels','max_pixels','output']},{k:v for k,v in b.items() if k not in ['min_pixels','max_pixels','output']})

if __name__=='__main__':unittest.main()
