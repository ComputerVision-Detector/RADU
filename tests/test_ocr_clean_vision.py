import unittest
import json
from pathlib import Path
import tempfile
from src.training.ocr_clean_vision import visual_targets,activate_adapters,verify_inputs
from src.ocr_data.audit import sha256

class VisualAdaptationContract(unittest.TestCase):
    def test_parent_processing_config_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'dataset').mkdir();(root/'dataset/manifest.json').write_text('{}')
            parent=root/'parent.json';parent.write_text(json.dumps(dict(dataset=str(root/'dataset'),prompt='CHANGED')))
            r=dict(output=d,parent_config=str(parent),epochs=3,learning_rate=.00002,seed=42,rank=8,alpha=16,dropout=.05,batch_size=2,accumulation=4,test_status='exploratory')
            (root/'start-receipt.json').write_text(json.dumps(dict(config=r,effective={'prompt':'OLD'},input_hashes={},dataset_manifest_sha256=sha256(root/'dataset/manifest.json'))))
            with self.assertRaisesRegex(ValueError,'effective config'):verify_inputs(r)
    def test_only_last_visual_qv_are_targeted(self):
        names=[f'model.visual.vision_model.encoder.layers.{i}.self_attn.{p}' for i in range(6) for p in ['q_proj','v_proj','out_proj']]+['model.language_model.layers.5.self_attn.q_proj']
        picked=visual_targets(names,2)
        self.assertEqual(len(picked),4)
        self.assertTrue(all('.layers.4.' in x or '.layers.5.' in x for x in picked))
        self.assertTrue(all('visual' in x and not x.endswith('out_proj') for x in picked))

    def test_both_adapters_active_but_only_vision_updates(self):
        import torch
        from peft import get_peft_model,LoraConfig
        base=torch.nn.Sequential(torch.nn.Linear(3,3,bias=False),torch.nn.Linear(3,3,bias=False))
        model=get_peft_model(base,LoraConfig(r=2,lora_alpha=2,target_modules=['1']),adapter_name='language')
        with torch.no_grad():model.base_model.model[1].lora_B['language'].weight.fill_(.1)
        x=torch.ones(2,3);model.eval();parent=model(x).detach().clone()
        model.add_adapter('vision',LoraConfig(r=2,lora_alpha=2,target_modules=['0']))
        activate_adapters(model,'vision',True);model.eval()
        self.assertTrue(torch.equal(parent,model(x)))
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        params=[p for p in model.parameters() if p.requires_grad]
        self.assertTrue(params)
        self.assertTrue(all('.vision.' in n for n,p in model.named_parameters() if p.requires_grad))
        opt=torch.optim.AdamW(params,lr=.01);model(x).sum().backward();opt.step()
        self.assertTrue(any(not torch.equal(parent,model(x)) for _ in [0]))
        self.assertTrue(all(torch.equal(p,frozen[n]) for n,p in model.named_parameters() if n in frozen))
        self.assertEqual(set(model.active_adapters),{'language','vision'})
if __name__=='__main__':unittest.main()
