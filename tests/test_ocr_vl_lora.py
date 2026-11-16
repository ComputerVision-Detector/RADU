import unittest
from src.training.ocr_vl_lora import target_modules,completion_labels,generation_limit


class LoRAContract(unittest.TestCase):
    def test_padded_completed_generation_is_not_truncated(self):
        self.assertFalse(generation_limit([5,2,0,0],4,2))
        self.assertTrue(generation_limit([5,6,7,8],4,2))
    def test_only_language_attention_is_adapted(self):
        names=['model.visual.a.q_proj','model.language_model.layers.0.self_attn.q_proj','model.language_model.layers.0.self_attn.o_proj','lm_head']
        self.assertEqual(target_modules(names),names[1:3])

    def test_prompt_and_padding_are_not_supervised_eos_is(self):
        self.assertEqual(completion_labels([1,2,3,4,9,0],[1,2,3],[1,1,1,1,1,0]),[-100,-100,-100,4,9,-100])

    def test_prefix_mismatch_cannot_train(self):
        with self.assertRaises(ValueError):completion_labels([1,2,3],[1,7],[1,1,1])

if __name__=='__main__':unittest.main()
