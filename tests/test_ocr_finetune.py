import unittest
import tempfile
from pathlib import Path
from src.training.ocr_finetune import screen, compare, ctc_timesteps, verify_crops, require_same_config


class FineTuneContract(unittest.TestCase):
    def test_evaluation_rejects_changed_preprocessing(self):
        with self.assertRaisesRegex(ValueError,'width'):
            require_same_config({'width':768},{'width':832})
    def test_changed_crop_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'crop.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'crop hash'):
                verify_crops([dict(image='crop.png',crop_sha256='wrong')],d)
    def test_long_labels_need_repeat_aware_ctc_budget(self):
        self.assertEqual(ctc_timesteps('book'),5)
        self.assertEqual(ctc_timesteps('가가가'),5)
    def test_screen_never_substitutes_prediction(self):
        rows = [dict(page_id='p', region_id='a', text='가A', prediction='나'),
                dict(page_id='p', region_id='b', text='가Ω'),
                dict(page_id='p', region_id='c', text='가\nA')]
        good, held = screen(rows, list('가A '), 32)
        self.assertEqual([r['text'] for r in good], ['가A'])
        self.assertEqual(len(held), 2)
        self.assertNotIn('prediction', good[0])

    def test_comparison_includes_empty_failures_and_nfc(self):
        rows=[dict(page_id='p1',text='가',region_id='1'),dict(page_id='p2',text='AB',region_id='2')]
        result=compare(rows, ['가',''], ['가','AC'])
        self.assertAlmostEqual(result['before']['cer_micro'], 2/3)
        self.assertAlmostEqual(result['after']['cer_micro'], 1/3)
        self.assertAlmostEqual(result['relative_cer_reduction'], .5)

    def test_zero_baseline_relative_is_none(self):
        r=compare([dict(page_id='p',region_id='1',text='가')],['가'],['나'])
        self.assertIsNone(r['relative_cer_reduction'])

if __name__ == '__main__': unittest.main()
