import unittest
from src.ocr_data.train_audit import clean_training

class AuditContract(unittest.TestCase):
    def test_holds_preserve_gt_and_never_copy_prediction(self):
        rows=[dict(page_id='ok',region_id='a',split='train',text='원문',prediction='후보'),dict(page_id='hold',region_id='b',split='train',text='확인필요')]
        kept,held=clean_training(rows,{'hold':'neighbor line'})
        self.assertEqual(kept[0]['text'],'원문');self.assertNotIn('prediction',kept[0]);self.assertEqual(held[0]['text'],'확인필요')
    def test_cannot_use_validation_or_test_for_train_cleaning(self):
        for split in ['validation','test']:
            with self.assertRaises(ValueError):clean_training([dict(page_id='p',split=split,text='정답')],{})
if __name__=='__main__':unittest.main()
