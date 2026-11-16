import unittest
from src.evaluation.ocr_finetune_page import mask_policy, ordered_boxes


class PageAdapterContract(unittest.TestCase):
    def test_excluded_predictions_and_ambiguous_overlaps(self):
        good=[dict(bbox=[0,0,10,10])];held=[dict(bbox=[20,0,30,10])]
        self.assertEqual(mask_policy([0,0,10,10],good,held),'keep')
        self.assertEqual(mask_policy([20,0,30,10],good,held),'exclude')
        self.assertEqual(mask_policy([0,0,30,10],good,held),'keep')
        self.assertEqual(mask_policy([5,0,25,10],good,[dict(bbox=[10,0,30,10])]),'exclude')
        self.assertEqual(mask_policy([0,0,10,10],good,good),'ambiguous')

    def test_reading_order_is_fixed_y_then_x(self):
        boxes=[[8,20,9,25],[10,0,20,5],[0,0,5,5]]
        self.assertEqual(ordered_boxes(boxes),[boxes[2],boxes[1],boxes[0]])

if __name__=='__main__':unittest.main()
