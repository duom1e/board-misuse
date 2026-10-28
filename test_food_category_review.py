import unittest,copy
from PIL import Image
from food_category_review import review_board_foods
class CategoryReviewTest(unittest.TestCase):
    def setUp(self):self.image=Image.new('RGB',(100,100))
    def region(self,location='board_candidate',box=None):return {'box':box or [10,10,60,60],'location':location,'status':'needs_review','compliance_determined':False}
    def test_only_board_candidates_invoke_model(self):
        calls=[];rows=[self.region(),self.region('container_candidate'),self.region('uncertain')]
        result=review_board_foods(self.image,rows,lambda crop:calls.append(crop.size) or 'vegetable')
        self.assertEqual(len(calls),1);self.assertEqual(result[0]['food_category'],'vegetable');self.assertFalse(result[1]['category_inference_performed'])
    def test_unknown_answer_stays_unknown(self):
        result=review_board_foods(self.image,[self.region()],lambda crop:'unknown')[0];self.assertEqual(result['food_category'],'unknown');self.assertEqual(result['status'],'needs_review')
    def test_invalid_answer_is_not_fuzzy_matched(self):
        result=review_board_foods(self.image,[self.region()],lambda crop:'probably vegetable')[0];self.assertEqual(result['food_category'],'unknown');self.assertEqual(result['category_review_status'],'invalid_category_response')
    def test_small_crop_skips_model(self):
        result=review_board_foods(self.image,[self.region(box=[0,0,10,50])],lambda crop:self.fail('Must not call model'))[0];self.assertEqual(result['category_review_status'],'crop_too_small')
    def test_model_failure_keeps_original_geometry(self):
        rows=[self.region()];original=copy.deepcopy(rows)
        def fail(crop):raise RuntimeError('test failure')
        result=review_board_foods(self.image,rows,fail)[0];self.assertEqual(rows,original);self.assertEqual(result['box'],original[0]['box']);self.assertFalse(result['compliance_determined']);self.assertEqual(result['food_category'],'unknown')
    def test_margin_clamps_at_image_edge(self):
        result=review_board_foods(self.image,[self.region(box=[0,0,100,100])],lambda crop:'meat')[0];self.assertEqual(result['category_crop_xyxy'],[0,0,100,100])
if __name__=='__main__':unittest.main()
