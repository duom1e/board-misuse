import unittest
from PIL import Image
from food_context_review import compare_categories,review_food_context
class ContextReviewTest(unittest.TestCase):
    def test_vegetable_context_does_not_erase_local_meat(self):
        r=compare_categories('meat','vegetable');self.assertEqual(r['relation'],'contradiction');self.assertEqual(r['local_category'],'meat');self.assertFalse(r['category_replaced'])
    def test_mixed_context_does_not_require_local_mixed(self):
        self.assertEqual(compare_categories('meat','mixed')['relation'],'not_contradicted')
    def test_agreement_does_not_mean_compliance(self):
        r=compare_categories('vegetable','vegetable');self.assertEqual(r['status'],'needs_review');self.assertFalse(r['compliance_determined'])
    def test_unknown_is_not_evidence_for_a_relabel(self):self.assertEqual(compare_categories('unknown','vegetable')['relation'],'insufficient_evidence')
    def test_other_is_not_forced_into_binary_label(self):self.assertEqual(compare_categories('other','meat')['relation'],'not_comparable')
    def test_one_full_image_call_for_multiple_crops(self):
        rows=[{'food_category':'meat','category_inference_performed':True},{'food_category':'vegetable','category_inference_performed':True}];calls=[]
        result,review=review_food_context(Image.new('RGB',(50,50)),rows,lambda image:calls.append(1) or 'vegetable')
        self.assertEqual(len(calls),1);self.assertEqual(review['contradiction_count'],1);self.assertNotIn('category_context_comparison',rows[0]);self.assertEqual(result[0]['food_category'],'meat')
    def test_context_failure_preserves_local_and_geometry(self):
        def fail(image):raise RuntimeError('test')
        result,review=review_food_context(Image.new('RGB',(50,50)),[{'food_category':'meat','category_inference_performed':True,'box':[0,0,40,40]}],fail)
        self.assertEqual(result[0]['food_category'],'meat');self.assertEqual(result[0]['box'],[0,0,40,40]);self.assertEqual(result[0]['category_context_comparison']['relation'],'insufficient_evidence')
    def test_no_useful_categories_does_not_call_context_model(self):
        _,r=review_food_context(Image.new('RGB',(50,50)),[{'food_category':'unknown','category_inference_performed':True}],lambda image:self.fail('Do not call'))
        self.assertFalse(r['context_inference_performed'])
if __name__=='__main__':unittest.main()
