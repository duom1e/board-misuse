import copy,unittest
from build_misuse_demo import crop_pair_html
class PairEvidenceTest(unittest.TestCase):
 def test_old_rows_have_no_pair_section(self):
  self.assertEqual(crop_pair_html([{'food_category':'meat'}]),'')
 def test_pair_escaped_and_original_unchanged(self):
  rows=[{'food_category':'<script>','category_crop_xyxy':[1,2,3,4],'tight_crop_comparison':{'tight_opinion':'meat','relation':'same_category','tight_crop_xyxy':[2,2,3,3]}}];old=copy.deepcopy(rows)
  out=crop_pair_html(rows)
  self.assertNotIn('<script>',out);self.assertIn('&lt;script&gt;',out)
  self.assertIn('不证明正确',out);self.assertEqual(rows,old)
 def test_insufficient_evidence_visible(self):
  self.assertIn('证据不足',crop_pair_html([{'tight_crop_comparison':{'relation':'insufficient_evidence'}}]))
if __name__=='__main__':unittest.main()
