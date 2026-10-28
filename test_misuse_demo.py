import unittest
from build_misuse_demo import decide
class DemoTest(unittest.TestCase):
 def test_no_policy_never_invents_conflict(self):
  state,reason,evidence=decide(self.row(),[])
  self.assertEqual(state,'needs_review');self.assertEqual(len(evidence),1);self.assertIsNone(evidence[0]['conflict'])
 def row(self,category='meat',location='board_candidate',valid=True):return dict(validation=dict(valid=valid),regions=[dict(food_category=category,location=location,category_inference_performed=True,box=[0,0,10,10])])
 def test_conflict_requires_valid_board_evidence(self):
  self.assertEqual(decide(self.row(),['vegetable'])[0],'suspected_misuse')
  self.assertEqual(decide(self.row(valid=False),['vegetable'])[0],'needs_review')
  self.assertEqual(decide(self.row(location='container_candidate'),['vegetable'])[0],'needs_review')
 def test_unknown_never_compliance(self):
  self.assertEqual(decide(self.row('unknown'),['vegetable'])[0],'needs_review')
 def test_mixed_and_allowed(self):
  self.assertEqual(decide(self.row('mixed'),['vegetable'])[0],'suspected_misuse')
  self.assertEqual(decide(self.row('mixed'),['vegetable','meat'])[0],'no_supported_conflict')
if __name__=='__main__':unittest.main()
