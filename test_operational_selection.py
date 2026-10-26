import copy,unittest
from run_food_mixed_selection import operational,score_views

class OperationalSelectionTests(unittest.TestCase):
 def setUp(self):
  self.r={'counts':{'precision':.8,'fp':1},'rows':[
   {'class':'meat','tp':2,'fn':1,'fp':0,'predictions':[{'class':0},{'class':0}]},
   {'class':'vegetable','tp':2,'fn':0,'fp':1,'predictions':[{'class':1},{'class':1},{'class':0}]},
   {'class':'empty','tp':0,'fn':0,'fp':0,'predictions':[]}]}
 def test_wrong_class_false_positive_is_charged_to_predicted_class(self):
  s=operational(self.r)
  self.assertTrue(s['feasible']);self.assertEqual(s['classes']['meat']['fp'],1)
  self.assertEqual(s['classes']['vegetable']['fp'],0)
 def test_empty_false_positive_rejects_even_if_aggregate_precision_passes(self):
  self.r['rows'][2].update(fp=1,predictions=[{'class':1}])
  self.assertFalse(operational(self.r)['feasible'])

 def test_meat_recall_gate_cannot_be_hidden_by_good_vegetable_score(self):
  self.r['rows'][0].update(tp=1,fn=2,predictions=[{'class':0}])
  self.assertFalse(operational(self.r)['feasible'])
 def test_no_prediction_is_not_successful_empty_classification(self):
  for r in self.r['rows']:r.update(tp=0,fn=1 if r['class']!='empty' else 0,fp=0,predictions=[])
  self.r['counts'].update(precision=None,fp=0)
  self.assertFalse(operational(self.r)['feasible'])

class CameraGateTests(unittest.TestCase):
 def test_good_overhead_cannot_hide_missing_side_meat(self):
  rows=[{'image':'a','class':'meat','tp':2,'fn':0,'fp':0,'predictions':[{'class':0},{'class':0}]},
        {'image':'b','class':'vegetable','tp':2,'fn':0,'fp':0,'predictions':[{'class':1},{'class':1}]},
        {'image':'c','class':'meat','tp':0,'fn':1,'fp':0,'predictions':[]},
        {'image':'d','class':'vegetable','tp':1,'fn':0,'fp':0,'predictions':[{'class':1}]}]
  result={'counts':{'precision':1.,'fp':0},'rows':rows}
  self.assertTrue(operational(result)['feasible'])
  score=score_views(result,[{'image':x,'camera':'overhead' if x in 'ab' else 'side'} for x in 'abcd'])
  self.assertFalse(score['feasible']);self.assertEqual(score['rank'][0],0)
  self.assertEqual(score['views']['side']['classes']['meat']['recall'],0)

if __name__=='__main__':unittest.main()
