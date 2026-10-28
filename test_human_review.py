import unittest
from validate_human_review import validate
class ReviewTest(unittest.TestCase):
 def test_binding_and_indices(self):
  data=dict(source_events_sha256='abc',records=[dict(index=0,decision='pending',note='')]);events=dict(records=[{}])
  self.assertEqual(validate(data,events,'abc')['pending'],1)
  with self.assertRaises(ValueError):validate(data,events,'changed')
  data['records'][0]['index']=True
  with self.assertRaises(ValueError):validate(data,events,'abc')
 def test_reject_missing_duplicate_and_fake_verdict(self):
  events=dict(records=[{},{}]);data=dict(source_events_sha256='x',records=[dict(index=0,decision='pending',note='')]*2)
  with self.assertRaises(ValueError):validate(data,events,'x')
  data['records']=[dict(index=0,decision='confirmed_misuse',note=''),dict(index=1,decision='pending',note='')]
  with self.assertRaises(ValueError):validate(data,events,'x')
if __name__=='__main__':unittest.main()
