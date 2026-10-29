import unittest
from PIL import Image
from board_identity_review import review_boards
class IdentityTest(unittest.TestCase):
 def test_no_veto_and_input_unchanged(self):
  boxes=[[0,0,50,50]];r=review_boards(Image.new('RGB',(100,100)),boxes,lambda _: 'other')
  self.assertEqual(boxes,[[0,0,50,50]]);self.assertFalse(r[0]['detection_removed']);self.assertEqual(r[0]['opinion'],'other');self.assertEqual(r[0]['crop_box'],[0,0,58,58])
 def test_invalid_and_failure_preserve_candidate(self):
  im=Image.new('RGB',(100,100));boxes=[[10,10,40,40]]
  self.assertEqual(review_boards(im,boxes,lambda _:'definitely a board')[0]['opinion'],'unknown')
  def fail(_):raise RuntimeError('test')
  r=review_boards(im,boxes,fail);self.assertEqual(len(r),1);self.assertEqual(r[0]['opinion'],'unknown');self.assertFalse(r[0]['detection_removed'])
 def test_empty_does_not_infer(self):
  self.assertEqual(review_boards(Image.new('RGB',(10,10)),[],lambda _:self.fail()),[])
if __name__=='__main__':unittest.main()
