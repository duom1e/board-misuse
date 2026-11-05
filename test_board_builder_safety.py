import unittest,tempfile,json,hashlib,sys
from pathlib import Path
from unittest.mock import patch
import build_board_seg_dataset as b
class BuilderSafetyTest(unittest.TestCase):
 def test_existing_output_survives(self):
  with tempfile.TemporaryDirectory() as d:
   marker=Path(d)/'keep';marker.write_text('preserve')
   with patch.object(sys,'argv',['builder','--output-dir',d]),self.assertRaises(SystemExit):b.main()
   self.assertEqual(marker.read_text(),'preserve')
 def test_binding_and_conflicts(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);folder=root/'experiments/x/frames';folder.mkdir(parents=True)
   image=folder/'source_1.jpg';image.write_bytes(b'sample')
   data={'image':str(image),'image_sha256':hashlib.sha256(b'sample').hexdigest(),'board':[0,0,100,100]}
   label=folder/'a.board.json';label.write_text(json.dumps(data))
   with patch.object(b,'BASE',root),patch.object(b,'EXPERIMENTS',['x']):
    self.assertEqual(len(b.collect()),1)
    image.write_bytes(b'changed')
    with self.assertRaises(ValueError):b.collect()
    image.write_bytes(b'sample');data['board']=[0,0,200,200];(folder/'b.board.json').write_text(json.dumps(data))
    with self.assertRaises(ValueError):b.collect()
if __name__=='__main__':unittest.main()
