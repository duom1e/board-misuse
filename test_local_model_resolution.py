import unittest,tempfile,os
from pathlib import Path
from unittest.mock import patch
from region_vlm_runtime import resolve_local_model
class ResolutionTest(unittest.TestCase):
 def test_explicit_path_does_not_claim_revision(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'config.json').write_text('{}')
   with patch.dict(os.environ,{'BOARD_VLM_MODEL_DIR':d}):
    r=resolve_local_model();self.assertEqual(r['folder'],str(Path(d).resolve()));self.assertIsNone(r['revision'])
 def test_bad_override_does_not_silently_fallback(self):
  with tempfile.TemporaryDirectory() as d:
   with patch.dict(os.environ,{'BOARD_VLM_MODEL_DIR':d}):
    with self.assertRaises(ValueError):resolve_local_model()
if __name__=='__main__':unittest.main()
