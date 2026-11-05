import unittest,tempfile,json,hashlib
from pathlib import Path
from review_combined import load_human_annotation
class BindingTest(unittest.TestCase):
 def test_image_binding_and_coordinate_contract(self):
  with tempfile.TemporaryDirectory() as folder:
   image=Path(folder)/'image';image.write_bytes(b'first image');annotation=Path(folder)/'annotation.json'
   data={'board':[10,20,300,400],'normalized':True,'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest()}
   annotation.write_text(json.dumps(data));self.assertEqual(load_human_annotation(annotation,image),data['board'])
   image.write_bytes(b'other image')
   with self.assertRaisesRegex(ValueError,'SHA'):load_human_annotation(annotation,image)
   data.pop('image_sha256');data['normalized']=False;annotation.write_text(json.dumps(data))
   with self.assertRaises(ValueError):load_human_annotation(annotation,image)
 def test_legacy_simple_box(self):
  with tempfile.TemporaryDirectory() as folder:
   image=Path(folder)/'image';image.write_bytes(b'img');p=Path(folder)/'a.json';p.write_text(json.dumps({'board':[0,0,100,100]}))
   self.assertEqual(load_human_annotation(p,image),[0,0,100,100])
if __name__=='__main__':unittest.main()
