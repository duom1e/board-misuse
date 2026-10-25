import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from annotation_io import prepare_image, normalized_box, export_annotation

class AnnotationTests(unittest.TestCase):
    def test_roundtrip_and_negative_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source.jpg'
            im=Image.new('RGB',(120,80),'white')
            exif=im.getexif(); exif[274]=6
            im.save(source,exif=exif)
            folder, normalized, metadata=prepare_image(source,root/'cases')
            self.assertEqual(normalized.size,(80,120))
            boxes=[normalized_box((60,100),(10,20),80,120)]
            out=export_annotation(folder/'image.png',metadata,boxes,'video_1','all_boards_boxed',root/'pool')
            values=list(map(float,(out/'image.txt').read_text().split()))
            self.assertEqual(values[0],0)
            self.assertAlmostEqual(values[3],.625)
            self.assertTrue((out/'COMPLETE').is_file())
            self.assertEqual(json.loads((out/'metadata.json').read_text(encoding='utf-8'))['source_group'],'video_1')
            for b,g,d in [(boxes,'','all_boards_boxed'),(boxes,'g','confirmed_no_board'),([], 'g','all_boards_boxed'),([], 'g',None)]:
                with self.assertRaises(ValueError): export_annotation(folder/'image.png',metadata,b,g,d,root/'pool')
            neg=export_annotation(folder/'image.png',metadata,[],'video_2','confirmed_no_board',root/'pool')
            self.assertEqual((neg/'image.txt').read_text(),'')
            self.assertNotEqual(out,neg)
    def test_clamp_and_small_boxes(self):
        self.assertEqual(normalized_box((-10,-10),(200,100),100,100),[0,0,1,1])
        with self.assertRaises(ValueError): normalized_box((10,10),(11,11),100,100)
    def test_hidden_ui_load(self):
        import tkinter as tk
        from review_app import ReviewApp
        root=tk.Tk(); root.withdraw()
        try:
            app=ReviewApp(root)
            root.update_idletasks()
            self.assertEqual(app.rule.get(),'')
            self.assertFalse(app.busy)
            app.clear(); app.undo()
        finally: root.destroy()

if __name__=='__main__': unittest.main(verbosity=2)
