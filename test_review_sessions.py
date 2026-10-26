import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image, ImageTk
from annotation_io import prepare_image
from review_sessions import save_draft, load_draft
from review_app import ReviewApp

class ReviewSessionTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'source.png';Image.new('RGB',(100,60),'white').save(source)
        return prepare_image(source,root/'cases')
    def test_roundtrip_and_invalid_write_preserves_old_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,_,_=self.fixture(Path(tmp))
            path=save_draft(folder,[[.1,.1,.9,.9]],['肉类专用'],source_group='video1')
            before=path.read_bytes()
            _,draft,_=load_draft(path)
            self.assertEqual(draft['source_group'],'video1')
            for boxes,rules in [([[0,0,float('nan'),1]],[None]),([[0,0,1,1]],[]),([[0,0,1,1]],['未知'])]:
                with self.assertRaises(ValueError): save_draft(folder,boxes,rules)
                self.assertEqual(path.read_bytes(),before)
            self.assertFalse(list(folder.glob('*.tmp')))
    def test_image_and_source_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,_,_=self.fixture(Path(tmp))
            path=save_draft(folder,[],[])
            original=(folder/'source.json').read_bytes()
            (folder/'source.json').write_text('{}')
            with self.assertRaises(ValueError): load_draft(path)
            (folder/'source.json').write_bytes(original)
            Image.new('RGB',(100,60),'red').save(folder/'image.png')
            with self.assertRaises(ValueError): load_draft(path)
    def test_gui_restore_resets_review_and_recovers_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder,_,_=self.fixture(Path(tmp))
            path=save_draft(folder,[[.1,.1,.9,.9]],['蔬菜专用'],rule='肉类专用',source_group='scene_A')
            root=tk.Tk();root.withdraw()
            try:
                app=ReviewApp(root)
                app.decision.set('已检查整图，所有案板均已框出')
                with patch('review_app.filedialog.askopenfilename',return_value=str(path)):
                    app.restore_progress()
                self.assertEqual(app.decision.get(),'')
                self.assertEqual(app.group.get(),'scene_A')
                self.assertEqual(app.inspection_config(True)['boards'][0]['allowed_foods'],['vegetable'])
                self.assertEqual(app.board_list.item('0')['values'][1],'蔬菜专用')
                app.group.set('scene_B')
                app.save_progress(silent=True)
                self.assertEqual(load_draft(path)[1]['source_group'],'scene_B')
            finally: root.destroy()

if __name__=='__main__':unittest.main(verbosity=2)
