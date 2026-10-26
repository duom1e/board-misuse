"""Mixed-purpose board review is a software check, not detector accuracy evidence."""
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from types import SimpleNamespace
from PIL import Image, ImageTk
from review_app import ReviewApp
from misuse_rules import BoardRule, Detection, evaluate

class PerBoardReviewTests(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk();self.root.withdraw()
        self.app=ReviewApp(self.root)
        self.temp=tempfile.TemporaryDirectory()
        self.app.folder=Path(self.temp.name)
        self.app.im=Image.new('RGB',(400,200),'white')
        self.app.display_size=(400,200)
        self.app.photo=ImageTk.PhotoImage(self.app.im)
    def tearDown(self):
        self.root.destroy();self.temp.cleanup()
    def box(self,x1,y1,x2,y2):
        self.app.press(SimpleNamespace(x=x1,y=y1))
        self.app.release(SimpleNamespace(x=x2,y=y2))
    def test_each_board_requires_explicit_policy_and_snapshot(self):
        self.box(0,0,180,200);self.box(220,0,400,200)
        self.app.rule.set('肉类专用')
        with self.assertRaises(ValueError): self.app.inspection_config(True)
        self.app.board_list.selection_set('0');self.app.assign_rule()
        self.app.rule.set('蔬菜专用');self.app.board_list.selection_set('1');self.app.assign_rule()
        config=self.app.inspection_config(True)
        self.assertEqual([r['allowed_foods'] for r in config['boards']],[['meat'],['vegetable']])
        self.app.board_rules[0]='蔬菜专用'
        self.assertEqual(config['boards'][0]['allowed_foods'],['meat'])
        rules=[BoardRule('board_1',(0,0,180,200),frozenset(['meat'])),BoardRule('board_2',(220,0,400,200),frozenset(['vegetable']))]
        result=evaluate(rules,[Detection('vegetable',.95,(20,20,100,100)),Detection('vegetable',.95,(240,20,320,100))])
        self.assertEqual([r['status'] for r in result],['suspected_misuse','no_supported_conflict'])
    def test_undo_clear_and_new_boxes_do_not_inherit_policy(self):
        self.box(10,10,100,100);self.app.rule.set('肉类专用');self.app.assign_rule()
        self.box(220,20,380,180)
        self.assertEqual(self.app.board_rules,['肉类专用',None])
        self.app.undo();self.assertEqual(len(self.app.boxes),1)
        self.assertEqual(self.app.board_rules,['肉类专用'])
        self.app.clear();self.assertEqual(self.app.board_rules,[])
        self.assertEqual(self.app.board_list.get_children(),())
    def test_automatic_mode_cannot_silently_discard_manual_boxes(self):
        with self.assertRaises(ValueError): self.app.inspection_config(False)
        self.app.rule.set('蔬菜专用')
        self.assertEqual(self.app.inspection_config(False)['all_detected_boards_allowed_foods'],['vegetable'])
        self.box(10,10,100,100)
        with self.assertRaises(ValueError): self.app.inspection_config(False)

if __name__=='__main__':unittest.main(verbosity=2)

