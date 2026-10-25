import unittest
from misuse_rules import BoardRule,Detection,evaluate

class RulesTest(unittest.TestCase):
    def setUp(self): self.board=BoardRule('A',(0,0,100,100),frozenset({'meat'}))
    def test_conflict(self):
        self.assertEqual(evaluate([self.board],[Detection('vegetable',.9,(20,20,40,40))])[0]['status'],'suspected_misuse')
    def test_outside(self):
        self.assertEqual(evaluate([self.board],[Detection('vegetable',.9,(110,110,130,130))])[0]['status'],'needs_review')
    def test_low_confidence(self):
        self.assertEqual(evaluate([self.board],[Detection('vegetable',.2,(20,20,40,40))])[0]['status'],'needs_review')
    def test_empty_never_passes(self):
        self.assertEqual(evaluate([self.board],[])[0]['status'],'needs_review')
    def test_allowed_does_not_certify_compliance(self):
        b=BoardRule('B',(0,0,100,100),frozenset({'vegetable'}))
        self.assertEqual(evaluate([b],[Detection('vegetable',.9,(20,20,40,40))])[0]['status'],'no_supported_conflict')
    def test_ambiguous_boards(self):
        b=BoardRule('B',(10,10,90,90),frozenset({'vegetable'}))
        result=evaluate([self.board,b],[Detection('vegetable',.9,(20,20,40,40))])
        self.assertTrue(all(r['status']=='needs_review' for r in result))
    def test_edge_not_enough_overlap(self):
        self.assertEqual(evaluate([self.board],[Detection('vegetable',.9,(90,90,110,110))])[0]['status'],'needs_review')
    def test_invalid_detection(self):
        with self.assertRaises(ValueError): evaluate([self.board],[Detection('vegetable',.9,(40,20,20,40))])

if __name__=='__main__': unittest.main()
