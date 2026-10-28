import unittest
from food_region_gate import locate
class GateTest(unittest.TestCase):
    def test_threshold_boundaries_and_reasons(self):
        board=[[0,0,8,10]];food=[0,0,10,10]
        self.assertEqual(locate(food,board)['location_reason'],'mostly_inside_board_box')
        r=locate(food,board,[[0,0,2,10]])
        self.assertEqual(r['location'],'board_candidate');self.assertTrue(r['geometry_only'])
        self.assertEqual(locate(food,board,[[0,0,2.01,10]])['location_reason'],'partial_container_overlap')
        self.assertEqual(locate(food,board,[[0,0,8,10]])['location_reason'],'mostly_inside_container_box')
        self.assertEqual(locate(food,[[0,0,7.99,10]])['location_reason'],'insufficient_board_box_coverage')
    def test_container_inside_board_has_priority(self):
        self.assertEqual(locate([3,3,7,7],[[0,0,10,10]],[[2,2,8,8]])['location'],'container_candidate')
    def test_partial_container_overlap_is_uncertain(self):
        self.assertEqual(locate([0,0,10,10],[[0,0,10,10]],[[0,0,5,10]])['location'],'uncertain')
    def test_no_regions_is_not_empty_or_compliant(self):
        r=locate([0,0,10,10]);self.assertEqual(r['location'],'uncertain');self.assertFalse(r['compliance_determined'])
    def test_invalid_boxes_rejected(self):
        for box in [[0,0,0,1],[0,0,float('nan'),1]]:
            with self.assertRaises(ValueError):locate(box)
    def test_board_candidate_requires_review(self):
        r=locate([1,1,5,5],[[0,0,10,10]]);self.assertEqual(r['location'],'board_candidate');self.assertEqual(r['status'],'needs_review');self.assertFalse(r['physical_contact_verified'])
if __name__=='__main__':unittest.main()
