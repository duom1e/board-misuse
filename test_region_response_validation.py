import json,unittest
from region_response_validation import validate_region_response
class ResponseValidationTest(unittest.TestCase):
    def check_bad(self,data,code,path):
        raw=json.dumps(data);r=validate_region_response(raw)
        self.assertFalse(r['valid']);self.assertIsNone(r['normalized_boxes']);self.assertEqual(r['board_state'],'unknown');self.assertEqual(r['raw_response'],raw)
        self.assertIn((code,path),[(e['code'],e['path']) for e in r['errors']]);self.assertFalse(r['compliance_determined'])
    def test_valid_empty_is_not_compliance(self):
        r=validate_region_response('{"boards":[],"containers":[],"food":[]}');self.assertTrue(r['valid']);self.assertEqual(r['board_state'],'no_board_detected');self.assertEqual(r['status'],'needs_review');self.assertFalse(r['compliance_determined'])
    def test_known_nested_box_failure_is_not_silently_flattened(self):
        self.check_bad({'boards':[[0,0,1000,1000]],'containers':[],'food':[[312,166,443,366],[533,433,643,526],[[573,514,630,624]],[688,86,928,747]]},'box_shape','$.food[2]')
    def test_out_of_bounds(self):self.check_bad({'boards':[[0,0,1001,50]],'containers':[],'food':[]},'coordinate_range','$.boards[0][2]')
    def test_zero_area(self):self.check_bad({'boards':[[0,0,0,50]],'containers':[],'food':[]},'box_extent','$.boards[0]')
    def test_bool_is_not_integer_coordinate(self):self.check_bad({'boards':[[False,0,20,50]],'containers':[],'food':[]},'coordinate_type','$.boards[0][0]')
    def test_extra_field_invalidates_whole_response(self):self.check_bad({'boards':[],'containers':[],'food':[],'safe':True},'unexpected_field','$.safe')
    def test_missing_field_is_not_empty_list(self):self.check_bad({'boards':[],'containers':[]},'missing_field','$.food')
    def test_too_many_boxes(self):self.check_bad({'boards':[[0,0,20,50]]*7,'containers':[],'food':[]},'box_count','$.boards')
    def test_truncated_json(self):
        r=validate_region_response('{"boards":[');self.assertEqual(r['errors'][0]['code'],'json_error');self.assertEqual(r['board_state'],'unknown')
    def test_duplicate_keys_rejected(self):
        r=validate_region_response('{"boards":[],"boards":[[0,0,5,5]],"containers":[],"food":[]}');self.assertFalse(r['valid']);self.assertIn('duplicate_key',r['errors'][0]['message'])
    def test_nonfinite_rejected(self):
        r=validate_region_response('{"boards":[[0,0,NaN,50]],"containers":[],"food":[]}');self.assertFalse(r['valid'])
    def test_markdown_preserves_exact_raw(self):
        raw='```json\n{"boards":[[0,0,5,5]],"containers":[],"food":[]}\n```';r=validate_region_response(raw);self.assertTrue(r['valid']);self.assertEqual(r['raw_response'],raw);self.assertEqual(r['board_state'],'candidate_present')
if __name__=='__main__':unittest.main()
