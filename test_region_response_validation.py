import hashlib
import json
import unittest
from pathlib import Path

from region_response_validation import box_quality, validate_region_response

BASE = Path(__file__).resolve().parent
REAL_FAILURE = BASE / 'experiments' / 'qwen4b_temporal_presence_v1' / 'results.json'


class ResponseValidationTest(unittest.TestCase):
    def check_bad(self, data, code, path):
        raw = json.dumps(data)
        result = validate_region_response(raw)
        self.assertFalse(result['valid'])
        self.assertIsNone(result['normalized_boxes'])
        self.assertEqual(result['board_state'], 'unknown')
        self.assertEqual(result['raw_response'], raw)
        self.assertEqual(result['raw_sha256'], hashlib.sha256(raw.encode()).hexdigest())
        self.assertIn((code, path), [(e['code'], e['path']) for e in result['errors']])
        self.assertFalse(result['compliance_determined'])
        self.assertIn(code, result['errors_by_code'])

    def test_valid_empty_is_not_compliance(self):
        result = validate_region_response('{"boards":[],"containers":[],"food":[]}')
        self.assertTrue(result['valid'])
        self.assertEqual(result['board_state'], 'no_board_detected')
        self.assertEqual(result['status'], 'needs_review')
        self.assertFalse(result['compliance_determined'])
        self.assertEqual(result['field_counts'], {'boards': 0, 'containers': 0, 'food': 0})

    def test_nested_box_is_its_own_error_code(self):
        self.check_bad({'boards': [[0, 0, 1000, 1000]], 'containers': [],
                        'food': [[312, 166, 443, 366], [533, 433, 643, 526], [[573, 514, 630, 624]],
                                 [688, 86, 928, 747]]}, 'box_nested', '$.food[2]')

    def test_wrong_length_without_nesting_is_box_shape(self):
        self.check_bad({'boards': [[0, 0, 5]], 'containers': [], 'food': []}, 'box_shape', '$.boards[0]')

    def test_out_of_bounds(self):
        self.check_bad({'boards': [[0, 0, 1001, 50]], 'containers': [], 'food': []},
                       'coordinate_range', '$.boards[0][2]')

    def test_zero_area_is_distinct_from_inverted(self):
        self.check_bad({'boards': [[0, 0, 0, 50]], 'containers': [], 'food': []},
                       'box_extent_zero', '$.boards[0]')
        self.check_bad({'boards': [[50, 0, 10, 50]], 'containers': [], 'food': []},
                       'box_inverted', '$.boards[0]')

    def test_bool_is_not_integer_coordinate(self):
        self.check_bad({'boards': [[False, 0, 20, 50]], 'containers': [], 'food': []},
                       'coordinate_type', '$.boards[0][0]')

    def test_extra_field_invalidates_whole_response(self):
        self.check_bad({'boards': [], 'containers': [], 'food': [], 'safe': True},
                       'unexpected_field', '$.safe')

    def test_missing_field_is_not_empty_list(self):
        self.check_bad({'boards': [], 'containers': []}, 'missing_field', '$.food')

    def test_too_many_boxes(self):
        self.check_bad({'boards': [[0, 0, 20, 50]] * 7, 'containers': [], 'food': []},
                       'box_count', '$.boards')

    def test_truncated_json_has_its_own_code(self):
        result = validate_region_response('{"boards":[')
        self.assertEqual(result['errors'][0]['code'], 'json_truncated')
        self.assertEqual(result['board_state'], 'unknown')
        self.assertFalse(result['compliance_determined'])

    def test_malformed_but_closed_json_is_json_error(self):
        result = validate_region_response('{"boards":[,],"containers":[],"food":[]}')
        self.assertEqual(result['errors'][0]['code'], 'json_error')
        self.assertEqual(result['board_state'], 'unknown')

    def test_duplicate_keys_rejected(self):
        result = validate_region_response('{"boards":[],"boards":[[0,0,5,5]],"containers":[],"food":[]}')
        self.assertFalse(result['valid'])
        self.assertIn('duplicate_key', result['errors'][0]['message'])

    def test_nonfinite_rejected(self):
        result = validate_region_response('{"boards":[[0,0,NaN,50]],"containers":[],"food":[]}')
        self.assertFalse(result['valid'])

    def test_markdown_preserves_exact_raw(self):
        raw = '```json\n{"boards":[[0,0,5,5]],"containers":[],"food":[]}\n```'
        result = validate_region_response(raw)
        self.assertTrue(result['valid'])
        self.assertEqual(result['raw_response'], raw)
        self.assertEqual(result['board_state'], 'candidate_present')

    def test_field_isolation_keeps_board_judgement_but_stays_invalid(self):
        raw = json.dumps({'boards': [[100, 200, 300, 400]], 'containers': [],
                          'food': [[0, 0, 10, 10]] * 8})
        strict = validate_region_response(raw)
        self.assertFalse(strict['valid'])
        self.assertEqual(strict['board_state'], 'unknown')
        self.assertFalse(strict.get('partial_fields', False))
        partial = validate_region_response(raw, allow_partial_fields=True)
        self.assertFalse(partial['valid'], '整体仍判无效，不能悄悄当成成功')
        self.assertTrue(partial['partial_fields'])
        self.assertEqual(partial['invalid_fields'], ['food'])
        self.assertEqual(partial['board_state'], 'candidate_present')
        self.assertEqual(partial['schema_version'], 3)

    def test_field_isolation_does_not_rescue_bad_boards(self):
        raw = json.dumps({'boards': [[100, 200, 9999, 400]], 'containers': [], 'food': []})
        result = validate_region_response(raw, allow_partial_fields=True)
        self.assertEqual(result['board_state'], 'unknown')
        self.assertFalse(result['partial_fields'])


    def test_box_quality_flags_frame_filling_boxes(self):
        quality = box_quality([[0, 0, 998, 998]])
        self.assertFalse(quality['position_usable'])
        self.assertEqual(quality['reason'], 'all_boxes_cover_most_of_frame')
        self.assertEqual(quality['degenerate_boxes'][0]['index'], 0)
        tight = box_quality([[100, 120, 320, 400]])
        self.assertTrue(tight['position_usable'])
        self.assertEqual(tight['degenerate_boxes'], [])
        empty = box_quality([])
        self.assertFalse(empty['position_usable'])
        self.assertEqual(empty['reason'], 'no_boxes')

    def test_bom_is_stripped_before_parsing(self):
        raw = '\ufeff{"boards":[[0,0,5,5]],"containers":[],"food":[]}'
        result = validate_region_response(raw)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['board_state'], 'candidate_present')
    @unittest.skipUnless(REAL_FAILURE.exists(), 'saved temporal failure not present')
    def test_saved_temporal_failure_is_diagnosable(self):
        rows = json.loads(REAL_FAILURE.read_text(encoding='utf-8'))['rows']
        failures = [row for row in rows if row.get('pixel_boxes') in (None, {})]
        self.assertEqual(len(failures), 1, 'expected exactly one unconvertible response')
        row = failures[0]
        self.assertEqual(row['case'], 'prep_to_stove_010')
        result = validate_region_response(row['response'])
        self.assertFalse(result['valid'])
        self.assertEqual(result['board_state'], 'unknown')
        self.assertEqual(result['errors_by_code'].get('box_nested'), 1)
        self.assertEqual(result['errors'][0]['path'], '$.food[2]')
        self.assertEqual(result['raw_sha256'], hashlib.sha256(row['response'].encode()).hexdigest())
        self.assertGreater(result['raw_length'], 0)


if __name__ == '__main__':
    unittest.main()
