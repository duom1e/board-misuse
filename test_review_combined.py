import unittest

import review_combined as rc


class FuseTest(unittest.TestCase):
    def test_vlm_present_with_tight_box_wins(self):
        quality = {'position_usable': True, 'reason': None}
        presence, boxes, source = rc.fuse('candidate_present', quality, [[1, 2, 3, 4]], [[9, 9, 9, 9]])
        self.assertEqual((presence, boxes, source), ('present', [[1, 2, 3, 4]], 'vlm'))

    def test_vlm_present_but_coarse_falls_back_to_yolo(self):
        quality = {'position_usable': False, 'reason': 'all_boxes_cover_most_of_frame'}
        presence, boxes, source = rc.fuse('candidate_present', quality, [[0, 0, 1000, 1000]], [[10, 10, 20, 20]])
        self.assertEqual(presence, 'present_vlm_box_coarse_using_yolo')
        self.assertEqual(source, 'yolo_fallback')
        self.assertEqual(boxes, [[10, 10, 20, 20]])

    def test_vlm_present_coarse_without_yolo_keeps_coarse_flag(self):
        quality = {'position_usable': False, 'reason': 'all_boxes_cover_most_of_frame'}
        presence, _, source = rc.fuse('candidate_present', quality, [[0, 0, 1000, 1000]], [])
        self.assertEqual((presence, source), ('present_position_coarse', 'vlm_box_coarse'))

    def test_vlm_absent_ignores_yolo_by_default(self):
        presence, boxes, source = rc.fuse('no_board_detected', None, None, [[1, 1, 5, 5]])
        self.assertEqual((presence, boxes, source), ('absent', [], 'none'))

    def test_cross_check_only_when_explicitly_enabled(self):
        presence, _, source = rc.fuse('no_board_detected', None, None, [[1, 1, 5, 5]], cross_check=True)
        self.assertEqual((presence, source), ('uncertain_vlm_no_yolo_yes', 'yolo_crosscheck'))

    def test_no_vlm_mode_uses_detector_only(self):
        self.assertEqual(rc.fuse(None, None, None, [[1, 1, 2, 2]])[0], 'detector_only_present')
        self.assertEqual(rc.fuse(None, None, None, [])[0], 'detector_only_absent')

    def test_unknown_vlm_state_stays_unknown(self):
        presence, _, source = rc.fuse('unknown', None, None, [])
        self.assertEqual((presence, source), ('unknown', 'none'))




class _NoDetections:
    boxes = None


class _StubModel:
    def predict(self, **kwargs):
        return [_NoDetections()]


class HumanBoxTest(unittest.TestCase):
    def test_human_box_overrides_model_position(self):
        from PIL import Image
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        Image.new("RGB", (640, 480), "white").save(tmp.name)
        try:
            result, _ = rc.analyze(tmp.name, {"board": _StubModel(), "food": _StubModel()},
                                   no_vlm=True, human_box=[100, 100, 500, 600])
            self.assertEqual(result["board_presence"], "present_human_box")
            self.assertEqual(result["board_source"], "human")
            self.assertTrue(result["position_usable"])
            self.assertEqual(len(result["board_boxes"]), 1)
        finally:
            os.unlink(tmp.name)

    def test_without_human_box_falls_back_to_rules(self):
        from PIL import Image
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        Image.new("RGB", (320, 240), "white").save(tmp.name)
        try:
            result, _ = rc.analyze(tmp.name, {"board": _StubModel(), "food": _StubModel()}, no_vlm=True)
            self.assertEqual(result["board_presence"], "detector_only_absent")
            self.assertFalse(result["position_usable"])
        finally:
            os.unlink(tmp.name)


if __name__ == '__main__':
    unittest.main()
