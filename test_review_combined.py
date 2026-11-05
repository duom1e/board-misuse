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
            self.assertFalse(result["board_absence_verified"])
            self.assertEqual(result["board_observation"], "not_detected")
            self.assertIn("未检出案板不等于", rc.report_text(result))
            self.assertFalse(result["position_usable"])
        finally:
            os.unlink(tmp.name)


class PositionRegressionTest(unittest.TestCase):
    def test_full_frame_rejected_at_all_resolutions(self):
        for w,h in [(640,360),(1280,720),(1920,1080)]:
            q, boxes, usable = rc.position_evidence([[0,0,w,h]],w,h,'yolo','detector_only_present')
            self.assertFalse(usable)
            self.assertEqual(boxes, [])
            self.assertEqual(q['reason'], 'all_boxes_cover_most_of_frame')

    def test_fallback_usable_only_when_presence_supported(self):
        for presence, expected in [('present_vlm_box_coarse_using_yolo',True),('unknown',False)]:
            self.assertEqual(rc.position_evidence([[10,10,100,100]],640,360,'yolo_fallback',presence)[2],expected)

    def test_mixed_boxes_cannot_assign_food_using_full_frame(self):
        from food_region_gate import locate
        _, boxes, usable = rc.position_evidence([[0,0,640,360],[10,10,100,100]],640,360,'vlm','present')
        self.assertTrue(usable)
        self.assertEqual(locate([300,200,350,250],boxes,[],usable)['location'],'uncertain')

    def test_human_full_frame_is_not_location_evidence(self):
        self.assertFalse(rc.position_evidence([[0,0,640,360]],640,360,'human','present_human_box')[2])

    def test_invalid_human_boxes_rejected(self):
        for box in [[0,0,0,2],[-1,0,100,100],[0,0,float('nan'),100],[0,0,1001,100],[False,0,100,100]]:
            with self.assertRaises(ValueError):
                rc.validate_human_box(box)




class TilingFallbackTest(unittest.TestCase):
    def test_tile_boxes_map_back_to_frame(self):
        # 右下角块：x 500..1000, y 500..1000；块内框 (0,0)-(1000,1000) 应铺满该块
        mapped = rc.map_tile_boxes_to_frame((500, 500, 1000, 1000), [[0, 0, 1000, 1000]])
        self.assertEqual(mapped, [[500.0, 500.0, 1000.0, 1000.0]])
        half = rc.map_tile_boxes_to_frame((0, 0, 400, 200), [[500, 500, 1000, 1000]])
        self.assertEqual(half, [[200.0, 100.0, 400.0, 200.0]])

    def test_fuse_prefers_tiling_over_absent(self):
        presence, boxes, source = rc.fuse('no_board_detected', None, None, [],
                                          tiling={'boxes_px': [[10, 10, 20, 20]]})
        self.assertEqual((presence, source), ('present_tiling_fallback', 'vlm_tiling'))
        self.assertEqual(boxes, [[10, 10, 20, 20]])

    def test_fuse_without_tiling_boxes_stays_absent(self):
        presence, _, source = rc.fuse('no_board_detected', None, None, [], tiling={'boxes_px': []})
        self.assertEqual((presence, source), ('absent', 'none'))

    def test_tiling_source_is_usable_for_position(self):
        quality, eligible, usable = rc.position_evidence([[100, 100, 300, 400]], 640, 480, 'vlm_tiling', 'present_tiling_fallback')
        self.assertTrue(usable)
        self.assertTrue(eligible)


if __name__ == '__main__':
    unittest.main()
