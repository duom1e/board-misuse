import unittest
from video_sample_plan import plan_frames
class SamplePlanTest(unittest.TestCase):
    def test_inclusive_schedule(self):
        rows=plan_frames(0,24,2,30,800);self.assertEqual(len(rows),13);self.assertEqual(rows[-1]['frame_index'],720)
    def test_outside_video_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):plan_frames(0,1,1,30,30)
    def test_duplicate_frame_indices_rejected(self):
        with self.assertRaises(ValueError):plan_frames(0,.1,.001,30,30)
    def test_invalid_numbers(self):
        for start,end,interval in [(-1,2,1),(0,float('nan'),1),(2,1,1),(0,1,0)]:
            with self.assertRaises(ValueError):plan_frames(start,end,interval,30,100)
    def test_budget_limit_not_silent_sampling_change(self):
        with self.assertRaises(ValueError):plan_frames(0,240,1,30,10000)
    def test_requested_and_frame_times_distinguished(self):
        row=plan_frames(.01,.01,1,25,100)[0];self.assertEqual(row['requested_seconds'],.01);self.assertEqual(row['frame_seconds'],0)
if __name__=='__main__':unittest.main()
