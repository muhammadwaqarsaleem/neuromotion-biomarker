"""
Unit Tests for NeuroMotion Core Vision Pipeline
================================================
Comprehensive test suite verifying HandTracker, KinematicBuffer, and VisionPipeline.
"""

import threading
import time
import unittest
import numpy as np

from core.vision import (
    HandTracker,
    KinematicBuffer,
    VisionPipeline,
    HandednessInfo,
    WRIST_IDX,
    THUMB_TIP_IDX,
    INDEX_TIP_IDX,
    MIDDLE_TIP_IDX,
    KEY_LANDMARK_INDICES,
)


class TestHandednessInfo(unittest.TestCase):
    """Test HandednessInfo dictionary and attribute semantics."""

    def test_handedness_properties(self):
        info = HandednessInfo(label="Right", score=0.9876)
        self.assertEqual(info["label"], "Right")
        self.assertAlmostEqual(info["score"], 0.9876)
        self.assertEqual(info.label, "Right")
        self.assertAlmostEqual(info.score, 0.9876)
        self.assertEqual(str(info), "Right (0.99)")
        self.assertEqual(info, "Right")
        self.assertEqual(info, {"label": "Right", "score": 0.9876})


class TestKinematicBuffer(unittest.TestCase):
    """Test KinematicBuffer rolling FIFO, thread safety, and time series calculation."""

    def test_initial_state(self):
        buf = KinematicBuffer(buffer_size=100)
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.current_size, 0)
        self.assertFalse(buf.is_ready(min_samples=10))

        empty_series = buf.get_time_series()
        self.assertEqual(empty_series["time"].shape, (0,))
        self.assertEqual(empty_series["thumb_tip"].shape, (0, 3))
        self.assertEqual(empty_series["index_tip"].shape, (0, 3))
        self.assertEqual(empty_series["wrist"].shape, (0, 3))
        self.assertEqual(empty_series["euclidean_distance"].shape, (0,))

    def test_push_and_fifo_eviction(self):
        capacity = 5
        buf = KinematicBuffer(buffer_size=capacity)

        # Push 8 samples into a capacity-5 buffer
        for i in range(8):
            t = 100.0 + i * 0.1
            thumb = (i, 0.0, 0.0)
            index = (i + 1.0, 0.0, 0.0)
            wrist = (0.0, i, 0.0)
            buf.push(t, thumb, index, wrist)

        self.assertEqual(len(buf), capacity)
        self.assertTrue(buf.is_ready(min_samples=capacity))

        series = buf.get_time_series()
        # The buffer should contain samples i = 3, 4, 5, 6, 7
        self.assertEqual(series["time"].shape, (capacity,))
        # Relative time should start at 0.0
        self.assertAlmostEqual(series["time"][0], 0.0)
        self.assertAlmostEqual(series["time"][-1], 0.4)

        # Check coordinates of first remaining sample (i=3) and last sample (i=7)
        self.assertAlmostEqual(series["thumb_tip"][0, 0], 3.0)
        self.assertAlmostEqual(series["thumb_tip"][-1, 0], 7.0)

        # Euclidean distance should be exactly 1.0 for each frame (since thumb is (i,0,0) and index is (i+1,0,0))
        expected_dist = np.ones(capacity)
        np.testing.assert_allclose(series["euclidean_distance"], expected_dist)

    def test_3d_euclidean_distance(self):
        buf = KinematicBuffer(buffer_size=10)
        # Thumb at (1, 2, 3), Index at (4, 6, 3) -> distance = sqrt(3^2 + 4^2 + 0^2) = 5.0
        buf.push(1.0, (1.0, 2.0, 3.0), (4.0, 6.0, 3.0), (0.0, 0.0, 0.0))
        series = buf.get_time_series()
        self.assertEqual(len(series["euclidean_distance"]), 1)
        self.assertAlmostEqual(series["euclidean_distance"][0], 5.0)

    def test_strictly_monotonic_timestamps(self):
        buf = KinematicBuffer(buffer_size=10)
        # Push timestamps that have duplicate or jitter values
        buf.push(10.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        buf.push(10.0, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        buf.push(10.05, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        series = buf.get_time_series()
        t = series["time"]
        self.assertEqual(len(t), 3)
        self.assertTrue(np.all(np.diff(t) > 0))
        self.assertAlmostEqual(t[0], 0.0)

    def test_scale_normalized_aperture(self):
        buf = KinematicBuffer(buffer_size=10)
        # Wrist at (0, 0, 0), Middle MCP at (0, 0.20, 0) -> L_ref = 0.20
        # Thumb at (0, 0, 0), Index at (0.10, 0, 0) -> Euclidean distance = 0.10
        # Normalized aperture A(t) = 0.10 / 0.20 = 0.50
        buf.push(
            timestamp=1.0,
            thumb_tip=(0.0, 0.0, 0.0),
            index_tip=(0.10, 0.0, 0.0),
            wrist=(0.0, 0.0, 0.0),
            middle_mcp=(0.0, 0.20, 0.0),
        )
        series = buf.get_time_series()
        self.assertAlmostEqual(series["euclidean_distance"][0], 0.10)
        self.assertAlmostEqual(series["l_ref"][0], 0.20)
        self.assertAlmostEqual(series["normalized_aperture"][0], 0.50)

    def test_thread_safety(self):
        buf = KinematicBuffer(buffer_size=50)

        def worker(start_val: int):
            for j in range(25):
                buf.push(
                    time.perf_counter(),
                    (start_val, 0.0, 0.0),
                    (start_val + 1, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                )
                time.sleep(0.001)

        threads = [
            threading.Thread(target=worker, args=(10,)),
            threading.Thread(target=worker, args=(20,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(buf), 50)
        series = buf.get_time_series()
        self.assertEqual(series["time"].shape, (50,))

    def test_clear(self):
        buf = KinematicBuffer(buffer_size=20)
        buf.push(1.0, (0, 0, 0), (1, 1, 1), (2, 2, 2))
        self.assertEqual(len(buf), 1)
        buf.clear()
        self.assertEqual(len(buf), 0)
        self.assertFalse(buf.is_ready(min_samples=1))


class TestHandTracker(unittest.TestCase):
    """Test HandTracker processing, empty frames, and landmark formats."""

    def setUp(self):
        self.tracker = HandTracker(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def tearDown(self):
        self.tracker.close()

    def test_empty_and_invalid_frames(self):
        empty_frame = np.empty((0,), dtype=np.uint8)
        annotated, data = self.tracker.process_frame(empty_frame)
        self.assertIsNone(data)

    def test_synthetic_black_frame(self):
        black_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        annotated, data = self.tracker.process_frame(black_frame)
        self.assertEqual(annotated.shape, (300, 400, 3))
        self.assertIsNone(data)

    def test_context_manager(self):
        with HandTracker() as tr:
            self.assertIsNotNone(tr)


class TestVisionPipelineFacade(unittest.TestCase):
    """Test VisionPipeline coordinator integrating tracker and buffer."""

    def test_pipeline_integration(self):
        pipeline = VisionPipeline(confidence_threshold=0.7, buffer_size=30)
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        annotated, data = pipeline.process_frame(blank)
        self.assertEqual(annotated.shape, (240, 320, 3))
        self.assertIsNone(data)

        series = pipeline.get_kinematic_series()
        self.assertEqual(len(series["time"]), 0)
        pipeline.close()


if __name__ == "__main__":
    unittest.main()
