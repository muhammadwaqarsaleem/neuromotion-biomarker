"""
Unit Tests for Signal Processing & Kinematic Biomarkers
=======================================================
Verifies Savitzky-Golay smoothing, FFT spectral estimation,
FingerTappingAnalyzer, TremorAnalyzer, and MDSUPDRSScorer.
"""

import unittest
import numpy as np

from core.signal_processing import SignalProcessor, SignalFilter
from core.biomarkers import (
    FingerTappingAnalyzer,
    TremorAnalyzer,
    TapStateMachine,
    TappingTrialManager,
    TapState,
)
from clinical.scoring_rules import MDSUPDRSScorer


class TestSignalProcessor(unittest.TestCase):
    """Test Savitzky-Golay filtering and FFT spectral decomposition."""

    def test_smooth_signal_normal(self):
        # 1D noisy sine wave
        t = np.linspace(0, 1, 50)
        clean = np.sin(2 * np.pi * 2 * t)
        noisy = clean + np.random.normal(0, 0.05, len(t))

        smoothed = SignalProcessor.smooth_signal(noisy, window_length=7, polyorder=2)
        self.assertEqual(smoothed.shape, noisy.shape)
        # Residual variance between smoothed and clean should be lower than noisy and clean
        self.assertLess(np.var(smoothed - clean), np.var(noisy - clean))

    def test_smooth_signal_edge_cases(self):
        # Empty array
        empty = np.empty((0,), dtype=np.float64)
        self.assertEqual(len(SignalProcessor.smooth_signal(empty)), 0)

        # Single element or fewer than polyorder
        single = np.array([3.14])
        smoothed_single = SignalProcessor.smooth_signal(single, window_length=7, polyorder=2)
        self.assertEqual(len(smoothed_single), 1)
        self.assertAlmostEqual(smoothed_single[0], 3.14)

        # Length smaller than window_length (e.g. 5 samples with window 7)
        small = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        smoothed_small = SignalProcessor.smooth_signal(small, window_length=7, polyorder=2)
        self.assertEqual(len(smoothed_small), 5)

    def test_compute_fft_spectrum(self):
        fs = 30.0
        duration = 4.0
        n = int(fs * duration)
        t = np.linspace(0, duration, n)

        # Pure 6.0 Hz tremor signal (Parkinsonian band)
        freq_target = 6.0
        signal_val = 0.08 * np.sin(2 * np.pi * freq_target * t)

        freqs, psd, peak_freq, peak_power = SignalProcessor.compute_fft_spectrum(
            signal_val, t, tremor_band=(3.0, 12.0)
        )

        self.assertGreater(len(freqs), 0)
        self.assertGreater(len(psd), 0)
        self.assertAlmostEqual(peak_freq, freq_target, delta=0.35)
        self.assertGreater(peak_power, 0.001)

    def test_compute_fft_empty(self):
        freqs, psd, pf, pp = SignalProcessor.compute_fft_spectrum(
            np.empty((0,)), np.empty((0,))
        )
        self.assertEqual(len(freqs), 0)
        self.assertEqual(pf, 0.0)
        self.assertEqual(pp, 0.0)


class TestFingerTappingAnalyzer(unittest.TestCase):
    """Test finger tapping kinematics and bradykinesia metrics."""

    def test_regular_tapping(self):
        fs = 30.0
        duration = 6.0
        n = int(fs * duration)
        t = np.linspace(0, duration, n)

        # 2.0 Hz tapping, constant amplitude
        cadence = 2.0
        d = 0.02 + 0.10 * (0.5 * (1.0 - np.cos(2 * np.pi * cadence * t)))

        analyzer = FingerTappingAnalyzer(default_fs=fs)
        metrics = analyzer.analyze_tapping(d, t)

        self.assertGreaterEqual(metrics["total_taps"], 10)
        self.assertAlmostEqual(metrics["tap_rate_hz"], cadence, delta=0.25)
        self.assertAlmostEqual(metrics["mean_amplitude"], 0.10, delta=0.03)
        self.assertAlmostEqual(metrics["amplitude_decay_slope"], 0.0, delta=0.01)
        self.assertLess(metrics["rhythm_cv"], 0.10)

    def test_fatigue_decrement_slope(self):
        fs = 30.0
        duration = 5.0
        n = int(fs * duration)
        t = np.linspace(0, duration, n)

        cadence = 2.5
        decay = 0.025
        amplitude = np.maximum(0.15 - decay * t, 0.02)
        d = 0.01 + amplitude * (0.5 * (1.0 - np.cos(2 * np.pi * cadence * t)))

        analyzer = FingerTappingAnalyzer(default_fs=fs)
        metrics = analyzer.analyze_tapping(d, t)

        self.assertLess(metrics["amplitude_decay_slope"], -0.015)

    def test_empty_signal(self):
        analyzer = FingerTappingAnalyzer()
        metrics = analyzer.analyze_tapping(np.empty((0,)), np.empty((0,)))
        self.assertEqual(metrics["total_taps"], 0)
        self.assertEqual(metrics["tap_rate_hz"], 0.0)


class TestTremorAnalyzer(unittest.TestCase):
    """Test tremor band spectral decomposition and classification."""

    def test_parkinsonian_tremor_classification(self):
        fs = 30.0
        duration = 4.0
        t = np.linspace(0, duration, int(fs * duration))
        # 5.0 Hz oscillation in x-coordinate
        x = 0.04 * np.sin(2 * np.pi * 5.0 * t)
        y = np.zeros_like(t)
        z = np.zeros_like(t)
        wrist = np.column_stack([x, y, z])

        analyzer = TremorAnalyzer()
        res = analyzer.analyze_tremor(wrist, t)

        self.assertAlmostEqual(res["peak_freq"], 5.0, delta=0.35)
        self.assertEqual(res["tremor_type"], "Parkinsonian / Resting tremor")
        self.assertTrue(res["is_significant"])

    def test_postural_essential_tremor_classification(self):
        fs = 30.0
        duration = 4.0
        t = np.linspace(0, duration, int(fs * duration))
        # 9.0 Hz oscillation (Essential / Physiological tremor band)
        x = 0.03 * np.sin(2 * np.pi * 9.0 * t)
        wrist = np.column_stack([x, np.zeros_like(t), np.zeros_like(t)])

        analyzer = TremorAnalyzer()
        res = analyzer.analyze_tremor(wrist, t)

        self.assertAlmostEqual(res["peak_freq"], 9.0, delta=0.35)
        self.assertEqual(res["tremor_type"], "Physiological / Postural / Essential tremor")
        self.assertTrue(res["is_significant"])

    def test_voluntary_movement(self):
        fs = 30.0
        duration = 4.0
        t = np.linspace(0, duration, int(fs * duration))
        # 1.5 Hz slow voluntary movement
        x = 0.10 * np.sin(2 * np.pi * 1.5 * t)
        wrist = np.column_stack([x, np.zeros_like(t), np.zeros_like(t)])

        analyzer = TremorAnalyzer()
        res = analyzer.analyze_tremor(wrist, t)
        self.assertEqual(res["tremor_type"], "Voluntary baseline movement")


class TestMDSUPDRSScorer(unittest.TestCase):
    """Test MDS-UPDRS scoring rule evaluations."""

    def test_score_finger_tapping_normal(self):
        metrics = {
            "total_taps": 15,
            "tap_rate_hz": 2.8,
            "amplitude_decay_slope": -0.005,
            "rhythm_cv": 0.12,
        }
        res = MDSUPDRSScorer.score_finger_tapping(metrics)
        self.assertEqual(res["score"], 0)
        self.assertEqual(res["label"], "Normal")

    def test_score_finger_tapping_slight(self):
        # Slight slowing (rate 2.2 taps/s)
        metrics = {
            "total_taps": 12,
            "tap_rate_hz": 2.2,
            "amplitude_decay_slope": -0.005,
            "rhythm_cv": 0.15,
        }
        res = MDSUPDRSScorer.score_finger_tapping(metrics)
        self.assertEqual(res["score"], 1)
        self.assertEqual(res["label"], "Slight")

    def test_score_finger_tapping_moderate_decrement(self):
        # Moderate decay slope (-0.045)
        metrics = {
            "total_taps": 10,
            "tap_rate_hz": 2.6,
            "amplitude_decay_slope": -0.045,
            "rhythm_cv": 0.15,
        }
        res = MDSUPDRSScorer.score_finger_tapping(metrics)
        self.assertEqual(res["score"], 2)
        self.assertEqual(res["label"], "Mild")

    def test_score_finger_tapping_severe_decay(self):
        # Severe decay slope (-0.070)
        metrics = {
            "total_taps": 8,
            "tap_rate_hz": 2.0,
            "amplitude_decay_slope": -0.070,
            "rhythm_cv": 0.15,
        }
        res = MDSUPDRSScorer.score_finger_tapping(metrics)
        self.assertEqual(res["score"], 3)
        self.assertEqual(res["label"], "Moderate")

    def test_score_finger_tapping_severe_unable(self):
        # Unable to sustain taps
        metrics = {
            "total_taps": 1,
            "tap_rate_hz": 0.3,
            "amplitude_decay_slope": 0.0,
            "rhythm_cv": 0.0,
        }
        res = MDSUPDRSScorer.score_finger_tapping(metrics)
        self.assertEqual(res["score"], 4)
        self.assertEqual(res["label"], "Severe")

    def test_score_tremor(self):
        # Normal
        norm_metrics = {"peak_freq": 2.0, "peak_power": 0.0002, "is_significant": False}
        self.assertEqual(MDSUPDRSScorer.score_tremor(norm_metrics)["score"], 0)

        # Slight
        slight_metrics = {"peak_freq": 5.0, "peak_power": 0.005, "is_significant": True}
        self.assertEqual(MDSUPDRSScorer.score_tremor(slight_metrics)["score"], 1)

        # Mild
        mild_metrics = {"peak_freq": 5.0, "peak_power": 0.03, "is_significant": True}
        self.assertEqual(MDSUPDRSScorer.score_tremor(mild_metrics)["score"], 2)

        # Moderate
        mod_metrics = {"peak_freq": 4.5, "peak_power": 0.08, "is_significant": True}
        self.assertEqual(MDSUPDRSScorer.score_tremor(mod_metrics)["score"], 3)

        # Severe
        sev_metrics = {"peak_freq": 4.8, "peak_power": 0.20, "is_significant": True}
        self.assertEqual(MDSUPDRSScorer.score_tremor(sev_metrics)["score"], 4)


class TestTapStateMachine(unittest.TestCase):
    """Test online persistent TapStateMachine and monotonic counter."""

    def test_state_machine_monotonicity(self):
        sm = TapStateMachine(delta_thresh=0.06, touch_threshold=0.22)
        fs = 30.0
        duration = 4.0
        t = np.linspace(0, duration, int(fs * duration))
        cadence = 2.5
        aperture = 0.12 + 0.38 * (0.5 * (1.0 - np.cos(2 * np.pi * cadence * t)))

        counts = []
        for i in range(len(t)):
            sm.update(t[i], aperture[i])
            counts.append(sm.session_total_taps)

        # Monotonicity: session_total_taps must never decrease
        counts_arr = np.array(counts)
        self.assertTrue(np.all(np.diff(counts_arr) >= 0))
        self.assertGreaterEqual(sm.session_total_taps, 8)
        self.assertEqual(len(sm.tap_events), sm.session_total_taps)

        # Instantaneous cadence from last 5 taps
        cadence_est = sm.get_instantaneous_cadence(current_timestamp=t[-1])
        self.assertAlmostEqual(cadence_est, 2.5, delta=0.3)


class TestTappingTrialManager(unittest.TestCase):
    """Test 10-second MDS-UPDRS Task 3.4 trial management."""

    def test_trial_lifecycle_and_scoring(self):
        trial = TappingTrialManager(trial_duration=10.0)
        self.assertEqual(trial.state, TappingTrialManager.IDLE)

        trial.start_trial(current_time=0.0)
        self.assertEqual(trial.state, TappingTrialManager.ACTIVE)

        sm = TapStateMachine()
        fs = 30.0
        t = np.linspace(0, 10.0, int(fs * 10.0))
        aperture = 0.10 + 0.40 * (0.5 * (1.0 - np.cos(2 * np.pi * 2.5 * t)))

        for i in range(len(t)):
            event = sm.update(t[i], aperture[i])
            trial.update(current_time=t[i], new_tap=event)

        # Finalize trial at t=10.01
        res = trial.update(current_time=10.01, new_tap=None)
        self.assertEqual(res["state"], TappingTrialManager.COMPLETED)
        self.assertIsNotNone(res["metrics"])
        self.assertIsNotNone(res["score"])
        self.assertGreaterEqual(res["metrics"]["total_taps"], 20)
        self.assertAlmostEqual(res["metrics"]["tap_rate_hz"], 2.4, delta=0.4)
        self.assertIn(res["score"]["score"], (0, 1))


if __name__ == "__main__":
    unittest.main()
