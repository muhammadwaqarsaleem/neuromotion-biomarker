"""
Kinematic Biomarkers Module
===========================
Quantifies motor signs including tremor amplitude & frequency, bradykinesia
(velocity decrement, amplitude decay slope, rhythm coefficient of variation),
online persistent tap state machine, and standardized 10-second MDS-UPDRS
Part III (Task 3.4) clinical evaluation trials.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal

from clinical.scoring_rules import MDSUPDRSScorer
from core.signal_processing import SignalProcessor


class TapState:
    """State constants for online finger tapping state machine."""
    TOUCH = "TOUCH"        # Fingers together, aperture near minimal excursion
    OPENING = "OPENING"    # Fingers moving apart, aperture expanding
    PEAK = "PEAK"          # Local maximum aperture reached, beginning to reverse
    CLOSING = "CLOSING"    # Fingers moving together, aperture contracting


class TapEvent:
    """
    Structured record representing a confirmed finger tap event.

    Attributes:
        tap_index (int): 1-based sequential tap counter for the session.
        timestamp (float): Timestamp when the tap reached maximum aperture peak (seconds).
        touch_timestamp (float): Timestamp when tap closure/touch was confirmed (seconds).
        peak_amplitude (float): Peak-to-trough excursion amplitude in normalized aperture units.
        peak_aperture (float): Maximum normalized aperture value reached.
        trough_aperture (float): Minimum normalized aperture value at closure.
        duration (float): Approximate duration of the tap cycle in seconds.
    """

    def __init__(
        self,
        tap_index: int,
        timestamp: float,
        touch_timestamp: float,
        peak_amplitude: float,
        peak_aperture: float,
        trough_aperture: float,
        duration: float,
    ) -> None:
        self.tap_index = int(tap_index)
        self.timestamp = float(timestamp)
        self.touch_timestamp = float(touch_timestamp)
        self.peak_amplitude = float(peak_amplitude)
        self.peak_aperture = float(peak_aperture)
        self.trough_aperture = float(trough_aperture)
        self.duration = float(duration)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tap event to a dictionary."""
        return {
            "tap_index": self.tap_index,
            "timestamp": self.timestamp,
            "touch_timestamp": self.touch_timestamp,
            "peak_amplitude": self.peak_amplitude,
            "peak_aperture": self.peak_aperture,
            "trough_aperture": self.trough_aperture,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        return (
            f"TapEvent(#{self.tap_index}, t={self.timestamp:.3f}s, "
            f"amp={self.peak_amplitude:.3f}, dur={self.duration:.3f}s)"
        )


class TapStateMachine:
    """
    Persistent online state machine and peak accumulator for finger tapping.

    Maintains a strictly monotonic session tap counter (`session_total_taps`) that never
    decreases as time elapses, preventing rolling buffer truncation bugs. Tracks active
    tap states (TOUCH, OPENING, PEAK, CLOSING) using hysteresis thresholding on normalized
    aperture A(t).

    Attributes:
        delta_thresh (float): Minimum hysteresis excursion to confirm state reversal.
        touch_threshold (float): Baseline normalized aperture below which fingers are considered touching.
        session_total_taps (int): Total confirmed taps since session startup (strictly monotonic).
        tap_events (List[TapEvent]): Chronological log of all confirmed tap events.
        current_state (str): Current active TapState.
    """

    def __init__(
        self,
        delta_thresh: float = 0.06,
        touch_threshold: float = 0.22,
    ) -> None:
        """
        Initialize TapStateMachine.

        Args:
            delta_thresh: Hysteresis threshold in normalized aperture units.
            touch_threshold: Upper threshold for fingers-together touching state.
        """
        self.delta_thresh = float(delta_thresh)
        self.touch_threshold = float(touch_threshold)

        # Persistent counters & event logs (never decreased)
        self.session_total_taps: int = 0
        self.tap_events: List[TapEvent] = []

        # Internal online state tracking
        self.current_state: str = TapState.TOUCH
        self._current_trough: float = 1.0
        self._current_peak: float = 0.0
        self._t_peak: float = 0.0
        self._t_trough: float = 0.0
        self._prev_aperture: Optional[float] = None
        self._prev_timestamp: Optional[float] = None

    def update(self, timestamp: float, aperture: float) -> Optional[TapEvent]:
        """
        Ingest a live normalized aperture sample and update state transitions.

        Args:
            timestamp: Monotonic sample timestamp in seconds.
            aperture: Normalized finger aperture A(t) = ||thumb - index|| / L_ref.

        Returns:
            TapEvent if a complete tap cycle was confirmed on this frame, else None.
        """
        t = float(timestamp)
        val = float(aperture)
        confirmed_event: Optional[TapEvent] = None

        if self._prev_aperture is None:
            self._prev_aperture = val
            self._prev_timestamp = t
            self._current_trough = val
            self._current_peak = val
            self._t_trough = t
            self._t_peak = t
            self.current_state = TapState.TOUCH if val <= self.touch_threshold else TapState.OPENING
            return None

        # State Transition Logic with Hysteresis
        if self.current_state == TapState.TOUCH:
            if val < self._current_trough:
                self._current_trough = val
                self._t_trough = t

            # If aperture expands beyond hysteresis threshold, begin opening
            if val > self._current_trough + self.delta_thresh:
                self.current_state = TapState.OPENING
                self._current_peak = val
                self._t_peak = t

        elif self.current_state == TapState.OPENING:
            if val > self._current_peak:
                self._current_peak = val
                self._t_peak = t
            elif val < self._current_peak - self.delta_thresh:
                # Peak identified; reversing direction towards closing
                self.current_state = TapState.PEAK
                self._current_trough = val
                self._t_trough = t

        elif self.current_state == TapState.PEAK:
            # Brief inflection state transitioning into CLOSING
            self.current_state = TapState.CLOSING
            if val < self._current_trough:
                self._current_trough = val
                self._t_trough = t

        elif self.current_state == TapState.CLOSING:
            if val < self._current_trough:
                self._current_trough = val
                self._t_trough = t

            # Tap closure confirmed if aperture reaches bottom and rebounds, or drops below touch baseline
            rebounded = val > self._current_trough + self.delta_thresh
            touched_and_stopped = (
                val <= self.touch_threshold
                and self._prev_aperture is not None
                and val >= self._prev_aperture
            )

            if rebounded or touched_and_stopped:
                # Confirm tap cycle completion!
                self.session_total_taps += 1
                peak_amplitude = max(0.0, self._current_peak - self._current_trough)
                duration = max(0.01, t - self._t_peak)

                confirmed_event = TapEvent(
                    tap_index=self.session_total_taps,
                    timestamp=self._t_peak,
                    touch_timestamp=t,
                    peak_amplitude=peak_amplitude,
                    peak_aperture=self._current_peak,
                    trough_aperture=self._current_trough,
                    duration=duration,
                )
                self.tap_events.append(confirmed_event)

                # Reset state for next tap cycle
                self._current_trough = val
                self._t_trough = t

                if rebounded:
                    # Already opening for next tap
                    self.current_state = TapState.OPENING
                    self._current_peak = val
                    self._t_peak = t
                else:
                    self.current_state = TapState.TOUCH

        self._prev_aperture = val
        self._prev_timestamp = t
        return confirmed_event

    def get_instantaneous_cadence(self, current_timestamp: Optional[float] = None) -> float:
        """
        Calculate instantaneous tapping rate (Hz) from inter-tap intervals of the last 5 taps.

        Never divides by overall program uptime. Resets to 0.0 if the user has been idle
        without tapping for more than 2.0 seconds.

        Args:
            current_timestamp: Current system timestamp to test for user inactivity.

        Returns:
            Tapping cadence in Hz (taps/second).
        """
        if len(self.tap_events) < 2:
            return 0.0

        recent = self.tap_events[-5:]
        if len(recent) < 2:
            return 0.0

        # Inactivity check: if user paused tapping for > 2.0s, instantaneous cadence drops to 0.0
        if current_timestamp is not None:
            last_tap_t = recent[-1].touch_timestamp
            if (current_timestamp - last_tap_t) > 2.0:
                return 0.0

        # Calculate inter-tap intervals
        intervals = [
            recent[i].timestamp - recent[i - 1].timestamp
            for i in range(1, len(recent))
        ]
        valid_intervals = [dt for dt in intervals if dt > 0.04]
        if not valid_intervals:
            return 0.0

        mean_dt = float(np.mean(valid_intervals))
        return float(1.0 / mean_dt) if mean_dt > 0.0 else 0.0

    def reset_session(self) -> None:
        """Reset session accumulator counters and event logs."""
        self.session_total_taps = 0
        self.tap_events.clear()
        self.current_state = TapState.TOUCH
        self._current_trough = 1.0
        self._current_peak = 0.0
        self._t_peak = 0.0
        self._t_trough = 0.0
        self._prev_aperture = None
        self._prev_timestamp = None


class TappingTrialManager:
    """
    Manages standardized 10-second MDS-UPDRS Task 3.4 evaluation trials.

    Enables bounded, clinical trials triggered by spacebar or 'r'.
    Runs a 10-second countdown, gathers clean tap events recorded strictly
    during that active window, computes cadence, amplitude decay slope, and rhythm CV,
    and freezes the MDS-UPDRS clinical score upon completion.

    States:
        IDLE: Awaiting user trigger to start trial.
        ACTIVE: 10-second countdown running and gathering taps.
        COMPLETED: Results finalized and frozen on screen.
    """

    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"

    def __init__(self, trial_duration: float = 10.0) -> None:
        """
        Initialize TappingTrialManager.

        Args:
            trial_duration: Standard evaluation duration in seconds (default: 10.0s).
        """
        self.trial_duration: float = float(trial_duration)
        self.state: str = self.IDLE
        self.trial_start_time: Optional[float] = None
        self.trial_end_time: Optional[float] = None
        self.trial_taps: List[TapEvent] = []
        self.trial_metrics: Optional[Dict[str, Any]] = None
        self.trial_score: Optional[Dict[str, Any]] = None

    def start_trial(self, current_time: float) -> None:
        """
        Start or restart a 10-second clinical evaluation trial.

        Args:
            current_time: System timestamp when the trial is initiated.
        """
        self.state = self.ACTIVE
        self.trial_start_time = float(current_time)
        self.trial_end_time = float(current_time + self.trial_duration)
        self.trial_taps = []
        self.trial_metrics = None
        self.trial_score = None

    def reset(self) -> None:
        """Reset trial back to IDLE state."""
        self.state = self.IDLE
        self.trial_start_time = None
        self.trial_end_time = None
        self.trial_taps = []
        self.trial_metrics = None
        self.trial_score = None

    def update(
        self,
        current_time: float,
        new_tap: Optional[TapEvent] = None,
    ) -> Dict[str, Any]:
        """
        Update the trial countdown and ingest any tap confirmed on this frame.

        Args:
            current_time: Current monotonic timestamp in seconds.
            new_tap: TapEvent if one occurred on this frame.

        Returns:
            Dictionary containing state, remaining_time, tap_count, metrics, and score.
        """
        t = float(current_time)

        if self.state == self.IDLE:
            return {
                "state": self.IDLE,
                "remaining_time": self.trial_duration,
                "trial_taps_count": 0,
                "metrics": None,
                "score": None,
            }

        if self.state == self.ACTIVE:
            # If a new tap was confirmed, check if its peak falls within active trial window
            if new_tap is not None and self.trial_start_time is not None:
                if new_tap.timestamp >= self.trial_start_time:
                    self.trial_taps.append(new_tap)

            remaining = max(0.0, self.trial_end_time - t)

            # Check if 10-second trial has concluded
            if t >= self.trial_end_time:
                self._finalize_trial()

            return {
                "state": self.state,
                "remaining_time": remaining,
                "trial_taps_count": len(self.trial_taps),
                "metrics": self.trial_metrics,
                "score": self.trial_score,
            }

        # COMPLETED state (frozen metrics & score)
        return {
            "state": self.COMPLETED,
            "remaining_time": 0.0,
            "trial_taps_count": len(self.trial_taps),
            "metrics": self.trial_metrics,
            "score": self.trial_score,
        }

    def _finalize_trial(self) -> None:
        """Compute clinical metrics and MDS-UPDRS Task 3.4 score on trial data."""
        self.state = self.COMPLETED
        taps_count = len(self.trial_taps)
        duration = self.trial_duration

        # 1. Tap rate: taps during the active trial / trial duration
        tap_rate_hz = float(taps_count / duration) if duration > 0 else 0.0

        amplitudes = [e.peak_amplitude for e in self.trial_taps]
        peak_times = [e.timestamp for e in self.trial_taps]
        mean_amplitude = float(np.mean(amplitudes)) if amplitudes else 0.0

        # 2. Amplitude decay: linear regression slope across peaks recorded strictly during that trial
        if taps_count >= 2 and self.trial_start_time is not None:
            t_rel = np.array(peak_times) - self.trial_start_time
            if np.ptp(t_rel) > 0:
                poly = np.polyfit(t_rel, amplitudes, deg=1)
                decay_slope = float(poly[0])
            else:
                decay_slope = 0.0
        else:
            decay_slope = 0.0

        # 3. Rhythm CV: std(intervals) / mean(intervals) within that trial
        if taps_count >= 3:
            intervals = np.diff(peak_times)
            mean_int = float(np.mean(intervals))
            if mean_int > 0:
                std_int = float(np.std(intervals, ddof=1))
                rhythm_cv = float(std_int / mean_int)
            else:
                rhythm_cv = 0.0
        else:
            rhythm_cv = 0.0

        self.trial_metrics = {
            "total_taps": taps_count,
            "tap_rate_hz": tap_rate_hz,
            "mean_amplitude": mean_amplitude,
            "amplitude_decay_slope": decay_slope,
            "rhythm_cv": rhythm_cv,
            "amplitudes": np.array(amplitudes, dtype=np.float64),
            "peak_times": np.array(peak_times, dtype=np.float64),
            "duration": duration,
        }

        # 4. Generate clean MDS-UPDRS Part III (Task 3.4) score
        self.trial_score = MDSUPDRSScorer.score_finger_tapping(self.trial_metrics)


class FingerTappingAnalyzer:
    """
    High-level analyzer for finger tapping kinematics (MDS-UPDRS Part III, Task 3.4).

    Integrates:
      1. Persistent TapStateMachine for strictly monotonic tap counting and instantaneous cadence.
      2. TappingTrialManager for 10-second standardized clinical evaluation trials.
      3. Batch analysis fallback for processing continuous time-series arrays.
    """

    def __init__(self, default_fs: float = 30.0) -> None:
        """
        Initialize FingerTappingAnalyzer.

        Args:
            default_fs: Nominal sampling frequency in Hz.
        """
        self.default_fs = float(default_fs)
        self.state_machine = TapStateMachine()
        self.trial_manager = TappingTrialManager(trial_duration=10.0)

    @property
    def session_total_taps(self) -> int:
        """Return strictly non-decreasing session tap count."""
        return self.state_machine.session_total_taps

    @property
    def tap_events(self) -> List[TapEvent]:
        """Return full event log of confirmed taps."""
        return self.state_machine.tap_events

    def process_sample(self, timestamp: float, aperture: float) -> Dict[str, Any]:
        """
        Process a single real-time frame sample through the persistent state machine
        and active trial manager.

        Args:
            timestamp: Monotonic frame timestamp in seconds.
            aperture: Normalized finger aperture A(t).

        Returns:
            Dictionary containing state machine telemetry, instantaneous cadence,
            and active trial status.
        """
        new_tap = self.state_machine.update(timestamp, aperture)
        trial_status = self.trial_manager.update(timestamp, new_tap)
        cadence = self.state_machine.get_instantaneous_cadence(current_timestamp=timestamp)

        return {
            "current_state": self.state_machine.current_state,
            "session_total_taps": self.state_machine.session_total_taps,
            "instantaneous_cadence": cadence,
            "new_tap": new_tap,
            "trial_status": trial_status,
        }

    def start_trial(self, current_time: float) -> None:
        """Start a 10-second clinical evaluation trial."""
        self.trial_manager.start_trial(current_time)

    def reset_trial(self) -> None:
        """Reset trial back to IDLE."""
        self.trial_manager.reset()

    def analyze_tapping(
        self,
        distance_series: np.ndarray,
        timestamps: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Batch analysis of finger aperture trajectory.

        Args:
            distance_series: 1D array of finger aperture distances.
            timestamps: 1D array of relative or monotonic timestamps in seconds.

        Returns:
            Dictionary containing total_taps, tap_rate_hz, mean_amplitude,
            amplitude_decay_slope, rhythm_cv, and arrays.
        """
        d = np.asarray(distance_series, dtype=np.float64).flatten()
        t = np.asarray(timestamps, dtype=np.float64).flatten()

        empty_result: Dict[str, Any] = {
            "total_taps": 0,
            "tap_rate_hz": 0.0,
            "mean_amplitude": 0.0,
            "amplitude_decay_slope": 0.0,
            "rhythm_cv": 0.0,
            "peak_indices": np.empty((0,), dtype=int),
            "amplitudes": np.empty((0,), dtype=np.float64),
            "intervals": np.empty((0,), dtype=np.float64),
            "smoothed_distance": np.empty((0,), dtype=np.float64),
        }

        n = len(d)
        if n < 4 or len(t) < 4:
            empty_result["smoothed_distance"] = d.copy()
            return empty_result

        # 1. Smooth signal via Savitzky-Golay to suppress tracking jitter
        smoothed = SignalProcessor.smooth_signal(d, window_length=7, polyorder=2)

        # 2. Estimate sampling rate Fs from timestamps
        dt = np.diff(t)
        dt_pos = dt[dt > 0]
        if len(dt_pos) > 0:
            mean_dt = float(np.mean(dt_pos))
            fs = 1.0 / mean_dt if mean_dt > 0 else self.default_fs
        else:
            fs = self.default_fs

        if fs <= 0.0 or not np.isfinite(fs):
            fs = self.default_fs

        # 3. Dynamic peak detection
        min_distance = max(int(round(fs * 0.10)), 2)
        amp_span = float(np.ptp(smoothed))
        prominence = max(0.12 * amp_span, 0.005)

        peaks, props = signal.find_peaks(
            smoothed,
            distance=min_distance,
            prominence=prominence,
        )

        total_taps = len(peaks)
        if total_taps == 0:
            empty_result["smoothed_distance"] = smoothed
            return empty_result

        # 4. Calculate peak-to-trough amplitudes for each tap cycle
        amplitudes_list: List[float] = []
        for i, peak_idx in enumerate(peaks):
            left_bound = 0 if i == 0 else peaks[i - 1]
            left_trough = float(np.min(smoothed[left_bound:peak_idx + 1]))

            right_bound = n - 1 if i == total_taps - 1 else peaks[i + 1]
            right_trough = float(np.min(smoothed[peak_idx:right_bound + 1]))

            trough = min(left_trough, right_trough)
            amp = float(smoothed[peak_idx] - trough)
            amplitudes_list.append(max(amp, 0.0))

        amplitudes = np.array(amplitudes_list, dtype=np.float64)
        mean_amplitude = float(np.mean(amplitudes)) if len(amplitudes) > 0 else 0.0

        # 5. Inter-tap intervals & Rhythm CV
        peak_times = t[peaks]
        if total_taps >= 2:
            intervals = np.diff(peak_times)
            mean_interval = float(np.mean(intervals))
            if mean_interval > 0.0:
                tap_rate_hz = float(1.0 / mean_interval)
                std_interval = float(np.std(intervals, ddof=1 if len(intervals) > 1 else 0))
                rhythm_cv = float(std_interval / mean_interval)
            else:
                tap_rate_hz = 0.0
                rhythm_cv = 0.0
        else:
            intervals = np.empty((0,), dtype=np.float64)
            total_duration = float(t[-1] - t[0])
            tap_rate_hz = float(1.0 / total_duration) if total_duration > 0 else 0.0
            rhythm_cv = 0.0

        # 6. Amplitude decay slope
        if total_taps >= 2:
            time_rel_peaks = peak_times - peak_times[0]
            if np.ptp(time_rel_peaks) > 0:
                poly = np.polyfit(time_rel_peaks, amplitudes, deg=1)
                amplitude_decay_slope = float(poly[0])
            else:
                amplitude_decay_slope = 0.0
        else:
            amplitude_decay_slope = 0.0

        return {
            "total_taps": int(total_taps),
            "tap_rate_hz": float(tap_rate_hz),
            "mean_amplitude": float(mean_amplitude),
            "amplitude_decay_slope": float(amplitude_decay_slope),
            "rhythm_cv": float(rhythm_cv),
            "peak_indices": peaks,
            "amplitudes": amplitudes,
            "intervals": intervals,
            "smoothed_distance": smoothed,
        }


class TremorAnalyzer:
    """
    Analyzes oscillatory tremor kinematics (MDS-UPDRS Part III, Tasks 3.15 - 3.18).

    Performs spectral decomposition of spatial wrist or fingertip displacement
    trajectories and classifies tremor frequency bands:
        - 3.0 - 7.0 Hz: Parkinsonian / Resting tremor
        - 7.0 - 12.0 Hz: Physiological / Postural / Essential tremor
        - < 3.0 Hz: Voluntary baseline movement
    """

    def __init__(self, significance_threshold: float = 0.001) -> None:
        """
        Initialize TremorAnalyzer.

        Args:
            significance_threshold: Minimum PSD peak power to qualify as significant tremor.
        """
        self.significance_threshold = float(significance_threshold)

    def analyze_tremor(
        self,
        wrist_series: np.ndarray,
        timestamps: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Analyze spatial displacement trajectory of the wrist or index tip.

        Args:
            wrist_series: 1D or 2D array of coordinates (N, 3), (N, 2), or (N,).
            timestamps: Array of timestamps in seconds.

        Returns:
            Dictionary containing:
                - 'peak_freq' (float): Dominant peak frequency in Hz.
                - 'peak_power' (float): Peak power spectral density magnitude.
                - 'tremor_type' (str): Clinical band classification.
                - 'is_significant' (bool): True if tremor power exceeds threshold.
                - 'freqs' (np.ndarray): Frequency vector.
                - 'psd' (np.ndarray): Power spectral density vector.
        """
        wrist_arr = np.asarray(wrist_series, dtype=np.float64)

        if wrist_arr.ndim == 2:
            # Multi-dimensional trajectory: project onto principal axis of motion (PCA/SVD)
            # This preserves signed directional oscillation and avoids frequency doubling
            centroid = np.mean(wrist_arr, axis=0)
            centered = wrist_arr - centroid
            if centered.shape[0] >= 3:
                try:
                    _, _, vt = np.linalg.svd(centered, full_matrices=False)
                    displacement = np.dot(centered, vt[0])
                except Exception:
                    displacement = np.linalg.norm(centered, axis=1)
            else:
                displacement = np.linalg.norm(centered, axis=1)
        elif wrist_arr.ndim == 1:
            displacement = wrist_arr
        else:
            displacement = np.empty((0,), dtype=np.float64)

        freqs, psd, peak_freq, peak_power = SignalProcessor.compute_fft_spectrum(
            displacement, timestamps, tremor_band=(3.0, 12.0)
        )

        is_significant = bool(
            peak_power >= self.significance_threshold and peak_freq >= 3.0
        )

        # Classify tremor frequency band
        if peak_freq < 3.0 or not is_significant:
            tremor_type = "Voluntary baseline movement"
        elif 3.0 <= peak_freq < 7.0:
            tremor_type = "Parkinsonian / Resting tremor"
        elif 7.0 <= peak_freq <= 12.0:
            tremor_type = "Physiological / Postural / Essential tremor"
        else:
            tremor_type = "High-frequency artifact"

        return {
            "peak_freq": float(peak_freq),
            "peak_power": float(peak_power),
            "tremor_type": tremor_type,
            "is_significant": is_significant,
            "freqs": freqs,
            "psd": psd,
        }


class BiomarkerExtractor:
    """
    High-level extractor combining finger tapping and tremor analytics.
    Maintained for backward compatibility.
    """

    def __init__(self, sampling_rate: float = 30.0) -> None:
        """
        Initialize BiomarkerExtractor.

        Args:
            sampling_rate: Video acquisition frame rate in Hz.
        """
        self.sampling_rate = float(sampling_rate)
        self.tapping_analyzer = FingerTappingAnalyzer(default_fs=sampling_rate)
        self.tremor_analyzer = TremorAnalyzer()

    def extract_tremor_metrics(
        self, positions: np.ndarray, timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Extract tremor spectral features."""
        return self.tremor_analyzer.analyze_tremor(positions, timestamps)

    def extract_bradykinesia_metrics(
        self, open_close_distances: np.ndarray, timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Extract finger tapping and bradykinesia features."""
        return self.tapping_analyzer.analyze_tapping(open_close_distances, timestamps)
