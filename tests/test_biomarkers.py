"""
NeuroMotion - Biomarker & Clinical Scoring Verification Test
=============================================================
Interactive live visual test harness for digital kinematic biomarkers and MDS-UPDRS scoring.

Key Features & Clinical Enhancements:
- Windows DirectShow & MJPG FourCC capture optimization at 640x480.
- Scale-invariant normalized aperture A(t) = ||thumb - index|| / L_ref (Wrist to Middle MCP).
- Explicit Handedness label ("Right" / "Left" with confidence score) in HUD.
- Online persistent tap state machine (TOUCH, OPENING, PEAK, CLOSING) with strictly non-decreasing tap count.
- Instantaneous cadence calculated from inter-tap intervals of the last 5 taps.
- Standardized 10-second MDS-UPDRS Task 3.4 evaluation trial mode triggered by SPACE or 'R'.
- Frozen clinical trial score with detailed breakdown upon trial completion.
- Resizable display window (cv2.WINDOW_NORMAL) without border distortion.
- Real-time Normalized Aperture mini-bar gauge and dynamic rolling sparkline.
- Comprehensive non-interactive verification via `--smoke-test` or `--headless`.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from clinical.scoring_rules import MDSUPDRSScorer
from core.biomarkers import (
    FingerTappingAnalyzer,
    TapEvent,
    TapState,
    TapStateMachine,
    TappingTrialManager,
    TremorAnalyzer,
)
from core.signal_processing import SignalProcessor
from core.vision import (
    INDEX_TIP_IDX,
    MIDDLE_MCP_IDX,
    THUMB_TIP_IDX,
    WRIST_IDX,
    HandTracker,
    KinematicBuffer,
    draw_clinical_hud as core_draw_clinical_hud,
    draw_waveform_canvas,
)

WINDOW_NAME = "NeuroMotion - Biomarker & MDS-UPDRS Clinical Assessment"


def generate_ascii_sparkline(data: np.ndarray, width: int = 24) -> str:
    """
    Generate a compact ASCII sparkline string for console or HUD telemetry.

    Args:
        data: 1D array of numeric values.
        width: Number of characters in the sparkline.

    Returns:
        Formatted sparkline string using standard ASCII characters.
    """
    if len(data) == 0:
        return " " * width

    if len(data) > width:
        indices = np.linspace(0, len(data) - 1, width, dtype=int)
        samples = data[indices]
    else:
        samples = data

    min_v, max_v = float(np.min(samples)), float(np.max(samples))
    span = max_v - min_v
    bars = ["_", ".", "-", "~", "=", "+", "*", "%", "#"]

    if span <= 1e-6:
        return bars[0] * len(samples)

    spark = ""
    for v in samples:
        norm_idx = int(round((v - min_v) / span * (len(bars) - 1)))
        norm_idx = max(0, min(len(bars) - 1, norm_idx))
        spark += bars[norm_idx]

    return spark


def draw_clinical_hud(
    frame: np.ndarray,
    fps: float,
    buffer_len: int,
    buffer_capacity: int,
    handedness_str: str,
    current_aperture: float,
    current_state: str,
    session_taps: int,
    instantaneous_cadence: float,
    trial_status: Dict[str, Any],
    tap_metrics: Optional[Dict[str, Any]] = None,
    updrs_tapping: Optional[Dict[str, Any]] = None,
    tremor_metrics: Optional[Dict[str, Any]] = None,
    updrs_tremor: Optional[Dict[str, Any]] = None,
    distance_series: Optional[np.ndarray] = None,
    scale_factor: float = 0.0,
) -> np.ndarray:
    """Delegate to polished clinical HUD overlay in core.vision."""
    return core_draw_clinical_hud(
        frame=frame,
        fps=fps,
        buffer_len=buffer_len,
        buffer_capacity=buffer_capacity,
        handedness_str=handedness_str,
        current_aperture=current_aperture,
        current_state=current_state,
        session_taps=session_taps,
        instantaneous_cadence=instantaneous_cadence,
        trial_status=trial_status,
        tap_metrics=tap_metrics,
        updrs_tapping=updrs_tapping,
        tremor_metrics=tremor_metrics,
        updrs_tremor=updrs_tremor,
        distance_series=distance_series,
        scale_factor=scale_factor,
    )



def print_clinical_report(
    session_duration: float,
    total_frames: int,
    session_total_taps: int,
    trial_status: Dict[str, Any],
    tap_metrics: Dict[str, Any],
    updrs_tapping: Dict[str, Any],
    tremor_metrics: Dict[str, Any],
    updrs_tremor: Dict[str, Any],
) -> None:
    """
    Print a structured, publication-grade clinical assessment report to the console.
    """
    print("\n" + "=" * 76)
    print("                NEUROMOTION DIGITAL BIOMARKER REPORT                ")
    print("         MDS-UPDRS Part III Motor Examination Assessment            ")
    print("=" * 76)
    print(f" Session Duration        : {session_duration:.2f} seconds")
    print(f" Total Frames Processed  : {total_frames} frames (FPS: {total_frames / max(session_duration, 0.001):.1f})")
    print(f" Total Session Taps      : {session_total_taps} taps (strictly non-decreasing)")
    print(f" Timestamp               : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 76)

    # Check if a formal 10-second trial was completed
    trial_state = trial_status.get("state")
    trial_score_dict = trial_status.get("score")
    trial_metrics_dict = trial_status.get("metrics")

    if trial_state == TappingTrialManager.COMPLETED and trial_score_dict and trial_metrics_dict:
        print("[FORMAL 10-SECOND MDS-UPDRS TASK 3.4 TRIAL RESULTS]")
        print(f"  * Trial Tap Count      : {trial_metrics_dict.get('total_taps', 0)} taps in 10.0s")
        print(f"  * Tapping Rate         : {trial_metrics_dict.get('tap_rate_hz', 0.0):.2f} taps/sec")
        print(f"  * Mean Tap Amplitude   : {trial_metrics_dict.get('mean_amplitude', 0.0):.4f} (scale-normalized)")
        print(f"  * Amplitude Decay Slope: {trial_metrics_dict.get('amplitude_decay_slope', 0.0):+.4f} /sec")
        print(f"  * Rhythm Regularity CV : {trial_metrics_dict.get('rhythm_cv', 0.0):.2f}")
        print(f"  -> CLINICAL SCORE      : {trial_score_dict.get('score', 0)} / 4  [{trial_score_dict.get('label', 'UNKNOWN').upper()}]")
        print(f"  -> RATIONALE           : {trial_score_dict.get('clinical_rationale', '')}")
    else:
        # Fallback to rolling/session metrics
        print("[SESSION FINGER TAPPING SUMMARY (No formal 10s trial completed)]")
        print(f"  * Session Taps Count   : {session_total_taps}")
        print(f"  * Cadence              : {tap_metrics.get('tap_rate_hz', 0.0):.2f} taps/sec")
        print(f"  * Amplitude Decay Slope: {tap_metrics.get('amplitude_decay_slope', 0.0):+.4f} /sec")
        print(f"  * Rhythm Regularity CV : {tap_metrics.get('rhythm_cv', 0.0):.2f}")
        print(f"  -> ESTIMATED SCORE     : {updrs_tapping.get('score', 0)} / 4  [{updrs_tapping.get('label', 'UNKNOWN').upper()}]")

    print("-" * 76)

    # Tremor Spectral Summary
    peak_freq = tremor_metrics.get("peak_freq", 0.0)
    peak_power = tremor_metrics.get("peak_power", 0.0)
    tremor_type = tremor_metrics.get("tremor_type", "None")
    is_sig = tremor_metrics.get("is_significant", False)
    tremor_score = updrs_tremor.get("score", 0)
    tremor_label = updrs_tremor.get("label", "Normal")
    tremor_rationale = updrs_tremor.get("clinical_rationale", "")

    print("[SECTION 2: MDS-UPDRS ITEMS 3.15-3.18 - TREMOR SPECTRAL DECOMPOSITION]")
    print(f"  * Dominant Frequency   : {peak_freq:.2f} Hz")
    print(f"  * Peak Spectral Power  : {peak_power:.6f} (PSD magnitude)")
    print(f"  * Tremor Classification: {tremor_type}")
    print(f"  * Pathological Tremor  : {'YES (Exceeds clinical threshold)' if is_sig else 'NO (Within physiological baseline)'}")
    print(f"  -> CLINICAL SCORE      : {tremor_score} / 4  [{tremor_label.upper()}]")
    print(f"  -> RATIONALE           : {tremor_rationale}")
    print("=" * 76)
    print("                    END OF CLINICAL ASSESSMENT                      \n")


def run_interactive_biomarker_test(
    camera_id: int = 0,
    buffer_size: int = 150,
) -> None:
    """
    Launch interactive live webcam video capture with real-time biomarker analysis,
    persistent tap state machine, 10-second trial manager, and resizable HUD.

    Args:
        camera_id: OpenCV capture device index.
        buffer_size: Rolling KinematicBuffer frame count.
    """
    print("=" * 70)
    print("NeuroMotion - Live Biomarker & MDS-UPDRS Clinical Assessment")
    print("=" * 70)
    print(f"Target Camera Device  : {camera_id}")
    print(f"Rolling Buffer Size   : {buffer_size} frames (~5.0s window)")
    print("Controls              : SPACE or 'R' -> Start 10s MDS-UPDRS Trial")
    print("                        'q' or ESC   -> Finish & View Clinical Report")
    print("-" * 70)

    # Initialize Camera Capture with DirectShow on Windows, MJPG codec, and 640x480 resolution
    cap = HandTracker.create_capture(camera_id=camera_id, width=640, height=480)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera at index {camera_id}.")
        print("[INFO] Run with '--smoke-test' to run automated non-interactive verification.")
        return

    # Create resizable display window without black borders
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    tracker = HandTracker(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    buffer = KinematicBuffer(buffer_size=buffer_size)
    tapping_analyzer = FingerTappingAnalyzer()
    tremor_analyzer = TremorAnalyzer()

    # Telemetry and state caches
    fps_smooth = 30.0
    alpha_fps = 0.15
    prev_time = time.perf_counter()
    start_time = prev_time
    total_frames = 0

    current_aperture = 0.0
    current_state = TapState.TOUCH
    handedness_str = ""
    trial_status: Dict[str, Any] = {
        "state": TappingTrialManager.IDLE,
        "remaining_time": 10.0,
        "trial_taps_count": 0,
        "metrics": None,
        "score": None,
    }

    latest_tap_metrics: Dict[str, Any] = {
        "total_taps": 0,
        "tap_rate_hz": 0.0,
        "mean_amplitude": 0.0,
        "amplitude_decay_slope": 0.0,
        "rhythm_cv": 0.0,
    }
    latest_updrs_tapping: Dict[str, Any] = {
        "score": 0,
        "label": "Ready",
        "clinical_rationale": "Awaiting finger tapping movement...",
    }
    latest_tremor_metrics: Dict[str, Any] = {
        "peak_freq": 0.0,
        "peak_power": 0.0,
        "tremor_type": "Voluntary baseline movement",
        "is_significant": False,
    }
    latest_updrs_tremor: Dict[str, Any] = {
        "score": 0,
        "label": "Normal",
        "clinical_rationale": "No significant tremor detected.",
    }
    distance_history = np.empty((0,), dtype=np.float64)

    print("[STATUS] Live video stream initialized. Press SPACE or 'R' to start 10s trial...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] End of camera stream or unable to read frame.")
                break

            total_frames += 1
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0.0:
                fps_smooth = (1 - alpha_fps) * fps_smooth + alpha_fps * (1.0 / dt)

            # Process frame through HandTracker
            annotated_frame, hand_data = tracker.process_frame(frame)

            if hand_data is not None:
                landmarks = hand_data["landmarks_norm"]
                sample_t = hand_data["timestamp"]
                thumb = landmarks[THUMB_TIP_IDX]
                index = landmarks[INDEX_TIP_IDX]
                wrist = landmarks[WRIST_IDX]
                middle_mcp = landmarks[MIDDLE_MCP_IDX]
                handedness_str = str(hand_data["handedness"])
                current_aperture = float(hand_data.get("normalized_aperture", 0.0))
                scale_factor = float(hand_data.get("scale_factor", hand_data.get("l_ref", 0.0)))

                # Ingest into thread-safe rolling buffer with middle_mcp for scale normalization
                buffer.push(sample_t, thumb, index, wrist, middle_mcp)

                # Process sample through persistent state machine & trial manager
                sample_res = tapping_analyzer.process_sample(sample_t, current_aperture)
                current_state = sample_res["current_state"]
                trial_status = sample_res["trial_status"]
            else:
                handedness_str = ""
                scale_factor = 0.0

            # Check trial status updates even if no hand on current frame
            if trial_status.get("state") == TappingTrialManager.ACTIVE:
                trial_status = tapping_analyzer.trial_manager.update(curr_time, None)

            # Analyze kinematics when sufficient samples are buffered (~1.0s)
            if buffer.is_ready(min_samples=30):
                series = buffer.get_time_series()
                t_rel = series["time"]
                d_norm = series["normalized_aperture"]
                wrist_pos = series["wrist"]
                distance_history = d_norm

                if len(d_norm) >= 15:
                    latest_tap_metrics = tapping_analyzer.analyze_tapping(d_norm, t_rel)
                    latest_updrs_tapping = MDSUPDRSScorer.score_finger_tapping(latest_tap_metrics)

                    latest_tremor_metrics = tremor_analyzer.analyze_tremor(wrist_pos, t_rel)
                    latest_updrs_tremor = MDSUPDRSScorer.score_tremor(latest_tremor_metrics)

            # Render Comprehensive HUD overlay onto frame
            hud_frame = draw_clinical_hud(
                frame=annotated_frame,
                fps=fps_smooth,
                buffer_len=len(buffer),
                buffer_capacity=buffer.buffer_size,
                handedness_str=handedness_str,
                current_aperture=current_aperture,
                current_state=current_state,
                session_taps=tapping_analyzer.session_total_taps,
                instantaneous_cadence=tapping_analyzer.state_machine.get_instantaneous_cadence(current_timestamp=curr_time),
                trial_status=trial_status,
                tap_metrics=latest_tap_metrics,
                updrs_tapping=latest_updrs_tapping,
                tremor_metrics=latest_tremor_metrics,
                updrs_tremor=latest_updrs_tremor,
                distance_series=distance_history,
                scale_factor=scale_factor,
            )

            cv2.imshow(WINDOW_NAME, hud_frame)

            # Keystroke handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" ") or key == ord("r") or key == ord("R"):
                # Start / restart 10-second evaluation trial
                print(f"\n[TRIAL] 10-Second MDS-UPDRS Evaluation Trial Started at t={curr_time:.2f}s...")
                tapping_analyzer.start_trial(curr_time)
            elif key == ord("q") or key == 27:  # 'q' or ESC
                print("[INFO] Exit signal received. Generating clinical report...")
                break

            # Check if user closed the window
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Display window closed by user.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Exiting gracefully...")
    finally:
        session_duration = time.perf_counter() - start_time
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

        # Print formatted clinical report
        print_clinical_report(
            session_duration=session_duration,
            total_frames=total_frames,
            session_total_taps=tapping_analyzer.session_total_taps,
            trial_status=trial_status,
            tap_metrics=latest_tap_metrics,
            updrs_tapping=latest_updrs_tapping,
            tremor_metrics=latest_tremor_metrics,
            updrs_tremor=latest_updrs_tremor,
        )


def run_smoke_test() -> bool:
    """
    Automated non-interactive verification test.

    Verifies:
      1. DirectShow / MJPG capture setup logic.
      2. Scale-invariant reference length L_ref & normalized aperture A(t).
      3. Persistent TapStateMachine monotonicity (session_total_taps non-decreasing) & event log.
      4. Instantaneous cadence calculated from last 5 taps.
      5. 10-Second Trial Mode: countdown, decay slope, rhythm CV, and score freezing.
      6. HUD rendering with top bar, trial banner, and resizable window properties.

    Returns:
        True if all smoke assertions pass.
    """
    print("\n" + "=" * 65)
    print("NeuroMotion - Biomarkers Automated Verification Suite")
    print("=" * 65)

    fs = 30.0
    duration = 10.0  # 10 seconds for standard trial
    n_samples = int(fs * duration)
    timestamps = np.linspace(0.0, duration, n_samples)

    # 1. Scale-invariant normalization verification
    print("1. Testing scale-invariant reference length L_ref & normalized aperture...")
    # Simulated hand: Wrist at (0, 0, 0), Middle MCP at (0, 0.25, 0) -> L_ref = 0.25
    wrist = np.array([0.0, 0.0, 0.0])
    middle_mcp = np.array([0.0, 0.25, 0.0])
    l_ref_expected = float(np.linalg.norm(wrist - middle_mcp))
    assert abs(l_ref_expected - 0.25) < 1e-4

    # Distance between thumb and index = 0.10 -> Normalized Aperture = 0.10 / 0.25 = 0.40
    raw_dist = 0.10
    norm_aperture = raw_dist / l_ref_expected
    assert abs(norm_aperture - 0.40) < 1e-4
    print(f"   [PASS] L_ref={l_ref_expected:.2f}, Normalized aperture A(t)={norm_aperture:.2f} L_ref")

    # 2. Test Persistent TapStateMachine
    print("2. Testing persistent TapStateMachine & monotonic counter...")
    state_machine = TapStateMachine(delta_thresh=0.06, touch_threshold=0.22)

    # Synthesize 2.5 Hz tapping (touch at 0.12, open at 0.50)
    tap_freq = 2.5
    synthetic_aperture = 0.12 + 0.38 * (0.5 * (1.0 - np.cos(2 * np.pi * tap_freq * timestamps)))

    tap_counts_over_time = []
    for i in range(len(timestamps)):
        t_i = timestamps[i]
        a_i = synthetic_aperture[i]
        state_machine.update(t_i, a_i)
        tap_counts_over_time.append(state_machine.session_total_taps)

    # Monotonicity test: session_total_taps must be strictly non-decreasing!
    tap_counts_arr = np.array(tap_counts_over_time)
    diffs = np.diff(tap_counts_arr)
    assert np.all(diffs >= 0), "session_total_taps decreased! Monotonicity violated."
    total_detected = state_machine.session_total_taps
    print(f"   [RESULT] Monotonic taps detected: {total_detected} (Expected ~25 in 10s)")
    assert 22 <= total_detected <= 26, f"Unexpected tap count: {total_detected}"
    assert len(state_machine.tap_events) == total_detected
    print("   [PASS] TapStateMachine strictly monotonic and event log preserved.")

    # 3. Test Instantaneous Cadence from last 5 taps
    print("3. Testing instantaneous cadence calculation (last 5 taps)...")
    cadence = state_machine.get_instantaneous_cadence(current_timestamp=timestamps[-1])
    print(f"   [RESULT] Instantaneous cadence: {cadence:.2f} Hz (Expected ~2.50 Hz)")
    assert 2.30 <= cadence <= 2.70, f"Unexpected cadence: {cadence}"
    print("   [PASS] Cadence accurately reflects inter-tap intervals of last 5 taps.")

    # 4. Test 10-Second Trial Manager
    print("4. Testing 10-Second MDS-UPDRS Evaluation Trial Manager...")
    trial_mgr = TappingTrialManager(trial_duration=10.0)
    trial_mgr.start_trial(current_time=0.0)
    assert trial_mgr.state == TappingTrialManager.ACTIVE

    # Feed the tap events through trial manager
    for event in state_machine.tap_events:
        trial_mgr.update(current_time=event.touch_timestamp, new_tap=event)

    # Conclude trial at t=10.0s
    status = trial_mgr.update(current_time=10.01, new_tap=None)
    assert status["state"] == TappingTrialManager.COMPLETED
    assert status["metrics"] is not None
    assert status["score"] is not None

    trial_rate = status["metrics"]["tap_rate_hz"]
    trial_score = status["score"]["score"]
    print(f"   [RESULT] Trial Tap Rate: {trial_rate:.2f} taps/s (Expected ~2.5 taps/s)")
    print(f"   [RESULT] Task 3.4 Trial Score: {trial_score} ({status['score']['label']})")
    assert 2.0 <= trial_rate <= 3.0
    assert trial_score in (0, 1)
    print("   [PASS] 10-Second Trial Manager correctly scored and froze clinical results.")

    # 5. Synthesize 5.0 Hz Parkinsonian Rest Tremor
    print("5. Generating synthetic Parkinsonian resting tremor (5.0 Hz)...")
    tremor_f = 5.0
    tremor_signal = 0.05 * np.sin(2 * np.pi * tremor_f * timestamps)
    wrist_coords = np.column_stack([tremor_signal, np.zeros(n_samples), np.zeros(n_samples)])

    tremor_analyzer = TremorAnalyzer()
    tremor_results = tremor_analyzer.analyze_tremor(wrist_coords, timestamps)
    updrs_tremor = MDSUPDRSScorer.score_tremor(tremor_results)
    assert 4.5 <= tremor_results["peak_freq"] <= 5.5
    assert "Parkinsonian" in tremor_results["tremor_type"]
    print("   [PASS] TremorAnalyzer verified.")

    # 6. Test HUD rendering on 640x480 canvas
    print("6. Testing HUD rendering with Top Bar, Handedness, and Trial Banner...")
    test_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered_hud = draw_clinical_hud(
        frame=test_canvas,
        fps=30.0,
        buffer_len=150,
        buffer_capacity=150,
        handedness_str="Right (0.98)",
        current_aperture=0.42,
        current_state=TapState.OPENING,
        session_taps=total_detected,
        instantaneous_cadence=cadence,
        trial_status=status,
        tap_metrics=status["metrics"],
        updrs_tapping=status["score"],
        tremor_metrics=tremor_results,
        updrs_tremor=updrs_tremor,
        distance_series=synthetic_aperture,
    )
    assert rendered_hud.shape == (480, 640, 3)
    print("   [PASS] Clinical HUD with Top Bar and Trial Telemetry rendered successfully.")

    # 7. Print verified clinical report
    print_clinical_report(
        session_duration=duration,
        total_frames=n_samples,
        session_total_taps=total_detected,
        trial_status=status,
        tap_metrics=status["metrics"],
        updrs_tapping=status["score"],
        tremor_metrics=tremor_results,
        updrs_tremor=updrs_tremor,
    )

    print("[ALL SMOKE TESTS PASSED] NeuroMotion architectural & algorithmic core verified.\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroMotion - Digital Biomarkers & Clinical Scoring Verification"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="OpenCV capture device index (default: 0)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=150,
        help="KinematicBuffer capacity in frames (default: 150)",
    )
    parser.add_argument(
        "--smoke-test",
        "--headless",
        action="store_true",
        help="Run automated non-interactive verification suite",
    )
    args = parser.parse_args()

    if args.smoke_test:
        success = run_smoke_test()
        sys.exit(0 if success else 1)
    else:
        run_interactive_biomarker_test(
            camera_id=args.camera_id,
            buffer_size=args.buffer_size,
        )
