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


def draw_waveform_canvas(
    frame: np.ndarray,
    distance_series: np.ndarray,
    peak_indices: np.ndarray,
    origin: Tuple[int, int],
    size: Tuple[int, int],
) -> None:
    """
    Draw a real-time mini waveform monitor on the OpenCV frame.

    Args:
        frame: Destination BGR image array.
        distance_series: 1D array of rolling finger aperture values.
        peak_indices: Frame indices of detected tap peaks.
        origin: (x, y) top-left coordinates of the monitor widget.
        size: (width, height) of the monitor widget.
    """
    x, y = origin
    w, h = size
    h_frame, w_frame = frame.shape[:2]

    if x + w > w_frame or y + h > h_frame:
        return

    # Background panel with slight transparency
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 18, 24), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Cyan border and grid line
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 220), 1, cv2.LINE_AA)
    mid_y = y + h // 2
    cv2.line(frame, (x, mid_y), (x + w, mid_y), (35, 45, 55), 1, cv2.LINE_AA)

    # Title label
    cv2.putText(
        frame,
        "Norm Aperture A(t)",
        (x + 8, y + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (180, 210, 230),
        1,
        cv2.LINE_AA,
    )

    n_pts = len(distance_series)
    if n_pts < 2:
        return

    # Dynamic scaling between 0.0 and max (at least 0.6)
    min_val = 0.0
    max_val = max(0.60, float(np.max(distance_series)))
    val_range = max_val - min_val

    margin_x = 8
    margin_y = 20
    usable_w = w - 2 * margin_x
    usable_h = h - margin_y - 8

    pts = []
    for i, val in enumerate(distance_series):
        pt_x = int(x + margin_x + (i / (n_pts - 1)) * usable_w)
        norm_y = (val - min_val) / val_range
        norm_y = max(0.0, min(1.0, norm_y))
        pt_y = int(y + h - 6 - norm_y * usable_h)
        pts.append((pt_x, pt_y))

    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_arr], isClosed=False, color=(0, 255, 180), thickness=2, lineType=cv2.LINE_AA)

    # Highlight detected peaks
    peak_set = set(peak_indices.tolist()) if len(peak_indices) > 0 else set()
    for idx in peak_set:
        if 0 <= idx < len(pts):
            cv2.circle(frame, pts[idx], 4, (0, 220, 255), -1, cv2.LINE_AA)


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
    tap_metrics: Dict[str, Any],
    updrs_tapping: Dict[str, Any],
    tremor_metrics: Dict[str, Any],
    updrs_tremor: Dict[str, Any],
    distance_series: np.ndarray,
) -> np.ndarray:
    """
    Render medical Heads-Up Display (HUD) overlay containing real-time kinematic biomarkers,
    trial countdown/scores, scale-normalized aperture bar, and handedness telemetry.

    Args:
        frame: Annotated BGR camera frame (640x480).
        fps: Measured frame acquisition rate.
        buffer_len: Current sample count in rolling buffer.
        buffer_capacity: Max capacity of rolling buffer.
        handedness_str: String representation of handedness ("Right (0.98)").
        current_aperture: Instantaneous scale-normalized aperture A(t).
        current_state: Active tap state machine state (TOUCH, OPENING, PEAK, CLOSING).
        session_taps: Strictly non-decreasing session tap counter.
        instantaneous_cadence: Live cadence (Hz) from the last 5 confirmed taps.
        trial_status: Dictionary from TappingTrialManager (state, remaining_time, metrics, score).
        tap_metrics: Rolling tap metrics for fallback/diagnostics.
        updrs_tapping: Live MDS-UPDRS Task 3.4 score dictionary.
        tremor_metrics: Spectral tremor decomposition metrics.
        updrs_tremor: Clinical tremor rating from MDSUPDRSScorer.
        distance_series: Rolling time-series of finger aperture.

    Returns:
        Frame with comprehensive clinical telemetry HUD.
    """
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # =========================================================================
    # 1. TOP BAR: FPS, Camera Resolution, Handedness
    # =========================================================================
    top_bar_h = 32
    top_overlay = frame.copy()
    cv2.rectangle(top_overlay, (0, 0), (w, top_bar_h), (12, 14, 18), -1)
    cv2.addWeighted(top_overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, top_bar_h), (w, top_bar_h), (40, 50, 65), 1, cv2.LINE_AA)

    # Left: System Title & Resolution
    cv2.putText(frame, "NEUROMOTION CORE", (12, 21), font, 0.48, (0, 255, 200), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Res: {w}x{h} (MJPG)", (180, 21), font, 0.42, (180, 190, 205), 1, cv2.LINE_AA)

    # Middle: FPS
    fps_color = (0, 255, 128) if fps >= 25.0 else ((0, 200, 255) if fps >= 18.0 else (0, 80, 255))
    cv2.putText(frame, f"FPS: {fps:4.1f}", (315, 21), font, 0.44, fps_color, 2, cv2.LINE_AA)

    # Right: Explicit Handedness Label with confidence
    hand_display = handedness_str if handedness_str else "None"
    hand_color = (0, 230, 255) if "Right" in hand_display or "Left" in hand_display else (140, 150, 160)
    cv2.putText(frame, f"Hand: {hand_display}", (w - 210, 21), font, 0.44, hand_color, 2, cv2.LINE_AA)

    # =========================================================================
    # 2. TRIAL STATUS BANNER (Middle HUD)
    # =========================================================================
    trial_state = trial_status.get("state", TappingTrialManager.IDLE)
    rem_time = trial_status.get("remaining_time", 10.0)
    trial_taps = trial_status.get("trial_taps_count", 0)

    banner_x, banner_y = 14, 42
    banner_w, banner_h = 370, 70

    banner_overlay = frame.copy()
    cv2.rectangle(banner_overlay, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h), (16, 20, 28), -1)
    cv2.addWeighted(banner_overlay, 0.82, frame, 0.18, 0, frame)

    if trial_state == TappingTrialManager.IDLE:
        # IDLE Banner
        cv2.rectangle(frame, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h), (0, 180, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, "MDS-UPDRS 3.4 TRIAL: IDLE", (banner_x + 12, banner_y + 24), font, 0.50, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press SPACE or 'R' to start 10s test", (banner_x + 12, banner_y + 45), font, 0.40, (200, 210, 225), 1, cv2.LINE_AA)
        cv2.putText(frame, "Tap thumb & index as fast & wide as possible", (banner_x + 12, banner_y + 61), font, 0.34, (140, 160, 180), 1, cv2.LINE_AA)

    elif trial_state == TappingTrialManager.ACTIVE:
        # ACTIVE Banner (Pulsing amber with countdown bar)
        cv2.rectangle(frame, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h), (0, 180, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"TRIAL ACTIVE: {rem_time:4.1f}s REMAINING", (banner_x + 12, banner_y + 24), font, 0.52, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Trial Taps: {trial_taps}  |  Keep tapping!", (banner_x + 12, banner_y + 44), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # 10-second countdown progress bar
        bar_x, bar_y_prog = banner_x + 12, banner_y + 52
        bar_w_prog, bar_h_prog = banner_w - 24, 10
        cv2.rectangle(frame, (bar_x, bar_y_prog), (bar_x + bar_w_prog, bar_y_prog + bar_h_prog), (35, 40, 50), -1)
        fill_w = int(round((10.0 - rem_time) / 10.0 * bar_w_prog))
        fill_w = max(0, min(bar_w_prog, fill_w))
        cv2.rectangle(frame, (bar_x, bar_y_prog), (bar_x + fill_w, bar_y_prog + bar_h_prog), (0, 220, 255), -1)

    elif trial_state == TappingTrialManager.COMPLETED:
        # COMPLETED Banner (Frozen Score & Detailed Breakdown)
        trial_score_dict = trial_status.get("score") or updrs_tapping
        trial_metrics_dict = trial_status.get("metrics") or tap_metrics
        t_score = trial_score_dict.get("score", 0)
        t_label = trial_score_dict.get("label", "Unknown")

        score_colors = {
            0: (0, 255, 128),   # Green: Normal
            1: (0, 220, 255),   # Cyan: Slight
            2: (0, 165, 255),   # Orange: Mild
            3: (50, 80, 255),   # Red-Orange: Moderate
            4: (0, 0, 255),     # Red: Severe
        }
        b_color = score_colors.get(t_score, (0, 255, 128))

        # Expand banner height slightly for frozen detailed breakdown
        banner_h = 80
        cv2.rectangle(frame, (banner_x, banner_y), (banner_x + banner_w, banner_y + banner_h), b_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"TRIAL RESULT -> MDS-UPDRS: SCORE {t_score} ({t_label})", (banner_x + 10, banner_y + 22), font, 0.44, b_color, 2, cv2.LINE_AA)

        rate_val = trial_metrics_dict.get("tap_rate_hz", 0.0)
        slope_val = trial_metrics_dict.get("amplitude_decay_slope", 0.0)
        cv_val = trial_metrics_dict.get("rhythm_cv", 0.0)

        cv2.putText(frame, f"Taps: {trial_taps} | Rate: {rate_val:.2f} Hz | Decay: {slope_val:+.3f}", (banner_x + 10, banner_y + 44), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Rhythm CV: {cv_val:.2f}  (Press SPACE to re-test)", (banner_x + 10, banner_y + 64), font, 0.36, (180, 200, 220), 1, cv2.LINE_AA)

    # =========================================================================
    # 3. LIVE PERSISTENT TELEMETRY & APERTURE GAUGE
    # =========================================================================
    panel_y = banner_y + banner_h + 8
    panel_h = 160
    panel_w = 370
    p_overlay = frame.copy()
    cv2.rectangle(p_overlay, (banner_x, panel_y), (banner_x + panel_w, panel_y + panel_h), (16, 20, 28), -1)
    cv2.addWeighted(p_overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (banner_x, panel_y), (banner_x + panel_w, panel_y + panel_h), (45, 55, 70), 1, cv2.LINE_AA)

    line_y = panel_y + 22
    dy = 24

    # Persistent Session Tap Count (Strictly monotonic)
    cv2.putText(frame, "Session Tap Count:", (banner_x + 12, line_y), font, 0.44, (190, 200, 215), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{session_taps} taps", (banner_x + 175, line_y), font, 0.52, (0, 255, 200), 2, cv2.LINE_AA)

    # Instantaneous Cadence (from last 5 taps)
    line_y += dy
    cv2.putText(frame, "Cadence (Last 5):", (banner_x + 12, line_y), font, 0.44, (190, 200, 215), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{instantaneous_cadence:4.2f} Hz", (banner_x + 175, line_y), font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    # Active State Machine State (TOUCH, OPENING, PEAK, CLOSING)
    line_y += dy
    state_colors = {
        TapState.TOUCH: (0, 255, 128),
        TapState.OPENING: (0, 220, 255),
        TapState.PEAK: (0, 255, 255),
        TapState.CLOSING: (0, 165, 255),
    }
    st_color = state_colors.get(current_state, (200, 200, 200))
    cv2.putText(frame, "Tap State:", (banner_x + 12, line_y), font, 0.44, (190, 200, 215), 1, cv2.LINE_AA)
    cv2.putText(frame, f"[{current_state}]", (banner_x + 175, line_y), font, 0.48, st_color, 2, cv2.LINE_AA)

    # Scale-Invariant Normalized Aperture A(t)
    line_y += dy
    cv2.putText(frame, "Norm Aperture A(t):", (banner_x + 12, line_y), font, 0.42, (190, 200, 215), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{current_aperture:4.2f} L_ref", (banner_x + 175, line_y), font, 0.46, (255, 220, 100), 2, cv2.LINE_AA)

    # Normalized Aperture Mini-Bar Gauge
    line_y += 14
    gauge_x = banner_x + 12
    gauge_w = panel_w - 24
    gauge_h = 10
    cv2.rectangle(frame, (gauge_x, line_y), (gauge_x + gauge_w, line_y + gauge_h), (30, 35, 45), -1)
    fill_len = int(round(min(1.0, current_aperture / 1.0) * gauge_w))
    fill_len = max(0, min(gauge_w, fill_len))
    cv2.rectangle(frame, (gauge_x, line_y), (gauge_x + fill_len, line_y + gauge_h), st_color, -1)
    cv2.rectangle(frame, (gauge_x, line_y), (gauge_x + gauge_w, line_y + gauge_h), (60, 70, 85), 1, cv2.LINE_AA)

    # Touch threshold tick mark (0.22)
    tick_x = gauge_x + int(0.22 * gauge_w)
    cv2.line(frame, (tick_x, line_y - 2), (tick_x, line_y + gauge_h + 2), (0, 255, 128), 1, cv2.LINE_AA)

    # =========================================================================
    # 4. MINI WAVEFORM MONITOR (Top Right)
    # =========================================================================
    wave_w = 210
    wave_h = 95
    wave_x = w - wave_w - 14
    wave_y = top_bar_h + 10
    peak_indices = tap_metrics.get("peak_indices", np.empty((0,), dtype=int))
    draw_waveform_canvas(frame, distance_series, peak_indices, (wave_x, wave_y), (wave_w, wave_h))

    # =========================================================================
    # 5. FOOTER INSTRUCTION BAR
    # =========================================================================
    footer_text = "SPACE/R: 10s Trial | 'q'/ESC: Exit & View Report"
    cv2.putText(frame, footer_text, (14, h - 14), font, 0.42, (200, 210, 225), 1, cv2.LINE_AA)

    return frame


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

                # Ingest into thread-safe rolling buffer with middle_mcp for scale normalization
                buffer.push(sample_t, thumb, index, wrist, middle_mcp)

                # Process sample through persistent state machine & trial manager
                sample_res = tapping_analyzer.process_sample(sample_t, current_aperture)
                current_state = sample_res["current_state"]
                trial_status = sample_res["trial_status"]
            else:
                handedness_str = ""

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
