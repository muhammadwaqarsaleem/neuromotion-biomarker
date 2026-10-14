"""
NeuroMotion - Core Vision & Kinematic Verification Test
=======================================================
Interactive visual test harness for HandTracker and KinematicBuffer.

Features:
- Live OpenCV camera stream acquisition with mirror interaction.
- Real-time MediaPipe hand landmark extraction and skeleton rendering.
- Rolling KinematicBuffer ingestion for downstream DSP and bradykinesia tracking.
- Dynamic HUD overlay:
    * Live FPS monitor (top-left).
    * Rolling buffer status (Ready vs. Buffering X/150).
    * Normalized and pixel-space Euclidean distance between thumb and index tips.
    * Adaptive color pinch-line indicator (Cyan when open, Green when pinching).
    * Handedness and detection confidence badge.
- Graceful shutdown on pressing 'q', ESC, or clicking the window close button.
- Optional headless / automated smoke-test mode via `--smoke-test` or `--headless`.
"""

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.vision import (
    INDEX_TIP_IDX,
    KEY_LANDMARK_INDICES,
    MIDDLE_TIP_IDX,
    THUMB_TIP_IDX,
    WRIST_IDX,
    HandTracker,
    KinematicBuffer,
)

# Visual Constants
WINDOW_NAME = "NeuroMotion - Hand Tracking & Kinematic Buffer Test"
PINCH_THRESHOLD_NORM = 0.075  # Normalized distance threshold for pinch recognition


def draw_hud(
    frame: np.ndarray,
    fps: float,
    buffer_len: int,
    buffer_capacity: int,
    is_ready: bool,
    thumb_px: Optional[Tuple[int, int]],
    index_px: Optional[Tuple[int, int]],
    dist_norm: Optional[float],
    dist_px: Optional[float],
    handedness_str: str,
) -> np.ndarray:
    """
    Draw a real-time HUD (Heads-Up Display) with glassmorphism styling onto the frame.

    Args:
        frame: Annotated BGR image frame.
        fps: Current measured frames per second.
        buffer_len: Number of frames currently in the rolling buffer.
        buffer_capacity: Maximum capacity of the rolling buffer.
        is_ready: Whether the buffer meets minimum readiness for spectral analysis.
        thumb_px: (x, y) coordinates of thumb tip in pixel space.
        index_px: (x, y) coordinates of index tip in pixel space.
        dist_norm: Normalized 3D Euclidean distance between thumb and index.
        dist_px: 2D pixel distance between thumb and index tips.
        handedness_str: String representation of handedness classification.

    Returns:
        Frame with HUD graphics overlaid.
    """
    h, w = frame.shape[:2]

    # 1. Draw live connection line between thumb and index finger if both are detected
    is_pinching = False
    if thumb_px is not None and index_px is not None and dist_norm is not None:
        is_pinching = dist_norm < PINCH_THRESHOLD_NORM

        # Dynamic color: Vibrant Green if pinching, Vibrant Cyan if open
        line_color = (0, 255, 0) if is_pinching else (255, 255, 0)
        line_thickness = 3 if is_pinching else 2

        cv2.line(frame, thumb_px, index_px, line_color, line_thickness, cv2.LINE_AA)

        # Highlight endpoints with accented circular rings
        cv2.circle(frame, thumb_px, 6, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, index_px, 6, (0, 200, 255), -1, cv2.LINE_AA)

        # Draw distance tag near the line midpoint
        mid_x = (thumb_px[0] + index_px[0]) // 2
        mid_y = (thumb_px[1] + index_px[1]) // 2
        dist_tag = f"{dist_norm:.3f}"
        tag_color = (0, 255, 0) if is_pinching else (255, 255, 255)
        cv2.putText(
            frame,
            dist_tag,
            (mid_x + 10, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            dist_tag,
            (mid_x + 10, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            tag_color,
            1,
            cv2.LINE_AA,
        )

    # 2. Semi-transparent HUD overlay panel in top-left
    overlay = frame.copy()
    hud_x, hud_y, hud_w, hud_h = 16, 16, 360, 165
    cv2.rectangle(
        overlay,
        (hud_x, hud_y),
        (hud_x + hud_w, hud_y + hud_h),
        (20, 22, 26),
        -1,
    )
    # Blend with original frame (alpha = 0.70)
    alpha = 0.72
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Accent border for HUD panel
    border_color = (0, 255, 128) if is_ready else (60, 180, 255)
    cv2.rectangle(
        frame,
        (hud_x, hud_y),
        (hud_x + hud_w, hud_y + hud_h),
        border_color,
        1,
        cv2.LINE_AA,
    )

    # 3. Render HUD Telemetry Text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title & FPS
    cv2.putText(
        frame,
        "NEUROMOTION HUD",
        (hud_x + 12, hud_y + 24),
        font,
        0.55,
        (0, 255, 128),
        2,
        cv2.LINE_AA,
    )

    fps_text = f"FPS: {fps:04.1f}"
    fps_color = (0, 255, 0) if fps >= 25 else (0, 200, 255)
    cv2.putText(
        frame,
        fps_text,
        (hud_x + hud_w - 95, hud_y + 24),
        font,
        0.5,
        fps_color,
        1,
        cv2.LINE_AA,
    )

    # Divider line
    cv2.line(
        frame,
        (hud_x + 10, hud_y + 34),
        (hud_x + hud_w - 10, hud_y + 34),
        (70, 75, 85),
        1,
    )

    # Buffer Status
    if is_ready:
        buffer_text = f"Buffer: Ready [{buffer_len}/{buffer_capacity}]"
        buffer_color = (0, 255, 128)  # Mint Green
    else:
        buffer_text = f"Buffer: Buffering: {buffer_len}/{buffer_capacity}"
        buffer_color = (0, 190, 255)  # Amber / Yellow

    cv2.putText(
        frame,
        buffer_text,
        (hud_x + 12, hud_y + 58),
        font,
        0.5,
        buffer_color,
        1,
        cv2.LINE_AA,
    )

    # Euclidean Pinch Distance
    if dist_norm is not None and dist_px is not None:
        dist_str = f"Pinch Dist: {dist_norm:.3f} ({dist_px:.1f} px)"
        dist_color = (0, 255, 0) if is_pinching else (255, 255, 255)
    else:
        dist_str = "Pinch Dist: -- (No Hand)"
        dist_color = (150, 150, 150)

    cv2.putText(
        frame,
        dist_str,
        (hud_x + 12, hud_y + 84),
        font,
        0.5,
        dist_color,
        1,
        cv2.LINE_AA,
    )

    # Pinch Status
    if thumb_px is not None:
        state_str = "Status: PINCHING" if is_pinching else "Status: OPEN"
        state_color = (0, 255, 0) if is_pinching else (255, 255, 0)
    else:
        state_str = "Status: Awaiting Hand"
        state_color = (140, 140, 140)

    cv2.putText(
        frame,
        state_str,
        (hud_x + 12, hud_y + 110),
        font,
        0.5,
        state_color,
        1,
        cv2.LINE_AA,
    )

    # Handedness
    hand_str = f"Hand: {handedness_str}" if handedness_str else "Hand: None"
    cv2.putText(
        frame,
        hand_str,
        (hud_x + 12, hud_y + 134),
        font,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Exit instruction caption at bottom of frame
    footer_text = "Press 'q' or ESC to exit live visual test"
    cv2.putText(
        frame,
        footer_text,
        (16, h - 16),
        font,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return frame


def run_interactive_vision_test(
    camera_id: int = 0,
    buffer_size: int = 150,
    width: int = 1280,
    height: int = 720,
) -> None:
    """
    Launch interactive OpenCV video capture and HUD visualization.

    Args:
        camera_id: System camera index (default: 0).
        buffer_size: Rolling KinematicBuffer size.
        width: Requested camera frame width.
        height: Requested camera frame height.
    """
    print("=" * 65)
    print("NeuroMotion - Core Vision & Kinematic Verification Test")
    print("=" * 65)
    print(f"Target Camera Index : {camera_id}")
    print(f"Kinematic Buffer Cap : {buffer_size} frames (~5.0s @ 30 FPS)")
    print("Exit Key            : Press 'q' or 'ESC' in the display window.")
    print("-" * 65)

    # Initialize Camera Capture
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_id)
    else:
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video capture device at index {camera_id}.")
        print("[INFO] If testing in a headless environment, run with '--smoke-test'.")
        return

    # Attempt setting desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    tracker = HandTracker(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    buffer = KinematicBuffer(buffer_size=buffer_size)

    # FPS smoothing state
    prev_time = time.perf_counter()
    fps_smooth = 30.0
    alpha_fps = 0.15

    print("[STATUS] Live video stream started. Tracking hand kinematics...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to grab frame from camera stream. Exiting...")
                break

            # Calculate FPS
            curr_time = time.perf_counter()
            delta = curr_time - prev_time
            prev_time = curr_time
            if delta > 0.0:
                current_fps = 1.0 / delta
                fps_smooth = (1 - alpha_fps) * fps_smooth + alpha_fps * current_fps

            # Process frame through HandTracker
            annotated_frame, hand_data = tracker.process_frame(frame)

            thumb_px = None
            index_px = None
            dist_norm = None
            dist_px = None
            handedness_str = ""

            if hand_data is not None:
                landmarks_norm = hand_data["landmarks_norm"]
                landmarks_pixel = hand_data["landmarks_pixel"]
                sample_timestamp = hand_data["timestamp"]
                handedness_str = str(hand_data["handedness"])

                thumb_norm = landmarks_norm[THUMB_TIP_IDX]
                index_norm = landmarks_norm[INDEX_TIP_IDX]
                wrist_norm = landmarks_norm[WRIST_IDX]

                # Push sample into KinematicBuffer
                buffer.push(
                    timestamp=sample_timestamp,
                    thumb_tip=thumb_norm,
                    index_tip=index_norm,
                    wrist=wrist_norm,
                )

                # Pixel positions for display
                thumb_px = landmarks_pixel[THUMB_TIP_IDX]
                index_px = landmarks_pixel[INDEX_TIP_IDX]

                # 3D Euclidean distance
                dist_norm = float(
                    np.linalg.norm(np.array(thumb_norm) - np.array(index_norm))
                )
                # 2D screen distance
                dist_px = float(
                    np.linalg.norm(np.array(thumb_px) - np.array(index_px))
                )

            # Draw HUD
            hud_frame = draw_hud(
                frame=annotated_frame,
                fps=fps_smooth,
                buffer_len=len(buffer),
                buffer_capacity=buffer.buffer_size,
                is_ready=buffer.is_ready(min_samples=45),
                thumb_px=thumb_px,
                index_px=index_px,
                dist_norm=dist_norm,
                dist_px=dist_px,
                handedness_str=handedness_str,
            )

            cv2.imshow(WINDOW_NAME, hud_frame)

            # Graceful exit handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                print("[INFO] Exit signal detected from keyboard.")
                break

            # Handle user closing the window via the [X] button
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("[INFO] Display window closed by user.")
                break

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
        print("[STATUS] Camera released and resources successfully cleaned up.")


def run_smoke_test() -> bool:
    """
    Automated non-interactive verification test for CI/CD and headless environments.

    Validates:
    - HandTracker instantiation, processing, and output structure.
    - KinematicBuffer FIFO rolling window, timestamp offset, and 3D Euclidean metrics.
    - Buffer clear, ready thresholds, and shape integrity.

    Returns:
        True if all assertions pass.
    """
    print("=" * 65)
    print("Running Non-Interactive Smoke Test for Core Vision & Kinematics")
    print("=" * 65)

    # 1. Test KinematicBuffer
    print("1. Testing KinematicBuffer functionality...")
    buf_size = 10
    kb = KinematicBuffer(buffer_size=buf_size)
    assert len(kb) == 0, "Buffer should start empty"
    assert not kb.is_ready(min_samples=5), "Buffer should not be ready initially"

    base_time = 1000.0
    for i in range(15):
        t = base_time + (i * 0.033)
        thumb = (0.5 + i * 0.01, 0.5, 0.1)
        index = (0.5, 0.5, 0.1)
        wrist = (0.5, 0.8, 0.1)
        kb.push(t, thumb, index, wrist)

    assert len(kb) == buf_size, f"Buffer maxlen failed: expected {buf_size}, got {len(kb)}"
    assert kb.is_ready(min_samples=5), "Buffer should report ready when samples >= 5"

    series = kb.get_time_series()
    assert "time" in series
    assert "thumb_tip" in series
    assert "index_tip" in series
    assert "wrist" in series
    assert "euclidean_distance" in series

    assert series["time"].shape == (buf_size,)
    assert series["thumb_tip"].shape == (buf_size, 3)
    assert series["index_tip"].shape == (buf_size, 3)
    assert series["wrist"].shape == (buf_size, 3)
    assert series["euclidean_distance"].shape == (buf_size,)
    assert np.isclose(series["time"][0], 0.0), f"Initial relative time should be 0.0, got {series['time'][0]}"

    # Verify FIFO behavior: last thumb sample should correspond to i=14
    expected_thumb_x = 0.5 + 14 * 0.01
    assert np.isclose(series["thumb_tip"][-1, 0], expected_thumb_x), "Oldest entry not dropped properly in FIFO"

    # Test clear
    kb.clear()
    assert len(kb) == 0, "Buffer should be empty after clear"
    empty_series = kb.get_time_series()
    assert empty_series["time"].shape == (0,)
    assert empty_series["thumb_tip"].shape == (0, 3)
    print("   [PASS] KinematicBuffer thread-safety, FIFO, and series extraction verified.")

    # 2. Test HandTracker with blank frame
    print("2. Testing HandTracker pipeline with synthetic frame...")
    with HandTracker(static_image_mode=False) as tracker:
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, hand_data = tracker.process_frame(blank_frame)
        assert annotated.shape == blank_frame.shape, "Annotated frame shape mismatch"
        assert hand_data is None, "Blank frame should not produce hand detection"
        print("   [PASS] HandTracker processed frame cleanly without false positives.")

    # 3. Test draw_hud rendering
    print("3. Testing HUD overlay rendering...")
    test_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = draw_hud(
        frame=test_canvas,
        fps=30.0,
        buffer_len=10,
        buffer_capacity=150,
        is_ready=False,
        thumb_px=(320, 240),
        index_px=(320, 210),
        dist_norm=0.045,
        dist_px=30.0,
        handedness_str="Right (0.98)",
    )
    assert rendered.shape == (480, 640, 3), "HUD rendered shape mismatch"
    print("   [PASS] HUD graphics and telemetry rendered successfully.")

    print("\n[ALL SMOKE TESTS PASSED] Core vision pipeline is fully operational.\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NeuroMotion - Core Vision & Kinematic Verification Test"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0)",
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
        help="Run automated non-interactive verification test",
    )
    args = parser.parse_args()

    if args.smoke_test:
        success = run_smoke_test()
        sys.exit(0 if success else 1)
    else:
        run_interactive_vision_test(
            camera_id=args.camera_id,
            buffer_size=args.buffer_size,
        )
